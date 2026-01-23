#!/usr/bin/env python3
"""
optimizer_ring_compare.py

Extends optimizer_ring.py to:
- Fix *one* ring at a time (indices from ring_variants_index_map.json), optimize remaining sensors up to k.
- Save one JSON per ring experiment, named after the ring key.
- Then take top-3 per ring group (vertical/horizontal/upperarm) and test:
    - pairs across groups
    - triple across all 3 groups

IMPORTANT:
- ring_variants_index_map.json stores indices starting at 0 for the appended block.
  If your appended block starts at absolute candidate index 2658, use --ring-base-index 2658.
"""

import argparse, os, sys, time, json, glob, random
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import yaml
try:
    from yaml import CSafeLoader as YLoader
except Exception:
    from yaml import SafeLoader as YLoader
from scipy.sparse import csr_matrix
from multiprocessing import Pool, cpu_count
from functools import partial

# -------------------- code transer --------------------

def objective_from_masks(covered_p: np.ndarray, W: np.ndarray,
                         mode: str = "sum", softmin_temp: float = 0.3, frac_alpha: float = 0.6) -> float:
    """
    covered_p: (P, V) bool  -> per-pose coverage masks
    W        : (V, P) float -> pose-specific voxel weights
    modes:
      - "sum"  : sum_p sum_v W[v,p] * covered[p,v]
      - "softmin": soft-min across poses, then weight by mean weight per voxel
      - "frac" : credit voxel if covered in >= alpha fraction of poses, weight by mean weight
    """
    P, V = covered_p.shape
    assert W.shape == (V, P)
    WT = W.T  # (P, V) for convenient broadcasting

    if mode == "sum":
        return float((WT * covered_p.astype(np.float32)).sum())

    elif mode == "softmin":
        tau = max(1e-6, float(softmin_temp))
        # m[v] = number of poses that cover voxel v
        m = covered_p.sum(axis=0).astype(np.float32)           # (V,)
        a = (P - m)                                            # uncovered-poses count
        b = m                                                  # covered-poses count
        # soft-min value per voxel
        z = a + b * np.exp(-1.0 / tau)
        z = np.clip(z, 1e-12, None)                            # guard against log(0)
        f = -tau * np.log(z)                                   # (V,)
        wbar = W.mean(axis=1).astype(np.float32)               # (V,)
        return float((wbar * f).sum())

    elif mode == "frac":
        frac = covered_p.mean(axis=0)                # (V,)
        hit  = (frac >= float(frac_alpha)).astype(np.float32)
        wbar = np.mean(W, axis=1)                    # (V,)
        return float((wbar * hit).sum())

    else:
        raise ValueError("mode must be 'sum', 'softmin', or 'frac'")
    
def extract_rows(A_list):
    """
    A_list: list of P csr_matrix (S x V).
    Returns:
      rows[p][s] -> np.ndarray of covered voxel indices for sensor s in pose p
      S, V, P
    """
    P = len(A_list)
    S, V = A_list[0].shape
    rows = []
    for p in range(P):
        Ap = A_list[p]
        rp = [Ap.getrow(s).indices for s in range(S)]
        rows.append(rp)
    return rows, S, V, P

# ---------------------------- Local Search  ----------------------------


def local_search_one_swap_multipose(sel, rows, V, W,
                                    mode="sum", softmin_temp=0.3, frac_alpha=0.6,
                                    max_rounds=100, verbose=False, sample_in=None,
                                    fixed_set=None):
    P = len(rows)
    S = len(rows[0])
    sel = list(sel)
    sel_set = set(sel)
    fixed_set = set(fixed_set or [])

    def build_masks(selected):
        covered_p = np.zeros((P, V), dtype=bool)
        for p in range(P):
            for s in selected:
                idx = rows[p][s]
                if idx.size:
                    covered_p[p, idx] = True
        return covered_p

    covered_p = build_masks(sel)
    best_score = objective_from_masks(covered_p, W, mode, softmin_temp, frac_alpha)

    rounds = 0
    improved = True
    while improved and rounds < max_rounds:
        improved = False
        rounds += 1

        # never remove fixed sensors
        out_list = [s for s in sel_set if s not in fixed_set]
        if not out_list:
            break
        random.shuffle(out_list)

        for s_out in out_list:
            base_sel = [s for s in sel if s != s_out]
            base_masks = build_masks(base_sel)
            base_score = objective_from_masks(base_masks, W, mode, softmin_temp, frac_alpha)

            candidates_in = list(set(range(S)) - set(base_sel))
            if sample_in is not None and sample_in < len(candidates_in):
                candidates_in = random.sample(candidates_in, sample_in)
            random.shuffle(candidates_in)

            best_pair = None
            best_gain = 0.0

            for s_in in candidates_in:
                new_masks = base_masks.copy()
                for p in range(P):
                    idx = rows[p][s_in]
                    if idx.size:
                        new_masks[p, idx] = True
                new_score = objective_from_masks(new_masks, W, mode, softmin_temp, frac_alpha)
                gain = new_score - base_score
                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best_pair = (s_out, s_in, new_masks, new_score)

            if best_pair is not None:
                s_out, s_in, new_masks, new_score = best_pair
                sel_set.remove(s_out)
                sel_set.add(s_in)
                sel = sorted(sel_set)
                covered_p = new_masks
                if new_score > best_score + 1e-12:
                    best_score = new_score
                    improved = True
                    if verbose:
                        print(f"[1-swap] {s_out} → {s_in}  | score={best_score:.3f} | k={len(sel)}")
                break

    return sel, covered_p, best_score
# ---------------------------- Multi-pose GRASP ----------------------------

def multipose_grasp(
    A_list, k, W,
    iters=1, rcl_size=5,
    mode="sum", softmin_temp=0.3, frac_alpha=0.6,
    seed=None, verbose=False,
    local_rounds=100, ls_sample_in=None,
    fixed=None
):
    if seed is not None:
        random.seed(seed); np.random.seed(seed)

    rows, S, V, P = extract_rows(A_list)

    fixed = sorted(set(fixed or []))
    fixed_set = set(fixed)
    if len(fixed) > k:
        raise ValueError(f"fixed sensors ({len(fixed)}) exceed budget k={k}")
    if any((f < 0 or f >= S) for f in fixed):
        raise ValueError("fixed contains sensor indices outside [0,S-1]")

    best_sel, best_obj = None, -1.0

    for rep in range(iters):
        sel = list(fixed)
        remaining = np.ones(S, dtype=bool)
        remaining[fixed] = False

        covered_p = np.zeros((P, V), dtype=bool)
        for s_fix in fixed:
            for p in range(P):
                idx = rows[p][s_fix]
                if idx.size:
                    covered_p[p, idx] = True

        for step in range(len(sel), k):
            gains = np.full(S, -np.inf, dtype=np.float32)
            avail = np.where(remaining)[0]
            if avail.size == 0:
                break

            if mode != "sum":
                base_obj = objective_from_masks(covered_p, W, mode, softmin_temp, frac_alpha)

            for s in avail:
                gain = 0.0
                if mode == "sum":
                    for p in range(P):
                        idx = rows[p][s]
                        if idx.size == 0:
                            continue
                        newly = idx[~covered_p[p, idx]]
                        if newly.size:
                            gain += float(W[newly, p].sum())
                else:
                    tmp = covered_p.copy()
                    for p in range(P):
                        idx = rows[p][s]
                        if idx.size:
                            tmp[p, idx] = True
                    gain = objective_from_masks(tmp, W, mode, softmin_temp, frac_alpha) - base_obj

                gains[s] = gain

            if np.all(gains[avail] <= 0):
                break

            r = min(rcl_size, avail.size)
            top = avail[np.argpartition(-gains[avail], kth=r-1)[:r]]
            top = top[np.argsort(-gains[top])]
            s_pick = int(random.choice(top.tolist()))

            sel.append(s_pick)
            remaining[s_pick] = False
            for p in range(P):
                idxp = rows[p][s_pick]
                if idxp.size:
                    covered_p[p, idxp] = True

            if verbose:
                print(f"[build] step {len(sel):02d}/{k} pick {s_pick} gain={gains[s_pick]:.3f}")

        obj0 = objective_from_masks(covered_p, W, mode, softmin_temp, frac_alpha)

        sel2, covered2, obj2 = local_search_one_swap_multipose(
            sel, rows, V, W,
            mode=mode, softmin_temp=softmin_temp, frac_alpha=frac_alpha,
            max_rounds=local_rounds, verbose=verbose, sample_in=ls_sample_in,
            fixed_set=fixed_set
        )

        if obj2 > best_obj + 1e-12:
            best_sel, best_obj = sel2, obj2

    return best_sel, best_obj

# ---------- Parallel multipose GRASP (pose-specific weights) ----------
_G_A_LIST = None
_G_W = None

def _init_globals(A_list, W):
    global _G_A_LIST, _G_W
    _G_A_LIST = A_list
    _G_W = W

def _one_seed_worker(seed, k, rcl_size, mode, softmin_temp, frac_alpha, iters_per_seed,
                     verbose, local_rounds, ls_sample_in, fixed):
    sel, obj = multipose_grasp(
        A_list=_G_A_LIST, k=k, W=_G_W,
        iters=iters_per_seed, rcl_size=rcl_size,
        mode=mode, softmin_temp=softmin_temp, frac_alpha=frac_alpha,
        seed=seed, verbose=verbose,
        local_rounds=local_rounds, ls_sample_in=ls_sample_in,
        fixed=fixed
    )
    return (obj, sel, seed)

def parallel_multipose_grasp_pose_weights(A_list, W, k,
                                          iters=40, rcl_size=5,
                                          mode="sum", softmin_temp=0.3, frac_alpha=0.6,
                                          procs=None, iters_per_seed=1,
                                          local_rounds=100, ls_sample_in=None,
                                          fixed=None,
                                          verbose=False):
    procs = procs or cpu_count()
    seeds = list(range(iters))
    best = None
    t0 = time.time()

    def _status(done, best_obj):
        elapsed = time.time() - t0
        eta = elapsed / done * (iters - done) if done else 0.0
        sys.stdout.write(f"\r[GRASP] {done}/{iters} | best={best_obj:.3f} | elapsed={elapsed:.1f}s | ETA~{eta:.1f}s")
        sys.stdout.flush()

    with Pool(processes=procs, initializer=_init_globals, initargs=(A_list, W)) as pool:
        done = 0
        for obj, sel, seed in pool.imap_unordered(
            partial(_one_seed_worker,
                    k=k, rcl_size=rcl_size, mode=mode,
                    softmin_temp=softmin_temp, frac_alpha=frac_alpha,
                    iters_per_seed=iters_per_seed,
                    verbose=False, local_rounds=local_rounds, ls_sample_in=ls_sample_in,
                    fixed=fixed or []),
            seeds,
            chunksize=max(1, iters // (procs * 4) or 1)
        ):
            done += 1
            if best is None or obj > best[0]:
                best = (obj, sel, seed)
            _status(done, best[0])

    print()
    best_obj, best_sel, best_seed = best
    print(f"[GRASP] Best seed {best_seed}: obj={best_obj:.3f}, k={len(best_sel)}")
    return best_sel, best_obj

# --------- heatmap loading (unchanged) ---------

def build_pose_csrs_from_heatmap(yaml_path: Path):
    raw = yaml.load(yaml_path.read_bytes(), Loader=YLoader)
    if "voxel_count" not in raw or "sensor_count" not in raw or "visible_by" not in raw:
        raise SystemExit(f"{yaml_path} missing required keys (voxel_count, sensor_count, visible_by)")

    V = int(raw["voxel_count"])
    S = int(raw["sensor_count"])
    vb = raw["visible_by"]

    if isinstance(vb, list) and vb and isinstance(vb[0], list) and (len(vb) == V or (vb and isinstance(vb[0][0], int))):
        vb = [vb]
    P = len(vb)

    A_list = []
    for p in range(P):
        sens_vox = [[] for _ in range(S)]
        vox_lists = vb[p]
        if len(vox_lists) != V:
            raise SystemExit(f"{yaml_path}: pose {p} has {len(vox_lists)} voxels != V={V}")
        for v, sens_list in enumerate(vox_lists):
            for s in sens_list:
                si = int(s)
                if 0 <= si < S:
                    sens_vox[si].append(v)
                else:
                    raise SystemExit(f"{yaml_path}: pose {p}, voxel {v}: sensor index {s} out of [0,{S-1}]")

        indptr = [0]
        indices = []
        for s in range(S):
            vs = sorted(sens_vox[s])
            indices.extend(vs)
            indptr.append(len(indices))
        data = np.ones(len(indices), dtype=np.float32)
        A = csr_matrix((data, np.array(indices, dtype=np.int32), np.array(indptr, dtype=np.int32)),
                       shape=(S, V), dtype=np.float32)
        A_list.append(A)
    return A_list, V, S

def load_all_pose_csrs(heatmaps_path: str):
    src = Path(heatmaps_path)
    files = [src] if src.is_file() else sorted([p for p in src.iterdir() if p.suffix.lower() in (".yaml",".yml")])
    if not files:
        raise SystemExit(f"No heatmap YAMLs found in {src}")

    A_list_total = []
    V = S = None
    total_poses = 0
    for f in files:
        A_list, Vp, Sp = build_pose_csrs_from_heatmap(f)
        if V is None:
            V, S = Vp, Sp
        else:
            if Vp != V: raise SystemExit(f"{f}: voxel_count {Vp} != {V}")
            if Sp != S: raise SystemExit(f"{f}: sensor_count {Sp} != {S}")
        A_list_total.extend(A_list)
        total_poses += len(A_list)

    return A_list_total, V, S, [str(p) for p in files], total_poses

# --------- weights loading (unchanged) ---------

def load_weight_npys_matrix(paths_or_globs, V, normalize=False, expect_P=None):
    if isinstance(paths_or_globs, str):
        cands = glob.glob(paths_or_globs) or ([paths_or_globs] if Path(paths_or_globs).suffix==".npy" else [])
    else:
        cands = []
        for pat in paths_or_globs:
            hits = glob.glob(pat)
            if hits:
                cands.extend(hits)
            elif Path(pat).suffix == ".npy":
                cands.append(pat)

    files = sorted({str(Path(p)) for p in cands})
    if not files:
        raise SystemExit("No weight .npy files matched for --weights-list")

    cols = []
    for f in files:
        arr = np.load(f)
        if arr.shape != (V,):
            raise SystemExit(f"{f} has shape {arr.shape} != (V,) = ({V},)")
        w = arr.astype(np.float64)
        if normalize:
            wmin, wmax = float(np.min(w)), float(np.max(w))
            w = (w - wmin) / (wmax - wmin) if wmax > wmin else np.zeros_like(w)
        cols.append(w[:, None])

    W = np.hstack(cols)
    if expect_P is not None and W.shape[1] != expect_P:
        raise SystemExit(f"Loaded {W.shape[1]} weight vectors but expected {expect_P} (poses).")
    return W, files

def remap_weights_piecewise(W: np.ndarray,
                            t: float = 0.5,
                            low_anchor: float = 0.1,
                            low_target: float = 0.01,
                            high_target_at_1: float = 2.0) -> np.ndarray:
    """
    Piecewise remap on [0,1]:
      - for w <= t: power-law squash that keeps t fixed and maps low_anchor -> low_target
      - for w >  t: linear stretch that maps 1 -> high_target_at_1 and keeps t fixed

    Anchors (default):
      1 -> 2
      0.5 -> 0.5
      0.1 -> 0.01
    """
    W = W.astype(np.float64, copy=True)

    # (Optional) clamp to [0,1] if you want to be safe
    W = np.clip(W, 0.0, 1.0)

    # Solve p from: low_target = (low_anchor^p) / (t^(p-1))
    # => p = log(low_target * t / t) ??? (derived carefully)
    # low_target = low_anchor^p / t^(p-1) = t * (low_anchor/t)^p
    # => low_target / t = (low_anchor/t)^p
    # => p = log(low_target/t) / log(low_anchor/t)
    p = np.log(low_target / t) / np.log(low_anchor / t)

    W2 = np.empty_like(W)

    mask_low = (W <= t)
    mask_high = ~mask_low

    # low side: squash
    # w' = w^p / t^(p-1)
    W2[mask_low] = (W[mask_low] ** p) / (t ** (p - 1.0))

    # high side: linear stretch, keep continuity at t
    # slope so that: w'=t at w=t and w'=high_target_at_1 at w=1
    slope = (high_target_at_1 - t) / (1.0 - t)
    W2[mask_high] = t + slope * (W[mask_high] - t)

    return W2

# -------------------- NEW: ring experiment orchestration --------------------

def load_ring_map(path: str) -> dict:
    return json.loads(Path(path).read_text())

def ring_group_from_name(name: str) -> str:
    # Based on your keys in ring_variants_index_map.json
    if "forearm_vertical" in name:
        return "vertical"
    if "forearm_horizontal" in name:
        return "horizontal"
    if "upperarm" in name:
        return "upperarm"
    return "other"

def to_absolute_indices(entry: dict, ring_base_index: int) -> List[int]:
    # JSON stores local indices (0 == appended start). You said: local 0 -> abs 2658.
    local = entry["indices"]
    return [ring_base_index + int(i) for i in local]

def run_one_experiment(*, name: str, fixed: List[int], k: int,
                       A_list, W, args,
                       heatmap_files: List[str], weight_files: List[str],
                       out_dir: Path) -> dict:
    fixed = sorted(set(fixed))
    if len(fixed) > k:
        return {
            "name": name,
            "status": "skipped_fixed_exceeds_k",
            "k": k,
            "fixed_count": len(fixed),
            "fixed": fixed,
        }

    sel, obj = parallel_multipose_grasp_pose_weights(
        A_list=A_list, W=W, k=k,
        iters=args.iters, rcl_size=args.rcl_size,
        mode=args.objective, softmin_temp=args.softmin_temp, frac_alpha=args.frac_alpha,
        procs=args.procs,
        iters_per_seed=1,
        local_rounds=args.local_rounds,
        ls_sample_in=args.ls_sample_in,
        fixed=fixed,
        verbose=False
    )

    res = {
        "name": name,
        "status": "ok",
        "objective_mode": args.objective,
        "objective_value": float(obj),
        "selected_sensors": sel,
        "k": int(k),
        "fixed": fixed,
        "fixed_count": int(len(fixed)),
        "S": int(A_list[0].shape[0]),
        "V": int(A_list[0].shape[1]),
        "P": int(len(A_list)),
        "heatmap_sources": heatmap_files,
        "weights_sources": weight_files,
        "normalize_weights": bool(args.normalize_weights),
        "iters": int(args.iters),
        "rcl_size": int(args.rcl_size),
        "local_rounds": int(args.local_rounds),
        "ls_sample_in": (None if args.ls_sample_in is None else int(args.ls_sample_in)),
        "seed_policy": "seeds = range(iters) (deterministic unless you change it)",
    }

    out_path = out_dir / f"{name}.json"
    out_path.write_text(json.dumps(res, indent=2))
    print(f"[OK] wrote {out_path}")
    return res

def top_n_by_group(results: List[dict], n: int = 3) -> Dict[str, List[dict]]:
    ok = [r for r in results if r.get("status") == "ok"]
    groups: Dict[str, List[dict]] = {}
    for r in ok:
        g = ring_group_from_name(r["name"])
        groups.setdefault(g, []).append(r)

    for g in groups:
        groups[g].sort(key=lambda x: x["objective_value"], reverse=True)
        groups[g] = groups[g][:n]
    return groups

def combine_fixed(*lists: List[int]) -> List[int]:
    out = set()
    for L in lists:
        out |= set(L)
    return sorted(out)

# ---------------------------- CLI ----------------------------

def _load_ring_map(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)

def _infer_appended_offset(S_total: int, ring_map: dict) -> int:
    """
    ring_map indices are RELATIVE to the appended block:
      rel=0 means absolute index = base_offset + 0
    We infer base_offset = S_total - appended_count, where appended_count = max_rel+1.
    """
    max_rel = -1
    for meta in ring_map.values():
        inds = meta.get("indices", [])
        if inds:
            max_rel = max(max_rel, max(inds))
    appended_count = max_rel + 1
    if appended_count <= 0 or appended_count > S_total:
        raise ValueError(f"Could not infer appended_count (got {appended_count}) from ring map.")
    return S_total - appended_count

def parse_args():
    ap = argparse.ArgumentParser(description="Exhaustive evaluation of 3-ring combinations (rings only).")

    ap.add_argument("--heatmaps", default="ur_sensor_sim/tmp/big_visibility_vars",
                    help="Directory (or single YAML) of COMBINED heatmaps (big + appended ring variants).")
    ap.add_argument("--weights-list", nargs="+", default=["ur_sensor_sim/tmp/weighted_poses_tcp/*.npy"],
                    help="One or more .npy paths or globs (one per pose).")
    ap.add_argument("--normalize-weights", action="store_true")

    ap.add_argument("--objective", choices=["sum","softmin","frac"], default="sum")
    ap.add_argument("--softmin-temp", type=float, default=0.3)
    ap.add_argument("--frac-alpha", type=float, default=0.6)

    ap.add_argument("--k", type=int, default=24, help="Total sensor budget (must equal total sensors in the chosen 3 rings).")

    ap.add_argument("--ring-map", default="ring_variants_index_map.json",
                    help="Path to ring_variants_index_map.json (stores LOCAL indices).")
    ap.add_argument("--ring-base-index", type=int, default=2658,
                    help="Absolute start index of appended ring variants (local 0 maps to this).")

    ap.add_argument("--out-dir", default="ur_sensor_sim/tmp/test",
                    help="Output folder for results.")

    ap.add_argument("--top-n", type=int, default=100,
                    help="How many top combinations to store in CSV/JSON.")

    return ap.parse_args()


def main():
    args = parse_args()

    # 1) Load A_list (P CSRs), dimensions
    A_list, V, S, heatmap_files, P = load_all_pose_csrs(args.heatmaps)
    print(f"[INFO] Loaded CSRs: S={S}, V={V}, P={P}")

    # 2) Load pose-specific weights W (V,P)
    W, weight_files = load_weight_npys_matrix(
        args.weights_list, V,
        normalize=args.normalize_weights,
        expect_P=P
    )
    # --- NEW: remap weights to emphasize high-weight voxels and suppress low-weight voxels ---
    #W = remap_weights_piecewise(W)
    #print(f"[INFO] Remapped weights: min={W.min():.4f}, max={W.max():.4f}")
    #print(f"[INFO] Loaded W: {W.shape} from {len(weight_files)} files")

    # 3) Load ring map + convert to absolute indices
    ring_map = _load_ring_map(args.ring_map)
    base_offset = int(args.ring_base_index)
    print(f"[INFO] Loaded ring map: {args.ring_map} with {len(ring_map)} rings")
    print(f"[INFO] ring-base-index: {base_offset} (local 0 -> abs {base_offset})")

    ring_names = sorted(ring_map.keys())
    rings_abs = {}
    ring_sizes = {}

    for name in ring_names:
        local = ring_map[name].get("indices", [])
        abs_inds = [base_offset + int(i) for i in local]
        # Basic safety checks
        if any((sidx < 0 or sidx >= S) for sidx in abs_inds):
            raise SystemExit(f"[ERROR] Ring '{name}' has abs indices outside [0,{S-1}]")
        rings_abs[name] = abs_inds
        ring_sizes[name] = len(abs_inds)

    # 4) Pre-extract rows for fast coverage OR-ing
    rows, S2, V2, P2 = extract_rows(A_list)
    assert S2 == S and V2 == V and P2 == P

    # 5) Precompute per-ring coverage masks (P,V) bool
    #    This makes the triple-combo loop fast.
    print("[INFO] Precomputing per-ring coverage masks ...")
    ring_cov = {}  # name -> (P,V) bool
    for name in ring_names:
        cov = np.zeros((P, V), dtype=bool)
        for sidx in rings_abs[name]:
            for p in range(P):
                idx = rows[p][sidx]
                if idx.size:
                    cov[p, idx] = True
        ring_cov[name] = cov

    print("\n[DEBUG] Per-ring weighted coverage (sum objective):")
    for name in ring_names:
        cov = ring_cov[name]              # (P,V) bool
        total = float((W.T * cov).sum())  # W.T is (P,V)
        print(f"{name}\t{total:.10f}")

    print("\n[DEBUG] Checking upperarm_ring abs index ranges...")
    for name in ring_names:
        if name.startswith("upperarm_ring_"):
            abs_inds = rings_abs[name]
            print(name, "size=", len(abs_inds),
                "min=", min(abs_inds), "max=", max(abs_inds), "S=", S)
            
    print("\n[DEBUG] Do upperarm_ring sensors see any voxels?")
    for name in ring_names:
        if name.startswith("upperarm_ring_"):
            tot = 0
            for sidx in rings_abs[name]:
                for p in range(P):
                    tot += rows[p][sidx].size
            print(name, "total_visible_entries=", tot)

    test_name = "upperarm_ring_tilt_all_0"
    test_idx = rings_abs[test_name][0]

    print("\n[DEBUG] Row nnz for", test_name, "sensor", test_idx)
    for p in range(P):
        nnz = A_list[p].getrow(test_idx).nnz
        if nnz:
            print("pose", p, "nnz", nnz)
            break
    else:
        print("ALL poses have nnz=0 for that sensor row.")

    name = "upperarm_ring_tilt_all_0"
    cov = ring_cov[name]            # (P,V)
    covered_any = cov.any(axis=0)   # (V,)
    print("[DEBUG] covered voxels:", int(covered_any.sum()))
    print("[DEBUG] mean weight on covered voxels:", float(W[covered_any].mean()))
    print("[DEBUG] max weight on covered voxels:", float(W[covered_any].max()))

    # 6) Exhaustive search over all triples
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best = None  # (obj, (a,b,c), fixed_count)
    topN = []    # list of dicts (kept sorted by obj desc, max len = args.top_n)

    names = ring_names
    n = len(names)

    checked = 0
    skipped_k = 0

    def _push_top(entry):
        topN.append(entry)
        topN.sort(key=lambda x: x["objective_value"], reverse=True)
        if len(topN) > int(args.top_n):
            topN.pop()

    print(f"[INFO] Evaluating all 3-ring combinations from {n} rings "
          f"(~{n*(n-1)*(n-2)//6} triples), keeping k={args.k} ...")

    t0 = time.time()
    for i in range(n):
        a = names[i]
        for j in range(i+1, n):
            b = names[j]
            for k in range(j+1, n):
                c = names[k]

                fixed_count = ring_sizes[a] + ring_sizes[b] + ring_sizes[c]
                if fixed_count != args.k:
                    skipped_k += 1
                    print("skipped k")
                    continue

                # OR coverage
                covered = ring_cov[a] | ring_cov[b] | ring_cov[c]
                obj = objective_from_masks(
                    covered, W,
                    mode=args.objective,
                    softmin_temp=args.softmin_temp,
                    frac_alpha=args.frac_alpha
                )

                checked += 1
                if (best is None) or (obj > best[0]):
                    best = (obj, (a, b, c), fixed_count)
                    elapsed = time.time() - t0
                    print(f"\n[BEST] obj={obj:.3f} | rings={a}, {b}, {c} | fixed={fixed_count} "
                          f"| checked={checked} | elapsed={elapsed:.1f}s")

                _push_top({
                    "objective_value": float(obj),
                    "rings": [a, b, c],
                    "fixed_count": int(fixed_count),
                    "fixed_indices": sorted(set(rings_abs[a] + rings_abs[b] + rings_abs[c])),
                })

        # lightweight progress per outer loop
        if (i % 5) == 0 and i > 0:
            elapsed = time.time() - t0
            print(f"[PROGRESS] i={i}/{n} | checked={checked} | skipped_k={skipped_k} | elapsed={elapsed:.1f}s")

    if best is None:
        raise SystemExit(f"[ERROR] No triple matched k={args.k}. "
                         f"Check ring sizes; maybe many rings are not size 8?")

    best_obj, (ra, rb, rc), fixed_count = best
    best_fixed = sorted(set(rings_abs[ra] + rings_abs[rb] + rings_abs[rc]))

    # 7) Write outputs
    best_json = {
        "status": "ok",
        "objective_mode": args.objective,
        "objective_value": float(best_obj),
        "rings": [ra, rb, rc],
        "k": int(args.k),
        "fixed_count": int(fixed_count),
        "fixed_indices": best_fixed,
        "S": int(S), "V": int(V), "P": int(P),
        "heatmap_sources": heatmap_files,
        "weights_sources": weight_files,
        "normalize_weights": bool(args.normalize_weights),
        "notes": "Exhaustive search over 3-ring combinations; no GRASP/random sensors.",
    }
    (out_dir / "best_3rings.json").write_text(json.dumps(best_json, indent=2))
    (out_dir / "top_3rings.json").write_text(json.dumps(topN, indent=2))

    # CSV (easy to paste/open in Excel)
    with open(out_dir / "top_3rings.csv", "w") as f:
        f.write("rank,objective_value,ring_a,ring_b,ring_c,fixed_count\n")
        for r, entry in enumerate(topN, start=1):
            a, b, c = entry["rings"]
            f.write(f"{r},{entry['objective_value']:.10f},{a},{b},{c},{entry['fixed_count']}\n")

    elapsed = time.time() - t0
    print("\n========== DONE ==========")
    print(f"[OK] Best: obj={best_obj:.3f} | rings={ra}, {rb}, {rc} | fixed={fixed_count}")
    print(f"[OK] Checked={checked} triples (skipped_k={skipped_k}) | elapsed={elapsed:.1f}s")
    print(f"[OK] Wrote: {out_dir / 'best_3rings.json'}")
    print(f"[OK] Wrote: {out_dir / 'top_3rings.csv'}")
    print(f"[OK] Wrote: {out_dir / 'top_3rings.json'}")


if __name__ == "__main__":
    main()

