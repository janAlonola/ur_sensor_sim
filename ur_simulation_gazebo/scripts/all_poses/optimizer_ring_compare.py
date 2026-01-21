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

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heatmaps", default="ur_sensor_sim/tmp/big_visibility_vars_patched",
                    help="Directory (or single YAML) of COMBINED heatmaps (big + ring already stacked).")
    ap.add_argument("--weights-list", nargs="+", default=["ur_sensor_sim/tmp/weighted_poses_tcp/*.npy"],
                    help="One or more .npy paths or globs (one per pose).")

    ap.add_argument("--objective", choices=["sum","softmin","frac"], default="sum")
    ap.add_argument("--softmin-temp", type=float, default=0.3)
    ap.add_argument("--frac-alpha", type=float, default=0.6)

    ap.add_argument("--k", type=int, default=25)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--rcl-size", type=int, default=5)
    ap.add_argument("--procs", type=int, default=None)
    ap.add_argument("--local-rounds", type=int, default=1000)
    ap.add_argument("--ls-sample-in", type=int, default=None)
    ap.add_argument("--normalize-weights", action="store_true")

    # NEW:
    ap.add_argument("--ring-map", default="ring_variants_index_map.json",
                    help="Path to ring_variants_index_map.json")
    ap.add_argument("--ring-base-index", type=int, default=2658,
                    help="Absolute start index of appended ring variants (local 0 maps to this).")
    ap.add_argument("--out-dir", default="ur_sensor_sim/tmp/ring_experiments_final",
                    help="Output folder for all results JSONs.")

    # how many top candidates per group to combine
    ap.add_argument("--top-per-group", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0,
                help="Base RNG seed (used only if --deterministic is set; otherwise each restart has its own seed).")
    ap.add_argument("--deterministic", action="store_true",
                help="If set, make runs repeatable by fixing RNG in each process.")

    return ap.parse_args()

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

def main():
    args = parse_args()

    # reproducibility (optional)
    if args.deterministic:
        random.seed(args.seed)
        np.random.seed(args.seed)

    # 1) Load A_list, V, S
    A_list, V, S, heatmap_files, P = load_all_pose_csrs(args.heatmaps)
    print(f"[INFO] Loaded CSRs: S={S}, V={V}, P={P}")

    # 2) Load weights W (V,P)
    W, weight_files = load_weight_npys_matrix(
        args.weights_list, V,
        normalize=args.normalize_weights,
        expect_P=P
    )
    print(f"[INFO] Loaded W: {W.shape} from {len(weight_files)} files")

    # 3) Load ring index map (RELATIVE indices)
    ring_map_path = args.ring_map
    ring_map = _load_ring_map(ring_map_path)
    print(f"[INFO] Loaded ring index map: {ring_map_path} with {len(ring_map)} rings")

    # 4) Map rel->abs
    base_offset = int(args.ring_base_index)
    print(f"[INFO] Using ring_base_index={base_offset} (rel=0 -> abs={base_offset})")

    def abs_inds(ring_name: str) -> list[int]:
        rel = ring_map[ring_name]["indices"]
        return [base_offset + int(i) for i in rel]

    # 5) Output dir
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output dir: {out_dir}")

    summary_csv = out_dir / "summary_rings.csv"
    summary_rows = []

    # 6) Run ONE experiment per ring
    ring_names = sorted(ring_map.keys())

    for ring_name in ring_names:
        fixed = sorted(set(abs_inds(ring_name)))

        # sanity checks
        if any((i < 0 or i >= S) for i in fixed):
            print(f"[WARN] Skipping {ring_name}: fixed indices out of range [0,{S-1}]")
            continue
        if len(fixed) > args.k:
            print(f"[WARN] Skipping {ring_name}: fixed_count={len(fixed)} > k={args.k}")
            continue

        print(f"\n=== [RING] {ring_name} | fixed={len(fixed)} | k={args.k} ===")

        sel, obj = parallel_multipose_grasp_pose_weights(
            A_list=A_list, W=W, k=args.k,
            iters=args.iters, rcl_size=args.rcl_size,
            mode=args.objective, softmin_temp=args.softmin_temp, frac_alpha=args.frac_alpha,
            procs=args.procs,
            iters_per_seed=1,
            local_rounds=args.local_rounds,
            ls_sample_in=args.ls_sample_in,
            fixed=fixed,
            verbose=False
        )

        out = {
            "name": ring_name,
            "status": "ok",
            "objective_mode": args.objective,
            "objective_value": float(obj),
            "selected_sensors": sel,
            "k": int(args.k),

            "fixed": fixed,
            "fixed_count": int(len(fixed)),

            "S": int(S), "V": int(V), "P": int(P),
            "heatmap_sources": heatmap_files,
            "weights_sources": weight_files,

            "normalize_weights": bool(args.normalize_weights),
            "iters": int(args.iters),
            "rcl_size": int(args.rcl_size),
            "local_rounds": int(args.local_rounds),
            "ls_sample_in": (None if args.ls_sample_in is None else int(args.ls_sample_in)),
            "ring_map": str(ring_map_path),
            "ring_base_index": int(base_offset),

            # note: the optimizer itself uses seeds = range(iters)
            "seed": int(args.seed),
            "deterministic": bool(args.deterministic),
        }

        out_path = out_dir / f"{ring_name}.json"
        out_path.write_text(json.dumps(out, indent=2))
        print(f"[OK] Saved: {out_path} | obj={obj:.3f}")

        summary_rows.append((ring_name, len(fixed), float(obj)))

    # 7) Write summary CSV
    with open(summary_csv, "w") as f:
        f.write("ring_name,fixed_count,objective\n")
        for ring_name, fixed_count, obj in sorted(summary_rows, key=lambda x: x[2], reverse=True):
            f.write(f"{ring_name},{fixed_count},{obj:.10f}\n")

    print(f"[OK] Wrote summary: {summary_csv}")


if __name__ == "__main__":
    main()
