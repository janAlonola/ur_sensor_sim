#!/usr/bin/env python3
"""
optimize_across_poses_multipose.py
Pick ONE set of sensors that works across MANY robot poses.

Key change vs your previous script:
- Builds ONE CSR PER POSE (A_list), then runs a multi-pose greedy/GRASP with an
  objective over poses:
    * sum      : sum over poses (pose-specific weights)
    * softmin  : robust/worst-case via soft minimum (controlled by --softmin-temp)
    * frac     : require coverage in ≥ alpha fraction of poses (--frac-alpha)

Inputs:
  --heatmaps     : a single heatmap YAML (with visible_by) OR a directory of them
                   Each YAML may store visible_by as:
                     - [pose][voxel] -> list[int]    (multi-pose file)
                     - [voxel]       -> list[int]    (single-pose file)
  --weights-list : one or more .npy paths or globs; one weight vector per pose
                   Aggregated into a single per-voxel weight vector.

Assumptions:
  - All heatmaps share the same voxel ordering (V) and sensor_count S.
  - Candidate sensor indices are consistent across files.
"""

import argparse, os, sys, time, json, glob, random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import yaml
try:
    from yaml import CSafeLoader as YLoader
except Exception:
    from yaml import SafeLoader as YLoader
from scipy.sparse import csr_matrix
from multiprocessing import Pool, cpu_count
from functools import partial

# ---------- Parallel multipose GRASP (pose-specific weights) ----------

# globals set once per worker to avoid re-pickling big matrices
_G_A_LIST = None   # list[csr_matrix]  (S x V) per pose
_G_W = None        # np.ndarray (V, P)

def _init_globals(A_list, W):
    global _G_A_LIST, _G_W
    _G_A_LIST = A_list
    _G_W = W

def _one_seed_worker(seed, k, rcl_size, mode, softmin_temp, frac_alpha, iters_per_seed, verbose,
                     local_rounds, ls_sample_in):
    sel, obj = multipose_grasp(
        A_list=_G_A_LIST, k=k, W=_G_W,
        iters=iters_per_seed, rcl_size=rcl_size,
        mode=mode, softmin_temp=softmin_temp, frac_alpha=frac_alpha,
        seed=seed, verbose=verbose,
        local_rounds=local_rounds, ls_sample_in=ls_sample_in
    )
    return (obj, sel, seed)

def parallel_multipose_grasp_pose_weights(A_list, W, k,
                                          iters=40, rcl_size=5,
                                          mode="sum", softmin_temp=0.3, frac_alpha=0.6,
                                          procs=None, iters_per_seed=1,
                                          local_rounds=100, ls_sample_in=None,
                                          verbose=False):
    """
    Run multipose_grasp_pose_weights in parallel: one (or few) restart(s) per seed.
    Returns: (best_sel, best_obj)
    """
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
        it = 0
        for obj, sel, seed in pool.imap_unordered(
            partial(_one_seed_worker,
                k=k, rcl_size=rcl_size, mode=mode,
                softmin_temp=softmin_temp, frac_alpha=frac_alpha,
                iters_per_seed=iters_per_seed,
                verbose=False, local_rounds=local_rounds, ls_sample_in=ls_sample_in),  # keep workers quiet; log in driver
            seeds,
            chunksize=max(1, iters // (procs * 4) or 1)
        ):
            it += 1
            if best is None or obj > best[0]:
                best = (obj, sel, seed)
            _status(it, best[0])

    print()  # newline for the progress row
    best_obj, best_sel, best_seed = best
    print(f"[GRASP] Best seed {best_seed}: obj={best_obj:.3f}, k={len(best_sel)}")
    return best_sel, best_obj

def build_pose_csrs_from_heatmap(yaml_path: Path):
    """Load a heatmap YAML that has visible_by as [P][V]->list[int] or [V]->list[int].
       Return (A_list, V, S) where A_list is list of SxV CSR matrices, one per pose."""
    raw = yaml.load(yaml_path.read_bytes(), Loader=YLoader)
    if "voxel_count" not in raw or "sensor_count" not in raw or "visible_by" not in raw:
        raise SystemExit(f"{yaml_path} missing required keys (voxel_count, sensor_count, visible_by)")

    V = int(raw["voxel_count"])
    S = int(raw["sensor_count"])
    vb = raw["visible_by"]

    # Normalize to [P][V] -> list[int]
    # Cases:
    #  - multi-pose: vb[p][v] = [sensor indices]
    #  - single-pose: vb[v] = [sensor indices]  -> wrap to [vb]
    if isinstance(vb, list) and vb and isinstance(vb[0], list) and (len(vb) == V or (vb and isinstance(vb[0][0], int))):
        vb = [vb]  # it was [V] -> wrap
    P = len(vb)

    A_list = []
    for p in range(P):
        # Build per-sensor voxel lists
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

        # Build CSR (S x V) with 1s where sensor sees voxel
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
    """Accept a single file or a directory of heatmaps and return a single A_list across ALL poses."""
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
                                    max_rounds=100, verbose=False, sample_in=None):
    """
    sel : list[int] current selected sensors
    rows: list[P][S] -> indices arrays (from extract_rows)
    V   : voxel count
    W   : (V,P) weight matrix (pose-specific weights)
    """
    P = len(rows)
    S = len(rows[0])
    sel = list(sel)
    sel_set = set(sel)

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

        out_list = list(sel_set)
        random.shuffle(out_list)

        for s_out in out_list:
            # base masks = selection minus s_out
            base_sel = [s for s in sel if s != s_out]
            base_masks = build_masks(base_sel)
            base_score = objective_from_masks(base_masks, W, mode, softmin_temp, frac_alpha)

            # candidates to add
            candidates_in = list(set(range(S)) - set(base_sel))
            if sample_in is not None and sample_in < len(candidates_in):
                # optional subsampling to speed up
                candidates_in = random.sample(candidates_in, sample_in)
            random.shuffle(candidates_in)

            best_pair = None
            best_gain = 0.0

            for s_in in candidates_in:
                # new masks = base OR row(s_in)
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
                # commit the swap
                sel_set.remove(s_out)
                sel_set.add(s_in)
                sel = sorted(sel_set)
                covered_p = new_masks
                if new_score > best_score + 1e-12:
                    best_score = new_score
                    improved = True
                    if verbose:
                        print(f"[1-swap] {s_out} → {s_in}  | score={best_score:.3f} | k={len(sel)}")
                break  # restart outer loop after improvement

    return sel, covered_p, best_score



# ---------------------------- Weight aggregation ----------------------------


def load_weight_npys_matrix(paths_or_globs, V, normalize=False, expect_P=None):
    """
    Return:
      W : (V, P) float64, pose-specific weights (column per pose)
      files : list[str] in the exact column order
    """
    # expand paths/globs
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

    W = np.hstack(cols)  # (V, P)
    if expect_P is not None and W.shape[1] != expect_P:
        raise SystemExit(f"Loaded {W.shape[1]} weight vectors but expected {expect_P} (poses).")
    return W, files


def load_and_aggregate_weight_npys(paths_or_globs, V, method="max", temperature=0.5, normalize=False):
    # Expand globs/paths
    if isinstance(paths_or_globs, str):
        candidates = glob.glob(paths_or_globs) or ([paths_or_globs] if Path(paths_or_globs).suffix.lower()==".npy" else [])
    else:
        candidates = []
        for item in paths_or_globs:
            hits = glob.glob(item)
            candidates.extend(hits if hits else ([item] if Path(item).suffix.lower()==".npy" else []))
    files = sorted({str(Path(p)) for p in candidates})
    if not files:
        raise SystemExit("No weight .npy files matched for --weights-list")

    mats = []
    for f in files:
        arr = np.load(f)
        if arr.shape != (V,):
            raise SystemExit(f"{f} shape {arr.shape} != (V,) = ({V},)")
        w = arr.astype(np.float64)
        if normalize:
            wmin, wmax = float(np.min(w)), float(np.max(w))
            w = (w - wmin) / (wmax - wmin) if wmax > wmin else np.zeros_like(w)
        mats.append(w[:, None])  # (V,1)

    W = np.hstack(mats)  # (V, P)

    if method == "max":
        agg = np.max(W, axis=1)
    elif method == "mean":
        agg = np.mean(W, axis=1)
    elif method == "softmax":
        tau = max(1e-6, float(temperature))
        Sft = np.exp(W / tau)
        P = Sft / np.clip(np.sum(Sft, axis=1, keepdims=True), 1e-12, None)
        agg = np.sum(P * W, axis=1)
    else:
        raise SystemExit(f"Unknown --weights-agg {method}")

    return agg, files, W.shape[1]

# ---------------------------- Multi-pose GRASP ----------------------------

def multipose_grasp(
    A_list, k, W,
    iters=1, rcl_size=5,
    mode="sum", softmin_temp=0.3, frac_alpha=0.6,
    seed=None, verbose=False,
    local_rounds=100, ls_sample_in=None):
    """
    - A_list : list of P csr_matrix (S x V)
    - W      : (V, P) weights per pose
    Returns (sel, obj)
    """
    if seed is not None:
        random.seed(seed); np.random.seed(seed)

    # preprocess
    rows, S, V, P = extract_rows(A_list)

    best_sel, best_obj = None, -1.0

    for rep in range(iters):
        # ---------- Construct (RCL greedy over pose-specific gain) ----------
        sel = []
        remaining = np.ones(S, dtype=bool)

        # maintain current masks for gain computation
        covered_p = np.zeros((P, V), dtype=bool)

        for step in range(k):
            gains = np.full(S, -np.inf, dtype=np.float32)
            avail = np.where(remaining)[0]
            if avail.size == 0:
                break

            # compute marginal gain of each available sensor
            for s in avail:
                # temp masks = covered_p OR row(s) for all poses
                # We can approximate gain cheaply for mode="sum" as the sum of W on newly covered voxels
                gain = 0.0
                if mode == "sum":
                    # fast path
                    for p in range(P):
                        idx = rows[p][s]
                        if idx.size == 0: continue
                        # newly covered = indices not yet covered
                        # sum W[idx, p] where not covered
                        if covered_p[p].any():
                            # mask check per index
                            newly = idx[~covered_p[p, idx]]
                            if newly.size:
                                gain += float(W[newly, p].sum())
                        else:
                            gain += float(W[idx, p].sum())
                else:
                    # generic path: actually OR and evaluate objective
                    tmp = covered_p.copy()
                    for p in range(P):
                        idx = rows[p][s]
                        if idx.size:
                            tmp[p, idx] = True
                    gain = objective_from_masks(tmp, W, mode, softmin_temp, frac_alpha) - \
                            objective_from_masks(covered_p, W, mode, softmin_temp, frac_alpha)

                gains[s] = gain

            if np.all(gains <= 0):
                break

            r = min(rcl_size, gains.size)
            idx = np.argpartition(-gains, kth=r-1)[:r]
            idx = idx[np.argsort(-gains[idx])]
            s_pick = int(random.choice(idx.tolist()))

            # commit
            sel.append(s_pick)
            remaining[s_pick] = False
            for p in range(P):
                idxp = rows[p][s_pick]
                if idxp.size:
                    covered_p[p, idxp] = True

            if verbose:
                print(f"[build] step {len(sel):02d}/{k} pick {s_pick} gain={gains[s_pick]:.3f}")

        # score before local search
        obj0 = objective_from_masks(covered_p, W, mode, softmin_temp, frac_alpha)

        # ---------- 1-swap Local Search ----------
        sel2, covered2, obj2 = local_search_one_swap_multipose(
            sel, rows, V, W,
            mode=mode, softmin_temp=softmin_temp, frac_alpha=frac_alpha,
            max_rounds=local_rounds, verbose=verbose, sample_in=ls_sample_in
        )

        # keep best
        if obj2 > best_obj + 1e-12:
            best_sel, best_obj = sel2, obj2
            if verbose:
                print(f"[rep {rep+1:02d}] improved by local search: {obj0:.3f} -> {obj2:.3f}")

    return best_sel, best_obj


# ---------------------------- CLI ----------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Optimize one sensor set over MANY poses (multi-pose objective).")
    ap.add_argument("--heatmaps", default="ur_sensor_sim/tmp/occlusion_heatmaps", help="Heatmap YAML (with visible_by) OR directory of such YAMLs")
    ap.add_argument("--weights-list", nargs="+", default=["ur_sensor_sim/tmp/weighted_poses_robot/*.npy"], help="One or more .npy paths or globs (one per pose).")
    # Die beiden Optionen hier sind für die Aggregation der Gewichte pro Pose zu einem einzigen Vektor pro Voxel (zurzeit unbenutzt)
    ap.add_argument("--weights-agg", choices=["max","mean","softmax"], default="max", help="Aggregate pose-specific weights into one per-voxel vector.")
    ap.add_argument("--weights-temp", type=float, default=0.5, help="Softmax temperature for --weights-agg=softmax.")
    ap.add_argument("--normalize-weights", action="store_true", help="Normalize each pose’s weight vector to [0,1] before aggregating.")

    ap.add_argument("--objective", choices=["sum","softmin","frac"], default="sum",
                    help=("Multi-pose objective over poses: "
                          "sum = sum_p sum_v W[v,p]*covered[p,v] "
                          "| softmin = soft minimum across poses (use --softmin-temp) "
                          "| frac = credit voxel if covered in ≥ alpha fraction of poses (use --frac-alpha)"))
    ap.add_argument("--softmin-temp", type=float, default=0.3,
                    help="Soft-min temperature for --objective=min (smaller -> closer to hard min).")
    ap.add_argument("--frac-alpha", type=float, default=0.6,
                    help="Required pose fraction for --objective=frac (e.g., 0.7 means ≥70% of poses).")

    ap.add_argument("--k", type=int, default=20, help="Sensor budget.")
    ap.add_argument("--iters", type=int, default=60, help="Greedy restarts.")
    ap.add_argument("--rcl-size", type=int, default=5, help="Restricted candidate list size.")
    ap.add_argument("--seed", type=int, default=0, help="Random seed.")
    ap.add_argument("--verbose", action="store_true", help="Verbose per-iter logs.")
    ap.add_argument("--export-json", type=str, default="ur_sensor_sim/tmp/result_better_weights.json",
                    help="Save results JSON here.")
    ap.add_argument("--procs", type=int, default=None, help="Processes for parallel seeds (default: CPU count)")

    ap.add_argument("--local-rounds", type=int, default=1000, help="1-swap local search rounds.")
    ap.add_argument("--ls-sample-in", type=int, default=None,
                    help="Optional subsample size for candidate 'add' set during 1-swap.")


    return ap.parse_args()

def main():
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed); np.random.seed(args.seed)

    # 1) Build per-pose CSRs across all heatmaps
    A_list, V, S, heatmap_files, P = load_all_pose_csrs(args.heatmaps)
    print(f"[INFO] Loaded CSRs: S={S}, V={V}, P={P}.")

    # 2) Pose-specific weights matrix W
    W, weight_files = load_weight_npys_matrix(args.weights_list, V, normalize=args.normalize_weights)
    if W.shape[1] != P:
        raise SystemExit(f"Weights pose count {W.shape[1]} != CSR poses {P}. "
                        "Make sure the .npy list/glob order matches the pose order.")
    print(f"[INFO] Loaded W: {W.shape} from {len(weight_files)} files.")

    # 3) Run optimizer with pose-specific weights
    sel, obj = parallel_multipose_grasp_pose_weights(
    A_list=A_list, W=W, k=args.k,
    iters=args.iters,
    rcl_size=args.rcl_size,
    mode=args.objective,
    softmin_temp=args.softmin_temp,
    frac_alpha=args.frac_alpha,
    procs=args.procs,
    iters_per_seed=1,
    local_rounds=args.local_rounds,
    ls_sample_in=args.ls_sample_in,
    verbose=False
    )

    # 4) Report
    print("\n========== RESULT (multi-pose) ==========")
    print(f"Objective: {args.objective}"
      + (f" (softmin_temp={args.softmin_temp})" if args.objective=='softmin' else "")
      + (f" (alpha={args.frac_alpha})" if args.objective=='frac' else ""))
    print(f"k={len(sel)} | selected sensors: {sel}")
    print(f"Objective value: {obj:.3f}")

    if args.export_json:
        out = {
            "selected_sensors": sel,
            "objective_mode": args.objective,
            "objective_value": float(obj),
            "S": S, "V": V, "P": P,
            "heatmap_sources": heatmap_files,
            "weights_sources": weight_files,
            "weights_agg": args.weights_agg,
            "weights_temp": args.weights_temp,
            "normalize_weights": bool(args.normalize_weights),
            "k": args.k, "iters": args.iters, "rcl_size": args.rcl_size,
            "seed": args.seed,
            "softmin_temp": args.softmin_temp,
            "frac_alpha": args.frac_alpha,
        }
        Path(args.export_json).write_text(json.dumps(out, indent=2))
        print(f"[OK] Saved JSON: {args.export_json}")

if __name__ == "__main__":
    main()
