#!/usr/bin/env python3
"""
GRASP (Greedy Randomized Adaptive Search Procedure) for Sensor Selection
----------------------------------------------------------------------

Now supports WEIGHTED coverage:
- If --weights <.npy> is provided, those weights are used.
- Else, if the YAML contains a per-voxel array under --yaml-weight-key (default: 'weights'),
  those are used.
- Else, falls back to unweighted coverage.

Author: you + ChatGPT
"""

import argparse
import json
import math
import sys
import os
import random
from pathlib import Path
from typing import List, Set, Tuple, Optional

import numpy as np
import yaml
try:
    from yaml import CSafeLoader as YLoader  # C-accelerated SafeLoader
    print("Using CSafeLoader for YAML.")
except Exception:
    from yaml import SafeLoader as YLoader   # fallback
    print("Using SafeLoader for YAML.")
import time

from multiprocessing import Pool, cpu_count
from functools import partial
from scipy.sparse import csr_matrix

def build_sensor_csr(sensor_sets, V) -> csr_matrix:
    """Build SxV boolean CSR from list[set[int]]."""
    S = len(sensor_sets)
    indptr = [0]
    indices = []
    for s in range(S):
        cols = sorted(sensor_sets[s])
        indices.extend(cols)
        indptr.append(len(indices))
    data = np.ones(len(indices), dtype=np.float32)  # boolean-as-float for matvec
    return csr_matrix((data, indices, np.array(indptr, dtype=np.int32)), shape=(S, V))

def grasp_max_coverage_sparse(A: csr_matrix,
                              universe_size: int,
                              k: int,
                              rcl_size: int = 5,
                              iters: int = 20,
                              weights: Optional[np.ndarray] = None,
                              local_search_rounds: int = 100,
                              early_stop_no_improve: Optional[int] = None,
                              verbose: bool = True,
                              seed: Optional[int] = None):
    """
    Same API as your original, but construction uses sparse matvec instead of Python sets.
    """
    if seed is not None:
        random.seed(seed); np.random.seed(seed)

    S, V = A.shape
    assert V == universe_size
    ones = (weights is None)
    wvec = np.ones(V, dtype=np.float32) if ones else weights.astype(np.float32)

    best_selected, best_covered_mask = [], np.zeros(V, dtype=bool)
    best_score = -1.0
    no_improve = 0

    for it in range(1, iters + 1):
        covered_mask = np.zeros(V, dtype=bool)
        selected = []
        remaining = np.ones(S, dtype=bool)  # True if sensor still available

        for step in range(k):
            # available weight per voxel
            avail = (~covered_mask).astype(np.float32) * wvec
            # gains for all sensors at once
            gains = A.dot(avail)  # shape (S,)
            gains[~remaining] = -1.0  # mask out already selected

            # build RCL
            # take top rcl_size positive gains
            if np.all(gains <= 0):
                break
            idx_sorted = np.argpartition(-gains, kth=min(rcl_size-1, gains.size-1))[:rcl_size]
            idx_sorted = idx_sorted[np.argsort(-gains[idx_sorted])]
            s_pick = int(random.choice(idx_sorted.tolist()))
            gain = float(gains[s_pick])

            # update covered & remaining
            row = A.getrow(s_pick)
            covered_mask[row.indices] = True
            remaining[s_pick] = False
            selected.append(s_pick)

            if verbose:
                if ones:
                    # approximate newly covered count:
                    new_cov = row.indices[~covered_mask[row.indices]].size  # tiny undercount due to update order
                    print(f"[it {it:02d} | step {step+1:02d}] pick {s_pick} +≈{new_cov} (cum {covered_mask.sum()}/{V})")
                else:
                    print(f"[it {it:02d} | step {step+1:02d}] pick {s_pick} +w{gain:.3f} (cum_w {A.dot((~covered_mask)*0 + wvec * covered_mask).sum():.3f})")

            if covered_mask.all():
                break

        # score the result
        if ones:
            score2 = float(covered_mask.sum())
        else:
            score2 = float(wvec[covered_mask].sum())

        if score2 > best_score + 1e-12:
            best_score = score2
            best_selected = selected[:]
            best_covered_mask = covered_mask.copy()
            no_improve = 0
            if verbose:
                if ones:
                    print(f"[it {it:02d}] New best (unweighted): {best_covered_mask.sum()}/{V}, k={len(best_selected)}")
                else:
                    print(f"[it {it:02d}] New best (weighted): score={best_score:.3f}, "
                          f"covered={best_covered_mask.sum()}/{V}, k={len(best_selected)}")
        else:
            no_improve += 1
        if early_stop_no_improve is not None and no_improve >= early_stop_no_improve:
            if verbose:
                print(f"[it {it:02d}] Early stop after {no_improve} non-improving iterations.")
            break

    # convert mask back to set of indices if you need it elsewhere
    best_covered = set(np.nonzero(best_covered_mask)[0].tolist())
    return best_selected, best_covered, best_score


# ---------------------------- Data loading ----------------------------

def load_visible_by(yaml_path: str):
    """
    Returns:
      sensor_sets: list[set[int]] where sensor_sets[j] is the set of voxels covered by sensor j (union across poses)
      V: int (voxel_count)
      raw: dict (full parsed YAML for optional fields like weights)
    """
    print(f"Loading visible_by from {yaml_path}...")
    raw = yaml.load(Path(yaml_path).read_bytes(), Loader=YLoader)
    print("loaded YAML.")
    V = int(raw["voxel_count"])
    P = int(raw["pose_count"])
    S = int(raw["sensor_count"])
    vis_by = raw["visible_by"]  # shape [P][V] -> list[int]

    sensor_sets = [set() for _ in range(S)]
    for p in range(P):
        vox_lists = vis_by[p]
        for v, sens_list in enumerate(vox_lists):
            for s in sens_list:
                sensor_sets[int(s)].add(int(v))

    return sensor_sets, V, raw


# ---------------------------- Utilities ----------------------------

def weighted_size(indices: Set[int], weights: Optional[np.ndarray]) -> float:
    if not indices:
        return 0.0
    if weights is None:
        return float(len(indices))
    # sum of weights for voxel indices
    return float(np.sum(weights[list(indices)]))


def union_sets(indices: List[int], sensor_sets: List[Set[int]]) -> Set[int]:
    out = set()
    for i in indices:
        out |= sensor_sets[i]
    return out


# ---------------------------- Local search (1-swap) ----------------------------

def one_swap_local_search(sensor_sets: List[Set[int]],
                          selected: List[int],
                          universe_size: int,
                          weights: Optional[np.ndarray] = None,
                          max_rounds: int = 100,
                          verbose: bool = False) -> Tuple[List[int], Set[int], float]:
    selected = list(selected)
    selected_set = set(selected)
    covered = union_sets(selected, sensor_sets)
    best_score = weighted_size(covered, weights)

    n_sensors = len(sensor_sets)
    all_sensors = set(range(n_sensors))

    rounds = 0
    improved = True
    while improved and rounds < max_rounds:
        improved = False
        rounds += 1

        # Try all swaps (in random order to escape patterns)
        out_list = list(selected_set)
        random.shuffle(out_list)
        for s_out in out_list:
            # coverage without s_out
            base_cov = union_sets([i for i in selected if i != s_out], sensor_sets)

            # candidates to add
            candidates_in = list(all_sensors - selected_set)
            random.shuffle(candidates_in)

            best_local_gain = 0.0
            best_pair = None
            for s_in in candidates_in:
                cand_cov = base_cov | sensor_sets[s_in]
                score = weighted_size(cand_cov, weights)
                gain = score - best_score
                if gain > best_local_gain + 1e-12:
                    best_local_gain = gain
                    best_pair = (s_out, s_in, cand_cov, score)

            if best_pair is not None:
                s_out, s_in, cand_cov, score = best_pair
                # commit
                selected_set.remove(s_out)
                selected_set.add(s_in)
                selected = sorted(selected_set)
                covered = cand_cov
                best_score = score
                improved = True
                if verbose:
                    print(f"[1-swap] Replace {s_out} -> {s_in}, new score={best_score:.3f}, "
                          f"covered={len(covered)}/{universe_size}")
                break  # restart outer loop after improvement

    return sorted(selected_set), covered, best_score


# ---------------------------- GRASP ----------------------------

def grasp_one(seed, sensor_sets, V, k, rcl_size, local_rounds, weights):
    print("[Worker] Starting GRASP with seed", seed)
    A = build_sensor_csr(sensor_sets, V)
    sel, cov, score = grasp_max_coverage_sparse(
        A, V, k,
        rcl_size=rcl_size, iters=1,
        weights=weights, local_search_rounds=local_rounds,
        early_stop_no_improve=None, verbose=False, seed=seed
    )
    print(f"[Worker] Finished seed {seed}: score={score:.3f}, covered={len(cov)}/{V}, k={len(sel)}")
    return (score, sel, cov, seed)

def parallel_grasp(sensor_sets, V, k, iters, rcl_size=5, local_rounds=100, weights=None, procs=None):
    procs = procs or cpu_count()
    seeds = list(range(iters))
    best = None
    t0 = time.time()
    # small helper to show ETA
    def _status(done, best_score):
        elapsed = time.time() - t0
        eta = elapsed / done * (iters - done) if done else 0.0
        sys.stdout.write(
            f"\r[GRASP] {done}/{iters} done | best={best_score:.3f} | "
            f"elapsed={elapsed:.1f}s | ETA~{eta:.1f}s"
        )
        sys.stdout.flush()

    # run
    with Pool(processes=procs) as pool:
        it = 0
        for res in pool.imap_unordered(
            partial(grasp_one, sensor_sets=sensor_sets, V=V, k=k,
                    rcl_size=rcl_size, local_rounds=local_rounds, weights=weights),
            seeds,
            chunksize=max(1, iters // (procs * 4) or 1)
        ):
            it += 1
            score, sel, cov, seed = res
            if best is None or score > best[0]:
                best = (score, sel, cov, seed)
            _status(it, best[0])
    print()  # finalize the progress line

    score, sel, cov, seed = best
    print(f"[GRASP] Best from seed {seed}: score={score:.3f}, covered={len(cov)}/{V}, k={len(sel)}")
    return sel, cov, score

def grasp_max_coverage(sensor_sets: List[Set[int]],
                       universe_size: int,
                       k: int,
                       rcl_size: int = 5,
                       iters: int = 20,
                       weights: Optional[np.ndarray] = None,
                       local_search_rounds: int = 100,
                       early_stop_no_improve: Optional[int] = None,
                       verbose: bool = True,
                       seed: Optional[int] = None) -> Tuple[List[int], Set[int], float]:
    print("Starting GRASP max coverage:")

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    def wgain(new_vox: Set[int]) -> float:
        return weighted_size(new_vox, weights)

    n_sensors = len(sensor_sets)
    best_selected, best_covered = [], set()
    best_score = -1.0
    no_improve = 0

    for it in range(1, iters + 1):
        # ----- Construct phase (randomized greedy with RCL) -----
        covered = set()
        selected = []
        remaining = set(range(n_sensors))

        for step in range(k):
            gains = []
            for s in remaining:
                new_vox = sensor_sets[s] - covered
                if not new_vox:
                    continue
                gains.append((wgain(new_vox), s, new_vox))

            if not gains:
                break

            gains.sort(key=lambda x: x[0], reverse=True)
            rcl = gains[:min(rcl_size, len(gains))]
            gain, s_pick, new_vox = random.choice(rcl)

            selected.append(s_pick)
            covered |= new_vox
            remaining.remove(s_pick)

            if verbose:
                if weights is None:
                    print(f"[it {it:02d} | step {step+1:02d}] pick {s_pick} +{len(new_vox)} "
                          f"(cum {len(covered)}/{universe_size})")
                else:
                    print(f"[it {it:02d} | step {step+1:02d}] pick {s_pick} +w{gain:.3f} "
                          f"(cum_w {weighted_size(covered, weights):.3f})")

            if len(covered) == universe_size:
                break

        # ----- Local search phase (1-swap) -----
        sel2, cov2, score2 = one_swap_local_search(
            sensor_sets, selected, universe_size,
            weights=weights, max_rounds=local_search_rounds, verbose=verbose
        )

        # ----- Keep the best -----
        if score2 > best_score + 1e-12:
            best_selected, best_covered, best_score = sel2, cov2, score2
            no_improve = 0
            if verbose:
                if weights is None:
                    print(f"[it {it:02d}] New best (unweighted): {len(best_covered)}/{universe_size}, k={len(best_selected)}")
                else:
                    print(f"[it {it:02d}] New best (weighted): score={best_score:.3f}, "
                          f"covered={len(best_covered)}/{universe_size}, k={len(best_selected)}")
        else:
            no_improve += 1

        if early_stop_no_improve is not None and no_improve >= early_stop_no_improve:
            if verbose:
                print(f"[it {it:02d}] Early stop after {no_improve} non-improving iterations.")
            break

    return best_selected, best_covered, best_score


# ---------------------------- CLI ----------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="GRASP for maximum coverage (sensor selection).")
    ap.add_argument("--yaml_path", type=str, default="ur_sensor_sim/tmp/weighted_heatmaps/b1_w1_heatmap.yaml",
                    help="Input YAML (with visible_by, voxel_count, etc.)")
    ap.add_argument("--k", type=int, default=25, help="Sensor budget (max number of sensors).")
    ap.add_argument("--iters", type=int, default=30, help="GRASP iterations (restarts).")
    ap.add_argument("--rcl-size", type=int, default=5, help="Restricted candidate list size.")
    ap.add_argument("--local-rounds", type=int, default=100, help="Max 1-swap local search rounds.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    ap.add_argument("--verbose", action="store_true", help="Print per-step logs.")

    # Weights sources / behavior:
    ap.add_argument("--weights", type=str, default="ur_sensor_sim/tmp/weighted_poses/b1_w1.npy",
                    help="Optional .npy file with per-voxel weights (float, shape (V,)). If provided, overrides YAML.")
    ap.add_argument("--yaml-weight-key", type=str, default="weights",
                    help="YAML key for per-voxel weights (default: 'weights').")
    ap.add_argument("--normalize-weights", action="store_true",
                    help="Normalize loaded weights to [0,1] before optimization.")
    ap.add_argument("--early-stop", type=int, default=None,
                    help="Stop GRASP if no improvement for N iterations.")
    ap.add_argument("--export-json", type=str, default="ur_sensor_sim/tmp/result_single_pose.json",
                    help="Optional path to save results JSON (selection, coverage, stats).")
    return ap.parse_args()


def main():
    start = time.time()
    args = parse_args()

    sensor_sets, V, raw = load_visible_by(args.yaml_path)
    print(f"Loaded sensor sets from {args.yaml_path}: {len(sensor_sets)} sensors, {V} voxels.")

    # -------- load weights (priority: external .npy > YAML key) --------
    weights = None
    source = None

    if args.weights is not None:
        arr = np.load(args.weights)
        if arr.shape != (V,):
            raise ValueError(f"weights shape {arr.shape} != (V,) = ({V},)")
        weights = arr.astype(np.float64)
        source = f".npy ({args.weights})"
    else:
        key = args.yaml_weight_key
        if key in raw:
            arr = np.asarray(raw[key], dtype=np.float64)
            if arr.shape != (V,):
                raise ValueError(f"YAML '{key}' length {arr.shape} != V ({V})")
            weights = arr
            source = f"YAML['{key}']"

    if weights is not None and args.normalize_weights:
        wmin, wmax = float(np.min(weights)), float(np.max(weights))
        if wmax > wmin:
            weights = (weights - wmin) / (wmax - wmin)
        else:
            weights = np.zeros_like(weights)
        source = (source or "weights") + " + normalized[0,1]"

    if source:
        print(f"Using weighted objective from {source}.")
    else:
        print("No weights provided/found; using UNWEIGHTED coverage.")

    if args.verbose:
        print(f"Loaded {len(sensor_sets)} sensors, {V} voxels from {args.yaml_path}")

    print("Starting GRASP max coverage...")

    best_sel, best_cov, best_score = parallel_grasp(
        sensor_sets=sensor_sets,
        V=V,
        k=args.k,
        iters=args.iters,
        rcl_size=args.rcl_size,
        local_rounds=args.local_rounds,
        weights=weights,
        procs=None,
    )

    covered_count = len(best_cov)
    coverage_frac = covered_count / V if V > 0 else 0.0
    print("\n========== RESULT ==========")
    print(f"Selected sensors (k={len(best_sel)}): {best_sel}")
    print(f"Covered voxels: {covered_count}/{V} ({100.0*coverage_frac:.2f}%)")
    if weights is None:
        print(f"Objective (unweighted): {covered_count}")
    else:
        print(f"Objective (weighted): {best_score:.3f}")

    if args.export_json:
        out = {
            "selected_sensors": best_sel,
            "covered_voxels_count": covered_count,
            "coverage_fraction": coverage_frac,
            "objective": float(best_score if weights is not None else covered_count),
            "k": args.k,
            "iters": args.iters,
            "rcl_size": args.rcl_size,
            "local_rounds": args.local_rounds,
            "seed": args.seed,
            "yaml_path": os.path.abspath(args.yaml_path),
            "weights_path": os.path.abspath(args.weights) if args.weights else None,
            "weights_source": source,
        }
        Path(args.export_json).write_text(json.dumps(out, indent=2))
        print(f"Saved JSON to: {args.export_json}")

    end = time.time()
    print(end - start)

if __name__ == "__main__":
    main()
