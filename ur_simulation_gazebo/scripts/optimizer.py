#!/usr/bin/env python3
"""
GRASP (Greedy Randomized Adaptive Search Procedure) for Sensor Selection
----------------------------------------------------------------------

Problem: pick up to k sensors to maximize covered voxels (weighted or unweighted)
from a YAML file with:
  - voxel_count (int)
  - sensor_count (int)
  - pose_count (int)
  - visible_by: list[pose][voxel] -> list of sensor indices

We precompute per-sensor voxel sets (union across all poses), then run:
  1) randomized greedy construction with a restricted candidate list (RCL)
  2) 1-swap local search improvement
Repeat for 'iters' restarts and keep the best.

Usage:
  python grasp_max_coverage.py heatmap_with_v_by_s.yaml --k 10 --iters 50 --rcl-size 5

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
# ---------------------------- Data loading ----------------------------

def load_visible_by(yaml_path: str) -> Tuple[List[Set[int]], int]:
    """
    Expects YAML with:
      - voxel_count
      - sensor_count
      - pose_count
      - visible_by: list[pose][voxel] -> list of sensor indices
    Returns:
      sensor_sets: list of sets; sensor_sets[j] = set of voxel indices covered by sensor j across all poses
      V: universe size (voxel_count)
    """
    print(f"Loading visible_by from {yaml_path}...")
    data = yaml.load(Path(yaml_path).read_bytes(), Loader=YLoader)
    #data = yaml.safe_load(Path(yaml_path).read_text())
    print("loaded YAML.")
    V = int(data["voxel_count"])
    P = int(data["pose_count"])
    S = int(data["sensor_count"])
    vis_by = data["visible_by"]  # shape [P][V] -> list[int]

    sensor_sets = [set() for _ in range(S)]
    for p in range(P):
        vox_lists = vis_by[p]
        for v, sens_list in enumerate(vox_lists):
            for s in sens_list:
                sensor_sets[int(s)].add(int(v))

    return sensor_sets, V


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
    """
    Improve a given selection by swapping one sensor out and one in, greedily.
    Stops when no 1-swap improves coverage or max_rounds reached.

    Returns:
      selected_best (sorted list), covered_best (set), best_score (float)
    """
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
    sel, cov, score = grasp_max_coverage(
        sensor_sets, V, k,
        rcl_size=rcl_size, iters=1,  # one restart per process
        weights=weights, local_search_rounds=local_rounds,
        early_stop_no_improve=None, verbose=False, seed=seed
    )
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
    """
    GRASP: randomized greedy construction + 1-swap local search, repeated.
    - rcl_size: restricted candidate list size for randomized greedy
    - iters: number of GRASP iterations
    - early_stop_no_improve: stop if no improvement for this many iterations

    Returns:
      best_selected (list), best_covered (set), best_score (float)
    """
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
            # compute gains for all remaining
            gains = []
            for s in remaining:
                new_vox = sensor_sets[s] - covered
                if not new_vox:
                    continue
                gains.append((wgain(new_vox), s, new_vox))

            if not gains:
                break

            # sort by gain and pick at random among top rcl_size
            gains.sort(key=lambda x: x[0], reverse=True)
            rcl = gains[:min(rcl_size, len(gains))]
            gain, s_pick, new_vox = random.choice(rcl)

            selected.append(s_pick)
            covered |= new_vox
            remaining.remove(s_pick)

            if verbose:
                print(f"[it {it:02d} | step {step+1:02d}] pick {s_pick} +{len(new_vox)} "
                      f"(cum {len(covered)}/{universe_size})")

            # optional early exit if fully covered
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
                print(f"[it {it:02d}] New best: score={best_score:.3f}, "
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
    ap.add_argument("--yaml_path", type=str, default = "rand_heatmap.yaml", help="Input YAML (with visible_by, voxel_count, etc.)")
    ap.add_argument("--k", type=int, default = 20, help="Sensor budget (max number of sensors).")
    ap.add_argument("--iters", type=int, default=30, help="GRASP iterations (restarts).")
    ap.add_argument("--rcl-size", type=int, default=5, help="Restricted candidate list size.")
    ap.add_argument("--local-rounds", type=int, default=100, help="Max 1-swap local search rounds.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    ap.add_argument("--verbose", action="store_true", help="Print per-step logs.")
    ap.add_argument("--weights", type=str, default=None,
                    help="Optional .npy file with per-voxel weights (float, shape (V,)).")
    ap.add_argument("--early-stop", type=int, default=None,
                    help="Stop GRASP if no improvement for N iterations.")
    ap.add_argument("--export-json", type=str, default=None,
                    help="Optional path to save results JSON (selection, coverage, stats).")
    return ap.parse_args()


def main():
    start = time.time()
    args = parse_args()

    sensor_sets, V = load_visible_by(args.yaml_path)
    print(f"Loaded sensor sets from {args.yaml_path}: {len(sensor_sets)} sensors, {V} voxels.")

    weights = None
    if args.weights is not None:
        weights_array = np.load(args.weights)
        if weights_array.shape != (V,):
            raise ValueError(f"weights shape {weights_array.shape} != (V,) = ({V},)")
        weights = weights_array.astype(np.float64)

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
            "objective": best_score if weights is not None else covered_count,
            "k": args.k,
            "iters": args.iters,
            "rcl_size": args.rcl_size,
            "local_rounds": args.local_rounds,
            "seed": args.seed,
            "yaml_path": os.path.abspath(args.yaml_path),
            "weights_path": os.path.abspath(args.weights) if args.weights else None,
        }
        Path(args.export_json).write_text(json.dumps(out, indent=2))
        print(f"Saved JSON to: {args.export_json}")

    end = time.time()
    print(end - start)

if __name__ == "__main__":
    main()
