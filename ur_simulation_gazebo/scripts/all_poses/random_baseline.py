#!/usr/bin/env python3
"""
random_baseline.py

Sample random sensor sets of size k and report their objective values.
Works with your existing optimizer_ring.py helpers / heatmaps.

Example:
  python3 random_baseline.py --heatmaps ur_sensor_sim/tmp/big_visibility_combined \
    --weights-list ur_sensor_sim/tmp/weighted_poses_semi_zeros/*.npy \
    --k 20 --samples 50 --fixed-tail 7 --objective sum --seed 0
"""

import argparse, random
from pathlib import Path

import numpy as np

# Reuse these from optimizer_ring.py (import by file or copy-paste):
# - load_all_pose_csrs
# - load_weight_npys_matrix
# - extract_rows
# - objective_from_masks
#
from optimizer_ring import load_all_pose_csrs, load_weight_npys_matrix, extract_rows, objective_from_masks


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heatmaps", default="ur_sensor_sim/tmp/big_visibility_ring", help="Combined heatmaps dir (or single YAML).")
    ap.add_argument("--weights-list", nargs="+", default=["ur_sensor_sim/tmp/weighted_poses_bigger_zeros/*.npy"], help="Pose weight .npy globs/paths.")
    ap.add_argument("--k", type=int, default=20, help="Total sensor budget (including fixed).")
    ap.add_argument("--samples", type=int, default=20, help="How many random sets to evaluate.")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed.")
    ap.add_argument("--objective", choices=["sum","softmin","frac"], default="sum")
    ap.add_argument("--softmin-temp", type=float, default=0.3)
    ap.add_argument("--frac-alpha", type=float, default=0.6)
    ap.add_argument("--fixed-tail", type=int, default=0,
                    help="If >0, treat the last fixed-tail sensors as fixed and always include them.")
    return ap.parse_args()


def score_selection(sel, rows, V, W, mode, softmin_temp, frac_alpha):
    P = len(rows)
    covered_p = np.zeros((P, V), dtype=bool)
    for p in range(P):
        for s in sel:
            idx = rows[p][s]
            if idx.size:
                covered_p[p, idx] = True
    return objective_from_masks(covered_p, W, mode, softmin_temp, frac_alpha)


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    A_list, V, S, heatmap_files, P = load_all_pose_csrs(args.heatmaps)
    W, weight_files = load_weight_npys_matrix(args.weights_list, V, expect_P=P)

    rows, S2, V2, P2 = extract_rows(A_list)
    assert S2 == S and V2 == V and P2 == P

    if args.fixed_tail < 0 or args.fixed_tail > S:
        raise SystemExit(f"--fixed-tail must be in [0,{S}], got {args.fixed_tail}")

    fixed = list(range(S - args.fixed_tail, S)) if args.fixed_tail > 0 else []
    if len(fixed) > args.k:
        raise SystemExit(f"fixed ({len(fixed)}) exceeds k={args.k}")

    k_free = args.k - len(fixed)
    pool = [i for i in range(S) if i not in set(fixed)]

    best = None
    for t in range(args.samples):
        pick = random.sample(pool, k_free) if k_free > 0 else []
        sel = sorted(fixed + pick)

        obj = score_selection(sel, rows, V, W,
                              mode=args.objective,
                              softmin_temp=args.softmin_temp,
                              frac_alpha=args.frac_alpha)

        print(f"[{t+1:03d}/{args.samples}] obj={obj:.3f} | sel={sel}")

        if best is None or obj > best[0]:
            best = (obj, sel)

    print("\n==== BEST RANDOM SAMPLE ====")
    print(f"obj={best[0]:.3f}")
    print(f"sel={best[1]}")


if __name__ == "__main__":
    main()
