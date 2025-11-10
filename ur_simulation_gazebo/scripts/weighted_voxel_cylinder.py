#!/usr/bin/env python3
"""
make_weighted_voxels_per_pose_two_cyl.py

For each pose (base, forearm, tcp) in centers.yaml, compute per-voxel weights
from TWO cylinders:
  - cylinder 1 axis: base -> forearm
  - cylinder 2 axis: forearm -> tcp

Weight per voxel = falloff( min( dist3D_to_segment(base,forearm),
                                 dist3D_to_segment(forearm,tcp) ) )

Writes one YAML per pose: out_dir/<pose>.yaml, copying the input voxel YAML and
adding a 'weights' array and a 'weighting' metadata block.

Usage:
  python3 make_weighted_voxels_per_pose_two_cyl.py \
    --voxels ur_sensor_sim/tmp/capsule.yaml \
    --centers ur_sensor_sim/tmp/centers.yaml \
    --out-dir ur_sensor_sim/tmp/weighted_poses_two_cyl \
    --aggregate max \
    --robot-radius 0.20 --mode gamma --gamma 2.0 --min-w 0.05
"""

import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import yaml

def _point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Vectorized 3D distance from points p (N,3) to segment ab (3,), (3,).
    Returns (N,) distances.
    """
    ab = b - a
    ap = p - a
    ab2 = np.dot(ab, ab)
    # handle degenerate segment (a==b)
    if ab2 < 1e-16:
        return np.linalg.norm(ap, axis=1)
    t = np.clip((ap @ ab) / ab2, 0.0, 1.0)  # projection parameter on segment
    proj = a + np.outer(t, ab)
    d = np.linalg.norm(p - proj, axis=1)
    return d

def _falloff(norm: np.ndarray, mode: str, gamma: float, alpha: float) -> np.ndarray:
    if mode == "linear":
        val = 1.0 - norm
    elif mode == "gamma":
        val = (1.0 - norm) ** float(gamma)
    elif mode == "exp":
        val = np.exp(-float(alpha) * norm)
    else:
        raise ValueError(f"Unknown mode '{mode}'")
    return np.clip(val, 0.0, 1.0)

def _weights_from_two_segments(
    voxels_xyz: np.ndarray,
    base: np.ndarray,
    forearm: np.ndarray,
    tcp: np.ndarray,
    robot_radius: float,
    r_max: float | None,
    mode: str,
    gamma: float,
    alpha: float,
    min_w: float,
) -> tuple[np.ndarray, float]:
    """
    Compute weights from two 3D cylinders (base->forearm and forearm->tcp).
    Returns (weights, r_max_used).
    """
    d1 = _point_segment_distance(voxels_xyz, base, forearm)
    d2 = _point_segment_distance(voxels_xyz, forearm, tcp)
    d = np.minimum(d1, d2)

    r0 = max(0.0, float(robot_radius))
    d_eff = np.clip(d - r0, 0.0, None)

    if r_max is None:
        # normalize by the farthest effective distance to span [min_w, 1]
        r_max_eff = float(np.max(d_eff)) if np.any(d_eff > 0) else 1e-6
    else:
        r_max_eff = max(1e-6, float(r_max - r0))

    norm = np.clip(d_eff / r_max_eff, 0.0, 1.0)
    fall = _falloff(norm, mode, gamma, alpha)

    min_w = float(np.clip(min_w, 0.0, 1.0))
    w = min_w + (1.0 - min_w) * fall
    return np.clip(w, min_w, 1.0), float(r0 + r_max_eff)

def main():
    ap = argparse.ArgumentParser(description="Create one weighted-voxel YAML per pose using TWO cylinders (base→forearm, forearm→tcp).")
    ap.add_argument("--voxels", default="ur_sensor_sim/tmp/capsule.yaml", help="Input voxel YAML with 'voxels': [[x,y,z], ...]")
    ap.add_argument("--centers", default="ur_sensor_sim/tmp/centers.yaml", help="centers.yaml with {centers: [[x,y,z]...], labels: [...]}")
    ap.add_argument("--out-dir", default="ur_sensor_sim/tmp/weighted_poses_two_cyl", help="Output directory")

    # shaping
    ap.add_argument("--robot-radius", type=float, default=0.20, help="Inner radius r0 (m) where weight=1")
    ap.add_argument("--r-max", type=float, default=None, help="Outer radius for normalization (default: farthest distance)")
    ap.add_argument("--mode", choices=["linear","gamma","exp"], default="gamma", help="Falloff mode")
    ap.add_argument("--gamma", type=float, default=2.0, help="Gamma exponent (mode=gamma)")
    ap.add_argument("--alpha", type=float, default=4.0, help="Alpha slope (mode=exp)")
    ap.add_argument("--min-w", type=float, default=0.05, help="Lower clamp for weights")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Load voxels
    vox_doc = yaml.safe_load(Path(args.voxels).read_text())
    voxels = np.asarray(vox_doc["voxels"], dtype=np.float32)
    if voxels.ndim != 2 or voxels.shape[1] != 3:
        raise SystemExit("voxels YAML must contain 'voxels' as Nx3 list.")

    # Load centers + labels and group indices by pose
    cent_doc = yaml.safe_load(Path(args.centers).read_text())
    centers = np.asarray(cent_doc["centers"], dtype=float)
    labels  = cent_doc.get("labels", None)
    if labels is None or len(labels) != len(centers):
        raise SystemExit("centers.yaml must include 'labels' aligned with 'centers'.")

    poses = {}
    for idx, lab in enumerate(labels):
        pname = str(lab.get("pose", "pose"))
        ptype = str(lab.get("type", ""))
        if pname not in poses:
            poses[pname] = {}
        poses[pname][ptype] = centers[idx]

    # Validate each pose has base_fixed, forearm, tcp
    valid_poses = {p: poses[p] for p in poses if all(k in poses[p] for k in ("base_fixed","forearm","tcp"))}
    missing = [p for p in poses if p not in valid_poses]
    if missing:
        print(f"[WARN] Skipping poses missing required points: {missing}")

    print(f"Found {len(valid_poses)} poses; computing two-cylinder weights per pose…")

    for pose_name, pts in valid_poses.items():
        base = np.asarray(pts["base_fixed"], dtype=float)
        fore = np.asarray(pts["forearm"], dtype=float)
        tcp  = np.asarray(pts["tcp"], dtype=float)

        w, rmu = _weights_from_two_segments(
            voxels_xyz=voxels,
            base=base, forearm=fore, tcp=tcp,
            robot_radius=args.robot_radius,
            r_max=args.r_max,
            mode=args.mode, gamma=args.gamma, alpha=args.alpha,
            min_w=args.min_w,
        )

        out_doc = dict(vox_doc)  # shallow copy of voxel YAML
        out_doc["weighting"] = {
            "type": "two_cylinders_3d_segments",
            "segments": {
                "base_to_forearm": base.tolist() + fore.tolist(),  # flattened for quick glance
                "forearm_to_tcp":  fore.tolist() + tcp.tolist(),
            },
            "robot_radius": float(args.robot_radius),
            "r_max_used": float(rmu),
            "mode": args.mode, "gamma": float(args.gamma), "alpha": float(args.alpha),
            "min_w": float(args.min_w),
            "pose_name": pose_name,
            "source_centers_file": str(Path(args.centers).resolve()),
            "note": "Weight = falloff(min(distance_to_segment(base→forearm), distance_to_segment(forearm→tcp))). 3D distances.",
        }
        out_doc["weights"] = w.astype(float).tolist()

        out_path = out_dir / f"{pose_name}.yaml"
        out_path.write_text(yaml.safe_dump(out_doc, sort_keys=False))
        print(f"[OK] wrote {out_path} | w∈[{w.min():.3f},{w.max():.3f}]")

if __name__ == "__main__":
    main()
