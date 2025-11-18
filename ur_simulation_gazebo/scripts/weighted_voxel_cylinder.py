#!/usr/bin/env python3
"""
weighted_voxel_cylinder.py  (two-segment version with inner zero-mask)

Adds --zero-inside so voxels within that distance to either segment
(base→forearm or forearm→tcp) are forced to weight=0.
"""

import argparse
from pathlib import Path
import numpy as np
import yaml

def _point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    ap = p - a
    ab2 = float(ab @ ab)
    if ab2 < 1e-16:
        return np.linalg.norm(ap, axis=1)
    t = np.clip((ap @ ab) / ab2, 0.0, 1.0)
    proj = a + np.outer(t, ab)
    return np.linalg.norm(p - proj, axis=1)

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
    zero_inside: float | None = None,
) -> tuple[np.ndarray, float, dict]:
    # distances to both segments
    d1 = _point_segment_distance(voxels_xyz, base, forearm)
    d2 = _point_segment_distance(voxels_xyz, forearm, tcp)
    d = np.minimum(d1, d2)

    # inner saturation (r0) as before
    r0 = max(0.0, float(robot_radius))
    d_eff = np.clip(d - r0, 0.0, None)

    # normalization radius
    if r_max is None:
        r_max_eff = float(np.max(d_eff)) if np.any(d_eff > 0) else 1e-6
    else:
        r_max_eff = max(1e-6, float(r_max - r0))

    # base weights from falloff
    norm = np.clip(d_eff / r_max_eff, 0.0, 1.0)
    w = _falloff(norm, mode, gamma, alpha)
    min_w = float(np.clip(min_w, 0.0, 1.0))
    w = min_w + (1.0 - min_w) * w
    w = np.clip(w, min_w, 1.0)

    # NEW: zero out voxels inside the robot “capsule” with buffer
    zero_mask = None
    if zero_inside is not None and zero_inside > 0.0:
        # distance to either segment <= buffer → weight = 0.0
        zero_mask = (d1 <= zero_inside) | (d2 <= zero_inside)
        if np.any(zero_mask):
            w[zero_mask] = 0.0  # override min_w

    info = {
        "r0_robot_radius": float(r0),
        "r_max_used": float(r0 + r_max_eff),
        "zero_inside": float(zero_inside if zero_inside is not None else 0.0),
        "zeroed_count": int(np.count_nonzero(zero_mask)) if zero_mask is not None else 0,
    }
    return w, float(r0 + r_max_eff), info

def main():
    ap = argparse.ArgumentParser(description="Two-cylinder weights with inner zero-mask near robot.")
    ap.add_argument("--voxels", default="ur_sensor_sim/tmp/capsule.yaml",
                    help="Input voxel YAML with 'voxels': [[x,y,z], ...]")
    ap.add_argument("--centers", default="ur_sensor_sim/tmp/centers.yaml",
                    help="centers.yaml with {centers: [[x,y,z]...], labels: [...]}")
    ap.add_argument("--out-dir", default="ur_sensor_sim/tmp/weighted_poses",
                    help="Output directory")

    # shaping
    ap.add_argument("--robot-radius", type=float, default=0.20,
                    help="Inner radius r0 (m) where base falloff saturates")
    ap.add_argument("--r-max", type=float, default=None,
                    help="Outer radius for normalization (default: farthest effective distance)")
    ap.add_argument("--mode", choices=["linear","gamma","exp"], default="gamma",
                    help="Falloff mode")
    ap.add_argument("--gamma", type=float, default=2.0,
                    help="Gamma exponent (mode=gamma)")
    ap.add_argument("--alpha", type=float, default=4.0,
                    help="Alpha slope (mode=exp)")
    ap.add_argument("--min-w", type=float, default=0.05,
                    help="Lower clamp for weights (except zero-mask)")
    ap.add_argument("--zero-inside", type=float, default=0.15,
                    help="Meters: if voxel is within this distance to either segment, set weight=0")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    vox_doc = yaml.safe_load(Path(args.voxels).read_text())
    voxels = np.asarray(vox_doc["voxels"], dtype=np.float32)
    if voxels.ndim != 2 or voxels.shape[1] != 3:
        raise SystemExit("voxels YAML must contain 'voxels' as Nx3 list.")

    cent_doc = yaml.safe_load(Path(args.centers).read_text())
    centers = np.asarray(cent_doc["centers"], dtype=float)
    labels  = cent_doc.get("labels", None)
    if labels is None or len(labels) != len(centers):
        raise SystemExit("centers.yaml must include 'labels' aligned with 'centers'.")

    # group points by pose
    poses = {}
    for idx, lab in enumerate(labels):
        pname = str(lab.get("pose", "pose"))
        ptype = str(lab.get("type", ""))
        poses.setdefault(pname, {})[ptype] = centers[idx]

    # keep only complete poses
    valid = {p: poses[p] for p in poses if all(k in poses[p] for k in ("base_fixed","forearm","tcp"))}
    missing = [p for p in poses if p not in valid]
    if missing:
        print(f"[WARN] Skipping poses missing required points: {missing}")

    print(f"Found {len(valid)} poses; computing two-cylinder weights per pose…")

    for pose_name, pts in valid.items():
        base = np.asarray(pts["base_fixed"], dtype=float)
        fore = np.asarray(pts["forearm"], dtype=float)
        tcp  = np.asarray(pts["tcp"], dtype=float)

        w, rmu, extra = _weights_from_two_segments(
            voxels_xyz=voxels,
            base=base, forearm=fore, tcp=tcp,
            robot_radius=args.robot_radius,
            r_max=args.r_max,
            mode=args.mode, gamma=args.gamma, alpha=args.alpha,
            min_w=args.min_w,
            zero_inside=args.zero_inside,
        )

        out_doc = dict(vox_doc)
        out_doc["weighting"] = {
            "type": "two_cylinders_3d_segments",
            "segments": {
                "base_to_forearm": base.tolist() + fore.tolist(),
                "forearm_to_tcp":  fore.tolist() + tcp.tolist(),
            },
            "robot_radius": float(args.robot_radius),
            "r_max_used": float(extra["r_max_used"]),
            "mode": args.mode, "gamma": float(args.gamma), "alpha": float(args.alpha),
            "min_w": float(args.min_w),
            "zero_inside": float(args.zero_inside),
            "zeroed_count": int(extra["zeroed_count"]),
            "pose_name": pose_name,
            "source_centers_file": str(Path(args.centers).resolve()),
            "note": "Weight = falloff(min(dist to segment1, segment2)); voxels within --zero-inside to either segment get weight=0.",
        }
        out_doc["weights"] = w.astype(float).tolist()

        out_path = out_dir / f"{pose_name}.yaml"
        out_path.write_text(yaml.safe_dump(out_doc, sort_keys=False))
        print(f"[OK] {pose_name:10s} → {out_path} | w∈[{w.min():.3f},{w.max():.3f}] | zeroed={extra['zeroed_count']}")

if __name__ == "__main__":
    main()
