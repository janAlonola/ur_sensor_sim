#!/usr/bin/env python3
"""
weighted_voxel.py

Compute per-voxel weights based on distance to the *actual robot geometry*
(from URDF collision meshes), for all poses in poses.yaml.

Uses true distance to robot collision mesh to compute weights.

Inputs:
  --voxels   : voxel grid YAML with 'voxels': [[x,y,z], ...] in base-frame coords
  --urdf     : robot URDF with collision geometry
  --poses    : poses.yaml mapping pose_name -> joint angles
  --base-frame : frame in which voxels are expressed (e.g. 'world')

Weights:
  - Distance d = distance from voxel to robot collision mesh (m).
  - Inner radius r0 = --robot-radius (inside this, distance is saturated).
  - Outer normalization radius r_max (explicit or derived from data).
  - Falloff modes: linear, gamma, exp.
  - Voxels with distance <= --zero-inside are forced to weight=0.

For each pose in poses.yaml, writes:
  <out-dir>/<pose_name>.yaml
"""

import argparse
from pathlib import Path
import numpy as np
import yaml
import math

import trimesh
from trimesh.proximity import closest_point

from urdfpy import URDF

# numpy compat
if not hasattr(np, 'float'):
    np.float = float
    np.int = int
    np.bool = bool

# ---------------- basic helpers ----------------

def load_yaml(path: Path):
    return yaml.safe_load(path.read_text())

def _falloff(norm: np.ndarray, mode: str, gamma: float, alpha: float) -> np.ndarray:
    if mode == "linear":
        val = 1.0 - norm
    elif mode == "gamma":
        val = (1.0 - norm) ** float(gamma)
    elif mode == "exp":
        val = np.exp(-float(alpha) * norm)
    elif mode == "sigmoid":
        # k controls how steep the S is; reuse gamma as "steepness"
        k = float(gamma) if gamma is not None else 8.0  # 8 is a decent default
        # decreasing logistic:
        # norm = 0   -> ~1
        # norm = 0.5 -> ~0.5
        # norm = 1   -> ~0
        val = 1.0 / (1.0 + np.exp(k * (norm - 0.5)))
    else:
        raise ValueError(f"Unknown mode '{mode}'")
    return np.clip(val, 0.0, 1.0)

# --------------- URDF + pose helpers ---------------

def load_pose_table(poses_yaml: Path, urdf_joint_suffix="_joint"):
    """
    Parse poses.yaml and return:
      - pose_to_cfg: dict[str -> dict{urdf_joint_name: rad}]
      - ordered_urdf_joint_names: list[str]

    Expected format (example):

    joint_names: [shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3]
    poses:
      - name: b1_w1
        joints_deg: [0, -90, 0, -90,   0,   0]
        joints_rad: [...]
      ...
    """
    doc = load_yaml(poses_yaml)
    base_names = doc.get("joint_names", ["shoulder_pan","shoulder_lift","elbow",
                                         "wrist_1","wrist_2","wrist_3"])
    urdf_names = [f"{n}{urdf_joint_suffix}" for n in base_names]

    pose_to_cfg = {}
    for p in doc.get("poses", []):
        name = p.get("name")
        if not name:
            continue
        if "joints_rad" in p and p["joints_rad"] is not None:
            vals = list(map(float, p["joints_rad"]))
        elif "joints_deg" in p and p["joints_deg"] is not None:
            vals = [math.radians(float(x)) for x in p["joints_deg"]]
        else:
            raise ValueError(f"Pose '{name}' has neither joints_rad nor joints_deg.")
        if len(vals) != len(urdf_names):
            raise ValueError(f"Pose '{name}' length mismatch: {len(vals)} vs {len(urdf_names)}")
        pose_to_cfg[name] = {jn: v for jn, v in zip(urdf_names, vals)}

    if not pose_to_cfg:
        raise ValueError("No poses found in poses.yaml")
    return pose_to_cfg, urdf_names

def build_robot_mesh_world_combined(urdf: URDF, joint_cfg: dict, base_frame: str | None):
    """
    Build a *single combined* Trimesh for the robot in the given joint_cfg,
    expressed in 'base_frame' coordinates (e.g. 'world').
    """
    # FK for links in URDF base_link frame
    link_fk = urdf.link_fk(cfg=joint_cfg)
    link_by_name = {L.name: L for L in urdf.links}

    # Optional re-basing
    if base_frame is not None and base_frame != urdf.base_link.name:
        req_link = link_by_name[base_frame]
        T_base_req = link_fk[req_link]         # base_link -> base_frame
        T_req_base = np.linalg.inv(T_base_req) # base_frame -> base_link
    else:
        T_req_base = None

    # Collision meshes with FK in base_link frame
    coll_fk = urdf.collision_trimesh_fk(cfg=joint_cfg)  # {Trimesh: 4x4}
    meshes = []

    for mesh, T_base_mesh in coll_fk.items():
        m = mesh.copy()
        if T_req_base is not None:
            T_req_mesh = T_req_base @ T_base_mesh
            m.apply_transform(T_req_mesh)
        else:
            m.apply_transform(T_base_mesh)
        meshes.append(m)

    if not meshes:
        return None
    return trimesh.util.concatenate(meshes)

# --------------- weight from robot mesh ---------------

def _weights_from_robot_mesh(
    voxels_xyz: np.ndarray,
    robot_mesh: trimesh.Trimesh,
    robot_radius: float,
    r_max: float | None,
    mode: str,
    gamma: float,
    alpha: float,
    min_w: float,
    zero_inside: float | None = None,
    chunk_size: int = 5000,
) -> tuple[np.ndarray, float, dict]:
    """
    Compute weights for each voxel based on distance to the robot's collision mesh.
    Returns:
      w       : array of weights, one per voxel
      r_total : effective outer radius used (r0 + r_max_eff)
      info    : dict with some diagnostics (r0, r_max_used, zeroed_count, ...)
    """
    N = len(voxels_xyz)
    d = np.empty(N, dtype=float)

    # For each voxel, compute its distance 'd' to the robot surface. Chunked for memory.
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        _, di, _ = closest_point(robot_mesh, voxels_xyz[start:end]) # distance from closest_point to voxel
        d[start:end] = di

    # Treat everything closer than 'robot_radius' as "inside".
    r0 = max(0.0, float(robot_radius))
    d_eff = np.clip(d - r0, 0.0, None)
    if r_max is None:
        r_max_eff = float(np.max(d_eff)) if np.any(d_eff > 0) else 1e-6
    else:
        r_max_eff = max(1e-6, float(r_max - r0))

    # Normalize distances into [0, 1] using r_max and apply a falloff function.
    norm = np.clip(d_eff / r_max_eff, 0.0, 1.0)
    w = _falloff(norm, mode, gamma, alpha)
    # Clamp weights so they never go below min_w (except where we explicitly zero).
    min_w = float(np.clip(min_w, 0.0, 1.0))
    w = min_w + (1.0 - min_w) * w
    w = np.clip(w, min_w, 1.0)
    # Optionally, force weight = 0 for voxels within 'zero_inside' of the robot.
    zero_mask = None
    if zero_inside is not None and zero_inside > 0.0:
        zero_mask = d <= zero_inside
        if np.any(zero_mask):
            w[zero_mask] = 0.0

    info = {
        "r0_robot_radius": float(r0),
        "r_max_used": float(r0 + r_max_eff),
        "zero_inside": float(zero_inside if zero_inside is not None else 0.0),
        "zeroed_count": int(np.count_nonzero(zero_mask)) if zero_mask is not None else 0,
    }
    return w, float(r0 + r_max_eff), info


# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser(description="Voxel weights based on distance to robot collision mesh (no centers.yaml).")
    ap.add_argument("--voxels", default="ur_sensor_sim/tmp/capsule.yaml",
                    help="Input voxel YAML with 'voxels': [[x,y,z], ...] in base-frame coords")
    ap.add_argument("--urdf", default="ur_sensor_sim/tmp/ur10.urdf",
                    help="URDF path (with collision geometry)")
    ap.add_argument("--poses", default="ur_sensor_sim/tmp/poses.yaml",
                    help="poses.yaml mapping pose_name -> joint angles")
    ap.add_argument("--base-frame", default="world",
                    help="Frame in which voxels are expressed and to which robot is transformed")
    ap.add_argument("--out-dir", default="ur_sensor_sim/tmp/weighted_poses_giant_zeros",
                    help="Output directory (one YAML per pose)")

    # shaping
    ap.add_argument("--robot-radius", type=float, default=0.2,
                    help="Inner radius r0 (m) where base falloff saturates")
    ap.add_argument("--r-max", type=float, default=None,
                    help="Outer radius for normalization (default: farthest effective distance)")
    ap.add_argument("--mode", choices=["linear","gamma","exp", "sigmoid"], default="gamma",
                    help="Falloff mode")
    ap.add_argument("--gamma", type=float, default=4.0,
                    help="Gamma exponent (mode=gamma)")
    ap.add_argument("--alpha", type=float, default=4.0,
                    help="Alpha slope (mode=exp)")
    ap.add_argument("--min-w", type=float, default=0.01,
                    help="Lower clamp for weights (except zero-mask)")
    ap.add_argument("--zero-inside", type=float, default=0.15,
                    help="Meters: if voxel is within this distance to robot surface, set weight=0")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- load voxels ---
    vox_doc = load_yaml(Path(args.voxels))
    voxels = np.asarray(vox_doc["voxels"], dtype=np.float32)
    if voxels.ndim != 2 or voxels.shape[1] != 3:
        raise SystemExit("voxels YAML must contain 'voxels' as Nx3 list.")

    # --- load URDF + pose table ---
    urdf = URDF.load(args.urdf)
    pose_to_cfg, urdf_joint_names = load_pose_table(Path(args.poses))
    pose_names = sorted(pose_to_cfg.keys())
    print(f"[INFO] Loaded {len(pose_names)} poses from {args.poses}: {pose_names}")

    # --- per-pose processing ---
    for pose_name in pose_names:
        print(pose_name)
        joint_cfg = pose_to_cfg[pose_name]

        # build robot mesh in base_frame (e.g. world)
        robot_mesh = build_robot_mesh_world_combined(
            urdf,
            joint_cfg,
            base_frame=args.base_frame
        )
        if robot_mesh is None:
            print(f"[WARN] No collision meshes for pose '{pose_name}', skipping.")
            continue

        w, rmu, extra = _weights_from_robot_mesh(
            voxels_xyz=voxels,
            robot_mesh=robot_mesh,
            robot_radius=args.robot_radius,
            r_max=args.r_max,
            mode=args.mode, gamma=args.gamma, alpha=args.alpha,
            min_w=args.min_w,
            zero_inside=args.zero_inside,
        )

        out_doc = dict(vox_doc)
        out_doc["weighting"] = {
            "type": "robot_collision_distance",
            "robot_radius": float(args.robot_radius),
            "r_max_used": float(extra["r_max_used"]),
            "mode": args.mode, "gamma": float(args.gamma), "alpha": float(args.alpha),
            "min_w": float(args.min_w),
            "zero_inside": float(args.zero_inside),
            "zeroed_count": int(extra["zeroed_count"]),
            "pose_name": pose_name,
            "source_urdf": str(Path(args.urdf).resolve()),
            "source_poses_file": str(Path(args.poses).resolve()),
            "note": "Weight based on distance to robot collision mesh; voxels within --zero-inside of surface get weight=0.",
        }
        out_doc["weights"] = w.astype(float).tolist()

        out_path = out_dir / f"{pose_name}.yaml"
        out_path.write_text(yaml.safe_dump(out_doc, sort_keys=False))
        print(f"[OK] {pose_name:10s} → {out_path} | w∈[{w.min():.3f},{w.max():.3f}] | zeroed={extra['zeroed_count']}")

if __name__ == "__main__":
    main()
