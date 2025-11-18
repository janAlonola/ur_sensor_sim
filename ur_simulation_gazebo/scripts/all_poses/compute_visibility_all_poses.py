#!/usr/bin/env python3
"""
compute_visibility_batch_with_poses.py

Inputs:
  --candidates  : path to single candidates.yaml (with {candidates:[...]})
  --voxels      : folder of voxel YAMLs (each with 'voxels': [[x,y,z], ...])
                  ideally each also has weighting.pose_name set (e.g., "b1_w2")
  --poses       : poses.yaml (as provided in the prompt)
  --out         : output folder. One <stem>_heatmap.yaml per input voxel YAML.

Notes:
  - Sensor local +Z is the viewing direction.
  - Symmetric cone FOV (--fov), range cutoff (--max-range).
  - No occlusion.
  - Sensors are transformed through URDF FK for the selected pose.
"""

import argparse
import math
from pathlib import Path
import re
import numpy as np
import yaml
import transforms3d as t3d
from transforms3d.euler import euler2mat, mat2euler
from urdfpy import URDF

# numpy compat (older code sometimes expects these)
if not hasattr(np, 'float'):
    np.float = float
    np.int = int
    np.bool = bool

# --------------- helpers ---------------

def load_yaml(p: Path):
    return yaml.safe_load(p.read_text())

def save_yaml(p: Path, data: dict):
    p.write_text(yaml.safe_dump(data, sort_keys=False))

def list_yaml_files(folder: Path):
    return sorted([x for x in folder.iterdir()
                   if x.is_file() and x.suffix.lower() in (".yaml", ".yml")])

def rpy_to_matrix(rpy):
    return t3d.euler.euler2mat(*rpy, axes='sxyz')

def sensor_pose_to_base(sensor, urdf: URDF, joint_cfg=None, base_frame=None):
    """
    Transform sensor pose (defined in sensor['link'] frame) to 'base_frame' (e.g., 'world').
    """
    if joint_cfg is None:
        joint_cfg = {}

    fk = urdf.link_fk(cfg=joint_cfg)
    link_by_name = {L.name: L for L in urdf.links}
    parent_link = link_by_name[sensor['link']]
    T_base_parent = fk[parent_link]

    xyz_local = np.array(sensor['xyz'], dtype=float)
    rpy_local = np.array(sensor['rpy'], dtype=float)
    R_local = euler2mat(*rpy_local, axes='sxyz')

    T_parent_sensor = np.eye(4)
    T_parent_sensor[:3, :3] = R_local
    T_parent_sensor[:3, 3]  = xyz_local

    T_base_sensor = T_base_parent @ T_parent_sensor

    if base_frame is not None and base_frame != urdf.base_link.name:
        req_link = link_by_name[base_frame]
        T_base_req = fk[req_link]
        T_req_sensor = np.linalg.inv(T_base_req) @ T_base_sensor
        R_out = T_req_sensor[:3, :3]
        p_out = T_req_sensor[:3, 3]
    else:
        R_out = T_base_sensor[:3, :3]
        p_out = T_base_sensor[:3, 3]

    rpy_out = mat2euler(R_out, axes='sxyz')
    return p_out, np.array(rpy_out)

def compute_visibility(voxels: np.ndarray, sensors: list, max_range=1.5, fov_deg=60.0):
    Nvox = len(voxels)
    Nsens = len(sensors)
    visible = np.zeros((Nvox, Nsens), dtype=bool)

    half_fov_cos = math.cos(math.radians(fov_deg) / 2.0)

    for j, s in enumerate(sensors):
        origin = np.array(s["xyz"])
        R = rpy_to_matrix(s["rpy"])
        view = R[:, 2]  # +Z axis

        diff = voxels - origin
        dist = np.linalg.norm(diff, axis=1)
        dirv = diff / (dist[:, None] + 1e-12)

        cosang = dirv @ view
        visible[:, j] = (cosang >= half_fov_cos) & (dist <= s.get("max_range", max_range))

    return visible

def load_pose_table(poses_yaml: Path, urdf_joint_suffix="_joint"):
    """
    Parse the provided poses.yaml and return:
      - pose_to_cfg: dict[str -> dict{urdf_joint_name: rad}]
      - ordered_urdf_joint_names: list[str]
    The YAML uses joint names without '_joint' suffix; we append it.
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

def infer_pose_name_from_voxel_doc_or_filename(vdoc: dict, stem: str) -> str | None:
    # 1) from metadata
    pose_meta = (vdoc.get("weighting") or {}).get("pose_name")
    if isinstance(pose_meta, str) and pose_meta.strip():
        return pose_meta.strip()
    # 2) filename like b3_w2, B5_W1, etc.
    m = re.search(r"(?i)\b(b\d+_w\d+)\b", stem)
    if m:
        return m.group(1).lower()
    return None

# --------------- main ---------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="ur_sensor_sim/mesh_sampling/selected_candidates.yaml",
                    help="Path to sel_candidates.yaml (single file)")
    ap.add_argument("--voxels", default="ur_sensor_sim/tmp/weighted_poses",
                    help="Directory containing per-pose voxel YAMLs")
    ap.add_argument("--poses", default="ur_sensor_sim/tmp/poses.yaml",
                    help="poses.yaml with joint_names and poses (as provided)")
    ap.add_argument("--out", default="ur_sensor_sim/tmp/weighted_heatmaps",
                    help="Output directory (one heatmap per voxel YAML)")
    ap.add_argument("--fov", type=float, default=60.0, help="Field of view (deg)")
    ap.add_argument("--max-range", type=float, default=1.5, help="Sensor max range (m)")
    ap.add_argument("--urdf", default="ur_sensor_sim/tmp/ur10.urdf", help="URDF path")
    ap.add_argument("--base-frame", default="world", help="Frame to express sensors/voxels in")
    args = ap.parse_args()

    cand_path = Path(args.candidates)
    vox_dir   = Path(args.voxels)
    poses_path= Path(args.poses)
    out_dir   = Path(args.out)

    if not cand_path.is_file():
        raise SystemExit(f"--candidates must be a file: {cand_path}")
    if not vox_dir.is_dir():
        raise SystemExit(f"--voxels must be a directory: {vox_dir}")
    if not poses_path.is_file():
        raise SystemExit(f"--poses must be a file: {poses_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Load inputs
    cand_doc = load_yaml(cand_path)
    sensors_src = cand_doc["candidates"]
    print(f"[INFO] Loaded {len(sensors_src)} candidate sensors from {cand_path.name}")

    pose_to_cfg, urdf_joint_names = load_pose_table(poses_path)
    print(f"[INFO] Loaded {len(pose_to_cfg)} poses from {poses_path.name}: {sorted(pose_to_cfg.keys())}")

    urdf = URDF.load(args.urdf)

    # Process all voxel YAMLs
    voxel_files = list_yaml_files(vox_dir)
    if not voxel_files:
        raise SystemExit(f"No voxel YAMLs found in {vox_dir}")

    print(f"[INFO] Processing {len(voxel_files)} voxel maps from {vox_dir} → {out_dir}")

    for vf in voxel_files:
        vdoc = load_yaml(vf)
        voxels = np.asarray(vdoc["voxels"], dtype=np.float32)
        # Choose pose
        pose_name = infer_pose_name_from_voxel_doc_or_filename(vdoc, vf.stem)
        print(pose_name)
        if pose_name is None or pose_name not in pose_to_cfg:
            # fallback to first pose in table
            pose_name = next(iter(pose_to_cfg.keys()))
            print(f"[WARN] Could not match pose for {vf.name}; using '{pose_name}'")
        joint_cfg = pose_to_cfg[pose_name]

        # Transform sensors for this pose
        sensors_world = []
        for s in sensors_src:
            xyz_w, rpy_w = sensor_pose_to_base(s, urdf, joint_cfg, base_frame=args.base_frame)
            sensors_world.append({
                "xyz": xyz_w,
                "rpy": rpy_w,
                "max_range": s.get("max_range", args.max_range),
            })

        # Visibility for this pose
        visible = compute_visibility(voxels, sensors_world, args.max_range, args.fov)
        coverage = visible.sum(axis=1)
        voxel_to_sensors = [np.nonzero(visible[i])[0].tolist() for i in range(len(voxels))]

        out_doc = {
            "voxel_size_m": vdoc.get("voxel_size_m"),
            "voxel_count": int(len(voxels)),
            "sensor_count": int(len(sensors_world)),
            "pose_count": 1,
            "pose_name": pose_name,
            "fov_deg": float(args.fov),
            "max_range_m": float(args.max_range),
            "voxels": voxels.tolist(),
            # carry over precomputed voxel weights if present
            **({"weights": vdoc["weights"]} if "weights" in vdoc else {}),
            "coverage": coverage.astype(int).tolist(),
            "visible_by": voxel_to_sensors,
        }

        out_path = out_dir / f"{vf.stem}_heatmap.yaml"
        save_yaml(out_path, out_doc)
        unseen = int((coverage == 0).sum())
        print(f"[OK] {vf.name} → {out_path.name} | pose={pose_name} | mean={coverage.mean():.2f} | unseen={unseen}")

if __name__ == "__main__":
    main()
