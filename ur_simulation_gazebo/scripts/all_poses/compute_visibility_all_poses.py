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
  - Sensors are transformed through URDF FK for the selected pose.
  - This version supports occlusion via trimesh raycasting and runs in parallel.
"""

import argparse
import math
from pathlib import Path
import re
import numpy as np
import yaml
import trimesh
import transforms3d as t3d
from transforms3d.euler import euler2mat, mat2euler
from urdfpy import URDF
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

# numpy compat (older code sometimes expects these)
if not hasattr(np, 'float'):
    np.float = float
    np.int = int
    np.bool = bool

# ---------- helpers ----------

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

def build_robot_meshes_world(urdf: URDF, joint_cfg: dict, base_frame: str | None):
    """
    Build a list of trimesh.Trimesh for the robot in the given joint_cfg,
    all expressed in base_frame coordinates (e.g. 'world').
    """
    link_fk = urdf.link_fk(cfg=joint_cfg)
    link_by_name = {L.name: L for L in urdf.links}

    if base_frame is not None and base_frame != urdf.base_link.name:
        req_link = link_by_name[base_frame]
        T_base_req = link_fk[req_link]         # base_link -> base_frame
        T_req_base = np.linalg.inv(T_base_req) # base_frame -> base_link
    else:
        T_req_base = None

    robot_meshes = []

    coll_fk = urdf.collision_trimesh_fk(cfg=joint_cfg)  # {Trimesh: 4x4} in base_link frame

    for mesh, T_base_mesh in coll_fk.items():
        m = mesh.copy()
        if T_req_base is not None:
            T_req_mesh = T_req_base @ T_base_mesh
            m.apply_transform(T_req_mesh)
        else:
            m.apply_transform(T_base_mesh)
        robot_meshes.append(m)

    return robot_meshes

def ray_blocked_by_robot(origin, target, robot_meshes, eps=1e-4):
    """
    Return True if the segment origin->target intersects any robot mesh
    before reaching the target.
    """
    direction = target - origin
    length = np.linalg.norm(direction)
    if length < eps:
        return False

    direction /= length
    ray_origins = origin[None, :]        # (1, 3)
    ray_directions = direction[None, :]  # (1, 3)

    for mesh in robot_meshes:
        locations, index_ray, index_tri = mesh.ray.intersects_location(
            ray_origins,
            ray_directions,
            multiple_hits=False
        )
        if len(locations) > 0:
            hit = locations[0]
            dist_hit = np.linalg.norm(hit - origin)
            if dist_hit < length - eps:
                return True

    return False

def compute_visibility_with_occlusion(voxels: np.ndarray, sensors: list,
                                      robot_meshes: list,
                                      max_range=1.5, fov_deg=60.0):
    """
    Like compute_visibility, but also checks line-of-sight against robot meshes.
    """
    Nvox = len(voxels)
    Nsens = len(sensors)
    visible = np.zeros((Nvox, Nsens), dtype=bool)

    half_fov_cos = math.cos(math.radians(fov_deg) / 2.0)

    for j, s in enumerate(sensors):
        origin = np.array(s["xyz"], dtype=float)
        R = rpy_to_matrix(s["rpy"])
        view = R[:, 2]  # +Z axis

        diff = voxels - origin
        dist = np.linalg.norm(diff, axis=1)
        dirv = diff / (dist[:, None] + 1e-12)

        cosang = dirv @ view
        in_fov = cosang >= half_fov_cos
        in_range = dist <= s.get("max_range", max_range)

        candidates = np.where(in_fov & in_range)[0]

        for i in candidates:
            if not ray_blocked_by_robot(origin, voxels[i], robot_meshes):
                visible[i, j] = True

    return visible

def load_pose_table(poses_yaml: Path, urdf_joint_suffix="_joint"):
    """
    Parse poses.yaml and return:
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

# ---------- globals used by workers ----------

GLOBAL_URDF = None
GLOBAL_SENSORS_SRC = None
GLOBAL_POSE_TO_CFG = None
GLOBAL_BASE_FRAME = None
GLOBAL_FOV = None
GLOBAL_MAX_RANGE = None
GLOBAL_OUT_DIR = None

def init_worker(urdf_path: str,
                sensors_src: list,
                pose_to_cfg: dict,
                base_frame: str,
                fov: float,
                max_range: float,
                out_dir: str):
    """
    Initializer for each worker process: loads URDF once, stores shared data.
    """
    global GLOBAL_URDF, GLOBAL_SENSORS_SRC, GLOBAL_POSE_TO_CFG
    global GLOBAL_BASE_FRAME, GLOBAL_FOV, GLOBAL_MAX_RANGE, GLOBAL_OUT_DIR

    GLOBAL_URDF = URDF.load(urdf_path)
    GLOBAL_SENSORS_SRC = sensors_src
    GLOBAL_POSE_TO_CFG = pose_to_cfg
    GLOBAL_BASE_FRAME = base_frame
    GLOBAL_FOV = fov
    GLOBAL_MAX_RANGE = max_range
    GLOBAL_OUT_DIR = Path(out_dir)

def process_voxel_file(vf_path_str: str):
    """
    Worker function: processes a single voxel YAML and writes its heatmap.
    Returns summary info for logging.
    """
    vf = Path(vf_path_str)
    vdoc = load_yaml(vf)
    voxels = np.asarray(vdoc["voxels"], dtype=np.float32)

    # Choose pose
    pose_name = infer_pose_name_from_voxel_doc_or_filename(vdoc, vf.stem)
    if pose_name is None or pose_name not in GLOBAL_POSE_TO_CFG:
        pose_name = next(iter(GLOBAL_POSE_TO_CFG.keys()))

    joint_cfg = GLOBAL_POSE_TO_CFG[pose_name]

    # Build robot meshes for this pose in base_frame
    robot_meshes = build_robot_meshes_world(
        GLOBAL_URDF,
        joint_cfg,
        base_frame=GLOBAL_BASE_FRAME
    )

    # Transform sensors for this pose
    sensors_world = []
    for s in GLOBAL_SENSORS_SRC:
        xyz_w, rpy_w = sensor_pose_to_base(
            s,
            GLOBAL_URDF,
            joint_cfg,
            base_frame=GLOBAL_BASE_FRAME
        )
        sensors_world.append({
            "xyz": xyz_w,
            "rpy": rpy_w,
            "max_range": s.get("max_range", GLOBAL_MAX_RANGE),
        })

    # Visibility with occlusion
    visible = compute_visibility_with_occlusion(
        voxels,
        sensors_world,
        robot_meshes,
        max_range=GLOBAL_MAX_RANGE,
        fov_deg=GLOBAL_FOV,
    )

    coverage = visible.sum(axis=1)
    voxel_to_sensors = [np.nonzero(visible[i])[0].tolist() for i in range(len(voxels))]

    out_doc = {
        "voxel_size_m": vdoc.get("voxel_size_m"),
        "voxel_count": int(len(voxels)),
        "sensor_count": int(len(sensors_world)),
        "pose_count": 1,
        "pose_name": pose_name,
        "fov_deg": float(GLOBAL_FOV),
        "max_range_m": float(GLOBAL_MAX_RANGE),
        "voxels": voxels.tolist(),
        **({"weights": vdoc["weights"]} if "weights" in vdoc else {}),
        "coverage": coverage.astype(int).tolist(),
        "visible_by": voxel_to_sensors,
    }

    out_path = GLOBAL_OUT_DIR / f"{vf.stem}_heatmap.yaml"
    save_yaml(out_path, out_doc)

    unseen = int((coverage == 0).sum())
    mean_cov = float(coverage.mean())

    # Return info so main process can print clean logs
    return vf.name, out_path.name, pose_name, mean_cov, unseen

# --------------- main ---------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="ur_sensor_sim/mesh_sampling/selected_candidates.yaml",
                    help="Path to sel_candidates.yaml (single file)")
    ap.add_argument("--voxels", default="ur_sensor_sim/tmp/weighted_poses",
                    help="Directory containing per-pose voxel YAMLs")
    ap.add_argument("--poses", default="ur_sensor_sim/tmp/poses.yaml",
                    help="poses.yaml with joint_names and poses (as provided)")
    ap.add_argument("--out", default="ur_sensor_sim/tmp/occlusion_heatmaps",
                    help="Output directory (one heatmap per voxel YAML)")
    ap.add_argument("--fov", type=float, default=60.0, help="Field of view (deg)")
    ap.add_argument("--max-range", type=float, default=1.5, help="Sensor max range (m)")
    ap.add_argument("--urdf", default="ur_sensor_sim/tmp/ur10.urdf", help="URDF path")
    ap.add_argument("--base-frame", default="world", help="Frame to express sensors/voxels in")
    ap.add_argument("--workers", type=int, default=os.cpu_count(),
                    help="Number of parallel worker processes (default: num CPU cores)")
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

    # Load inputs (once, then shared to workers)
    cand_doc = load_yaml(cand_path)
    sensors_src = cand_doc["candidates"]
    print(f"[INFO] Loaded {len(sensors_src)} candidate sensors from {cand_path.name}")

    pose_to_cfg, urdf_joint_names = load_pose_table(poses_path)
    print(f"[INFO] Loaded {len(pose_to_cfg)} poses from {poses_path.name}: {sorted(pose_to_cfg.keys())}")

    voxel_files = list_yaml_files(vox_dir)
    if not voxel_files:
        raise SystemExit(f"No voxel YAMLs found in {vox_dir}")

    print(f"[INFO] Processing {len(voxel_files)} voxel maps from {vox_dir} → {out_dir}")
    print(f"[INFO] Using {args.workers} worker processes")

    # Parallel execution
    vf_paths = [str(vf) for vf in voxel_files]

    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=init_worker,
        initargs=(args.urdf,
                  sensors_src,
                  pose_to_cfg,
                  args.base_frame,
                  args.fov,
                  args.max_range,
                  str(out_dir))
    ) as exe:
        futures = {exe.submit(process_voxel_file, p): p for p in vf_paths}

        for fut in as_completed(futures):
            vf_name, out_name, pose_name, mean_cov, unseen = fut.result()
            print(f"[OK] {vf_name} → {out_name} | pose={pose_name} | mean={mean_cov:.2f} | unseen={unseen}")

if __name__ == "__main__":
    main()
