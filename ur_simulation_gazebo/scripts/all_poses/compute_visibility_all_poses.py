#!/usr/bin/env python3
"""
compute_visibility_batch_over_poses.py

Inputs:
  --candidates : single candidates.yaml (with {candidates:[...]})
  --voxels     : single voxel YAML (with 'voxels': [[x,y,z], ...])
  --poses      : poses.yaml (list of named poses)
  --out        : output folder. One <vox_stem>_<pose_name>_heatmap.yaml per pose.

Notes:
  - Sensor local +Z is the viewing direction.
  - Symmetric cone FOV (--fov), range cutoff (--max-range).
  - Sensors are transformed through URDF FK for each pose.
  - Occlusion with trimesh raycasting.
  - Multiprocessing over poses.
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
import time

# numpy compat
if not hasattr(np, 'float'):
    np.float = float
    np.int = int
    np.bool = bool

# ---------- helpers ----------

def load_yaml(p: Path):
    return yaml.safe_load(p.read_text())

def save_yaml(p: Path, data: dict):
    p.write_text(yaml.safe_dump(data, sort_keys=False))

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

def build_robot_mesh_world_combined(urdf: URDF, joint_cfg: dict, base_frame: str | None):
    """
    Build a single combined Trimesh for the robot in the given joint_cfg,
    in base_frame coordinates.
    """
    robot_meshes = build_robot_meshes_world(urdf, joint_cfg, base_frame)
    if not robot_meshes:
        return None
    return trimesh.util.concatenate(robot_meshes)

def compute_visibility_with_occlusion_batched(voxels: np.ndarray,
                                              sensors: list,
                                              robot_mesh: trimesh.Trimesh | None,
                                              max_range=1.5,
                                              fov_deg=60.0,
                                              eps=1e-4,
                                              pose_name: str | None=None):
    """
    Occlusion-aware visibility using a *single* combined robot mesh and
    batched raycasts per sensor.
    """
    Nvox = len(voxels)
    Nsens = len(sensors)
    visible = np.zeros((Nvox, Nsens), dtype=bool)

    if robot_mesh is None:
        # Fallback: no robot geometry -> no occlusion
        return compute_visibility(voxels, sensors, max_range=max_range, fov_deg=fov_deg)

    half_fov_cos = math.cos(math.radians(fov_deg) / 2.0)

    t0 = time.time()                     # NEW: start timing
    last_print = t0                      # NEW: last time we printed

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
        if candidates.size == 0:
            continue

        # Batched ray origins & directions
        origins = np.repeat(origin[None, :], candidates.size, axis=0)
        directions = dirv[candidates]

        # One big ray query for all candidate voxels
        locations, index_ray, index_tri = robot_mesh.ray.intersects_location(
            origins,
            directions,
            multiple_hits=False
        )

        blocked = np.zeros(candidates.size, dtype=bool)

        if len(locations) > 0:
            # Distance from sensor to voxel (ground truth)
            voxel_dists = dist[candidates][index_ray]
            # Distance from sensor to hit
            hit_dists = np.linalg.norm(locations - origins[index_ray], axis=1)

            # Mark rays as blocked if hit is before voxel
            mask = hit_dists < voxel_dists - eps
            blocked[index_ray[mask]] = True

        # Voxels that are in FOV+range and not blocked
        visible[candidates[~blocked], j] = True

        now = time.time()
            # print at most every ~5 seconds or at fixed sensor intervals
        if (now - last_print > 5.0) or ((j + 1) % 50 == 0) or (j == 0):
            elapsed = now - t0
            avg_per_sensor = elapsed / (j + 1)
            name = pose_name if pose_name is not None else "pose"
            print(f"[PROGRESS] {name}: {j+1}/{Nsens} sensors "
                    f"| elapsed={elapsed:.1f}s | avg_per_sensor={avg_per_sensor:.3f}s")
            last_print = now

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

# ---------- globals used by workers ----------

GLOBAL_URDF = None
GLOBAL_SENSORS_SRC = None
GLOBAL_POSE_TO_CFG = None
GLOBAL_BASE_FRAME = None
GLOBAL_FOV = None
GLOBAL_MAX_RANGE = None
GLOBAL_OUT_DIR = None
GLOBAL_VOXELS = None
GLOBAL_VOXEL_SIZE = None
GLOBAL_VOXEL_WEIGHTS = None
GLOBAL_VOXEL_STEM = None

def init_worker(urdf_path: str,
                sensors_src: list,
                pose_to_cfg: dict,
                base_frame: str,
                fov: float,
                max_range: float,
                out_dir: str,
                voxels: np.ndarray,
                voxel_size,
                voxel_weights,
                voxel_stem: str):
    """
    Initializer for each worker process: loads URDF once, stores shared data.
    """
    global GLOBAL_URDF, GLOBAL_SENSORS_SRC, GLOBAL_POSE_TO_CFG
    global GLOBAL_BASE_FRAME, GLOBAL_FOV, GLOBAL_MAX_RANGE, GLOBAL_OUT_DIR
    global GLOBAL_VOXELS, GLOBAL_VOXEL_SIZE, GLOBAL_VOXEL_WEIGHTS, GLOBAL_VOXEL_STEM

    GLOBAL_URDF = URDF.load(urdf_path)
    GLOBAL_SENSORS_SRC = sensors_src
    GLOBAL_POSE_TO_CFG = pose_to_cfg
    GLOBAL_BASE_FRAME = base_frame
    GLOBAL_FOV = fov
    GLOBAL_MAX_RANGE = max_range
    GLOBAL_OUT_DIR = Path(out_dir)

    GLOBAL_VOXELS = voxels
    GLOBAL_VOXEL_SIZE = voxel_size
    GLOBAL_VOXEL_WEIGHTS = voxel_weights
    GLOBAL_VOXEL_STEM = voxel_stem

def process_pose(pose_name: str):
    """
    Worker function: compute visibility for a single pose, write heatmap.
    Returns summary info for logging.
    """
    start = time.time()

    joint_cfg = GLOBAL_POSE_TO_CFG[pose_name]

    # Robot mesh for this pose
    robot_mesh = build_robot_mesh_world_combined(
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
    visible = compute_visibility_with_occlusion_batched(
        GLOBAL_VOXELS,
        sensors_world,
        robot_mesh,
        max_range=GLOBAL_MAX_RANGE,
        fov_deg=GLOBAL_FOV,
        pose_name=pose_name
    )

    coverage = visible.sum(axis=1)
    voxel_to_sensors = [np.nonzero(visible[i])[0].tolist()
                        for i in range(len(GLOBAL_VOXELS))]

    out_doc = {
        "voxel_size_m": GLOBAL_VOXEL_SIZE,
        "voxel_count": int(len(GLOBAL_VOXELS)),
        "sensor_count": int(len(sensors_world)),
        "pose_count": 1,
        "pose_name": pose_name,
        "fov_deg": float(GLOBAL_FOV),
        "max_range_m": float(GLOBAL_MAX_RANGE),
        "voxels": GLOBAL_VOXELS.tolist(),
        **({"weights": GLOBAL_VOXEL_WEIGHTS} if GLOBAL_VOXEL_WEIGHTS is not None else {}),
        "coverage": coverage.astype(int).tolist(),
        "visible_by": voxel_to_sensors,
    }

    out_path = GLOBAL_OUT_DIR / f"{GLOBAL_VOXEL_STEM}_{pose_name}_heatmap.yaml"
    save_yaml(out_path, out_doc)

    unseen = int((coverage == 0).sum())
    mean_cov = float(coverage.mean())
    elapsed = time.time() - start

    return pose_name, out_path.name, mean_cov, unseen, elapsed

# --------------- main ---------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="ur_sensor_sim/mesh_sampling/selected_candidates.yaml",
                    help="Path to sel_candidates.yaml (single file)")
    ap.add_argument("--voxels", default="ur_sensor_sim/tmp/capsule.yaml",
                    help="Single voxel YAML")
    ap.add_argument("--poses", default="ur_sensor_sim/tmp/poses.yaml",
                    help="poses.yaml with joint_names and poses")
    ap.add_argument("--out", default="ur_sensor_sim/tmp/occlusion_heatmaps_best",
                    help="Output directory (one heatmap per pose)")
    ap.add_argument("--fov", type=float, default=60.0, help="Field of view (deg)")
    ap.add_argument("--max-range", type=float, default=1.5, help="Sensor max range (m)")
    ap.add_argument("--urdf", default="ur_sensor_sim/tmp/ur10.urdf", help="URDF path")
    ap.add_argument("--base-frame", default="world", help="Frame to express sensors/voxels in")
    ap.add_argument("--workers", type=int, default=os.cpu_count(),
                    help="Number of parallel worker processes (default: num CPU cores)")
    args = ap.parse_args()

    cand_path = Path(args.candidates)
    vox_path  = Path(args.voxels)
    poses_path= Path(args.poses)
    out_dir   = Path(args.out)

    if not cand_path.is_file():
        raise SystemExit(f"--candidates must be a file: {cand_path}")
    if not vox_path.is_file():
        raise SystemExit(f"--voxels must be a single YAML file: {vox_path}")
    if not poses_path.is_file():
        raise SystemExit(f"--poses must be a file: {poses_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Load inputs (once, then shared to workers)
    cand_doc = load_yaml(cand_path)
    sensors = cand_doc["candidates"]
    indices = [
    0,
    29,
    58,
    62,
    65,
    225,
    243,
    247,
    302,
    304,
    324,
    501,
    598,
    645,
    683,
    684,
    688,
    702,
    705,
    909
  ]
    sensors_src = [sensors[i] for i in indices]
    print(f"[INFO] Loaded {len(sensors_src)} candidate sensors from {cand_path.name}")

    vdoc = load_yaml(vox_path)
    voxels = np.asarray(vdoc["voxels"], dtype=np.float32)
    voxel_size = vdoc.get("voxel_size_m")
    voxel_weights = vdoc.get("weights", None)
    voxel_stem = vox_path.stem

    pose_to_cfg, urdf_joint_names = load_pose_table(poses_path)
    pose_names = list(pose_to_cfg.keys())
    print(f"[INFO] Loaded {len(pose_names)} poses from {poses_path.name}: {pose_names}")
    print(f"[INFO] 1 voxel map from {vox_path.name}, {len(voxels)} voxels")
    print(f"[INFO] Output dir: {out_dir}")
    print(f"[INFO] Using up to {args.workers} worker processes")

    # Limit workers to number of poses
    workers = min(args.workers, len(pose_names))

    t0 = time.time()

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=init_worker,
        initargs=(args.urdf,
                  sensors_src,
                  pose_to_cfg,
                  args.base_frame,
                  args.fov,
                  args.max_range,
                  str(out_dir),
                  voxels,
                  voxel_size,
                  voxel_weights,
                  voxel_stem)
    ) as exe:
        futures = {exe.submit(process_pose, name): name for name in pose_names}

        done = 0
        total = len(futures)

        for fut in as_completed(futures):
            pose_name, out_name, mean_cov, unseen, elapsed = fut.result()
            done += 1
            total_elapsed = time.time() - t0
            avg_per_pose = total_elapsed / done
            print(f"[OK] pose={pose_name} → {out_name} | mean={mean_cov:.2f} "
                  f"| unseen={unseen} | pose_time={elapsed:.1f}s")
            print(f"[TIME] processed {done}/{total} poses | "
                  f"elapsed={total_elapsed:.1f}s | avg_per_pose={avg_per_pose:.1f}s")

if __name__ == "__main__":
    main()
