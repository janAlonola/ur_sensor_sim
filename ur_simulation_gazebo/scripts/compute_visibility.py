#!/usr/bin/env python3
"""
compute_visibility.py
----------------------------

Given:
 - candidates.yaml  (sensor poses, orientations, max_range, etc.)
 - capsule.yaml  (voxel centers)

Compute which sensors can see which voxels (within FOV & range).
Output:
 - heatmap.yaml  (per-voxel coverage count and visibility list)

Assumptions:
 - Sensor local +Z axis is the viewing direction (or was it y?)
 - Field of view is symmetric cone (horizontal_fov_deg)
 - No occlusion check (line-of-sight optional extension)
"""

import argparse
import math
import numpy as np
if not hasattr(np, 'float'):
    np.float = float
    np.int = int
    np.bool = bool
import yaml
from pathlib import Path
import transforms3d as t3d
from urdfpy import URDF
from transforms3d.euler import euler2mat, mat2euler

# --------------------------
# Helper functions
# --------------------------

def rpy_to_matrix(rpy):
    """Convert roll, pitch, yaw (radians) to 3x3 rotation matrix."""
    return t3d.euler.euler2mat(*rpy, axes='sxyz')

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
def sensor_pose_to_base(sensor, urdf: URDF, joint_cfg=None, base_frame=None):
    """
    Transform a sensor pose (given in its parent link frame) to the URDF base frame.
    If your voxels are expressed in a different frame (base_frame), provide its link name
    and we'll transform from URDF base -> that frame as well.

    sensor: dict with keys {'link','xyz','rpy'}
    urdf  : urdfpy.URDF
    joint_cfg: dict {joint_name: position} (None -> all zeros)
    base_frame: None (use URDF base_link), or link name to re-base output.
    returns: (xyz_out[3], rpy_out[3]) in the requested base frame
    """
    if joint_cfg is None:
        joint_cfg = {}

    # FK: dict {Link: 4x4}, from URDF base_link to each link
    fk = urdf.link_fk(cfg=joint_cfg)

    # Name -> Link
    link_by_name = {L.name: L for L in urdf.links}
    parent_link = link_by_name[sensor['link']]

    # Transform (URDF base -> sensor's parent link)
    T_base_parent = fk[parent_link]

    # Sensor pose in parent link frame
    xyz_local = np.array(sensor['xyz'], dtype=float)
    rpy_local = np.array(sensor['rpy'], dtype=float)
    R_local = euler2mat(*rpy_local, axes='sxyz')

    T_parent_sensor = np.eye(4)
    T_parent_sensor[:3, :3] = R_local
    T_parent_sensor[:3, 3]  = xyz_local

    # Compose (URDF base -> sensor)
    T_base_sensor = T_base_parent @ T_parent_sensor

    # If the requested output frame is not the URDF base, re-base
    if base_frame is not None and base_frame != urdf.base_link.name:
        # Transform (URDF base -> requested base_frame)
        req_link = link_by_name[base_frame]
        T_base_req = fk[req_link]
        # We want pose in 'base_frame':  T_req_sensor = inv(T_base_req) @ T_base_sensor
        T_req_sensor = np.linalg.inv(T_base_req) @ T_base_sensor
        R_out = T_req_sensor[:3, :3]
        p_out = T_req_sensor[:3, 3]
    else:
        R_out = T_base_sensor[:3, :3]
        p_out = T_base_sensor[:3, 3]

    rpy_out = mat2euler(R_out, axes='sxyz')
    return p_out, np.array(rpy_out)

def compute_visibility(voxels, sensors, max_range=3.5, fov_deg=60.0):
    """
    Compute NxM boolean visibility matrix:
      visible[i, j] = True if voxel i is visible by sensor j
    """
    Nvox = len(voxels)
    Nsens = len(sensors)
    visible = np.zeros((Nvox, Nsens), dtype=bool)

    fov = math.radians(fov_deg)
    half_fov_cos = math.cos(fov / 2.0)

    for j, s in enumerate(sensors):
        origin = np.array(s["xyz"])
        R = rpy_to_matrix(s["rpy"])
        z_axis = R[:, 2]   # +Z is viewing direction

        diff = voxels - origin
        dist = np.linalg.norm(diff, axis=1)
        dir = diff / (dist[:, None] + 1e-12)

        # Cosine of angle between voxel direction and sensor +Z
        cosang = np.dot(dir, z_axis)
        within_fov = cosang >= half_fov_cos
        within_range = dist <= s.get("max_range", max_range)

        visible[:, j] = within_fov & within_range

    return visible

def print_urdf_link_transforms(urdf, joint_cfg=None, base_frame=None, limit=None):
    """
    Print all URDF link transforms in the current configuration.
    Helps debug misaligned joints or base transforms.
    """
    if joint_cfg is None:
        joint_cfg = {}

    fk = urdf.link_fk(cfg=joint_cfg)
    link_by_name = {L.name: L for L in urdf.links}

    if base_frame is not None and base_frame != urdf.base_link.name:
        req_link = link_by_name[base_frame]
        T_base_req = fk[req_link]
    else:
        T_base_req = np.eye(4)

    print("──────────────────────────────")
    print("📦 URDF LINK TRANSFORMS")
    print(f"Base link: {urdf.base_link.name}")
    if base_frame:
        print(f"Rebased to: {base_frame}")

    count = 0
    for link, T in fk.items():
        # Transform into base_frame if needed
        T_rel = np.linalg.inv(T_base_req) @ T
        xyz = T_rel[:3, 3]
        rpy = mat2euler(T_rel[:3, :3], axes="sxyz")
        print(f" {link.name:20s} → xyz=({xyz[0]:.3f},{xyz[1]:.3f},{xyz[2]:.3f})  "
              f"rpy=({rpy[0]:.2f},{rpy[1]:.2f},{rpy[2]:.2f})")
        count += 1
        if limit and count >= limit:
            print(f"… truncated ({len(fk)} total links)")
            break

    print("──────────────────────────────")



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="ur_sensor_sim/mesh_sampling/candidates.yaml", help="Path to candidates.yaml")
    ap.add_argument("--voxels", default="ur_sensor_sim/tmp/capsule.yaml", help="Path to workspace_voxels.yaml")
    ap.add_argument("--out", default="heatmap.yaml", help="Output YAML file")
    ap.add_argument("--hfov", type=float, default=60.0, help="Horizontal field of view (deg)")
    ap.add_argument("--max-range", type=float, default=0.5, help="Sensor max range (m)")
    args = ap.parse_args()

    # Load data
    cand = load_yaml(args.candidates)
    vox  = load_yaml(args.voxels)

    sensors = cand["candidates"]
    voxels = np.array(vox["voxels"], dtype=np.float32)

    print(f"[INFO] Loaded {len(sensors)} sensors and {len(voxels)} voxels.")

    # If you want current robot state instead of zeros, fill joint_cfg = {'shoulder_pan_joint': val, ...}
    joint_cfg = {'shoulder_pan_joint': 0.0, 'shoulder_lift_joint': 0.0, 'elbow_joint': -1.5708,
            'wrist_1_joint': 0.0, 'wrist_2_joint': 0.0, 'wrist_3_joint': 0.0}
    
    """
    Create urdf from xacro
    > xacro src/ur_sensor_sim/ur_tof_description/urdf/ur_with_tof.urdf.xacro -o  /tmp/ur10_expanded.urdf
    > strip all file:// prefixes
    """
    
    urdf = URDF.load("ur_sensor_sim/tmp/ur10.urdf")

    base_frame = "world" #"world"  # None or world?

    sensors_world = []
    for s in sensors:   # each has 'link','xyz','rpy', optionally 'max_range'
        xyz_w, rpy_w = sensor_pose_to_base(s, urdf, joint_cfg, base_frame=base_frame)
        sensors_world.append({
            'xyz': xyz_w,
            'rpy': rpy_w,
            'max_range': s.get('max_range', 0.5),
        })
    #sensors_world = [sensors_world[-1]]

    # Compute visibility
    visible = compute_visibility(voxels, sensors_world, args.max_range, args.hfov)
    coverage = visible.sum(axis=1)
    #voxel_to_sensors = [np.nonzero(visible[i])[0].tolist() for i in range(len(voxels))]

    print(f"[INFO] Average coverage: {coverage.mean():.2f} sensors per voxel")
    print(f"[INFO] {np.count_nonzero(coverage==0)} voxels are unseen by any sensor.")

    # Save result
    data = {
        "voxel_size_m": vox["voxel_size_m"],
        "voxel_count": int(len(voxels)),
        "sensor_count": int(len(sensors)),
        "hfov_deg": float(args.hfov),
        "max_range_m": float(args.max_range),
        "voxels": voxels.tolist(),
        "coverage": coverage.tolist(),
    #    "visible_by": voxel_to_sensors,
    }

    Path(args.out).write_text(yaml.safe_dump(data, sort_keys=False))
    print(f"[OK] Wrote {args.out}")

    print_urdf_link_transforms(urdf, joint_cfg, base_frame="world", limit=20)

if __name__ == "__main__":
    main()
