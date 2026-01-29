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

def _point_in_link_frame_to_base_frame(
    urdf: URDF,
    joint_cfg: dict,
    link_name: str,
    base_frame: str | None,
    p_link: np.ndarray,
) -> np.ndarray:
    link_by_name = {L.name: L for L in urdf.links}
    if link_name not in link_by_name:
        raise ValueError(f"Link '{link_name}' not found in URDF.")

    link_fk = urdf.link_fk(cfg=joint_cfg)
    T_base_link = link_fk[link_by_name[link_name]]  # behaves as (link -> base_link)

    # If user wants coordinates in another base_frame, rebase base_link -> base_frame
    if base_frame is None or base_frame == urdf.base_link.name:
        T_out_base = np.eye(4)
    else:
        # _compute_T_req_base returns (base_frame -> base_link)
        T_req_base = _compute_T_req_base(urdf, joint_cfg, base_frame)
        T_out_base = np.linalg.inv(T_req_base)  # (base_link -> base_frame)

    p = np.asarray(p_link, dtype=float).reshape(3)
    ph = np.array([p[0], p[1], p[2], 1.0], dtype=float)

    # link -> base_link
    p_base = T_base_link @ ph
    # base_link -> base_frame
    p_out = T_out_base @ p_base
    return p_out[:3].astype(float)


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

    Conceptually (for mode="gamma") this implements a 3-region weighting:
      1) Zeroed band (unobservable / too close to mesh):
           w = 0                     for d <= zero_inside
      2) Plateau band near the robot:
           w = 1                     for zero_inside < d <= r0
         (this happens implicitly because d_eff = max(d - r0, 0) => d_eff=0,
          norm=0, falloff(0)=1, and the min_w remap keeps it at 1)
      3) Distance falloff region:
           w = min_w + (1-min_w)*falloff(norm(d_eff))   for d > r0

    Returns:
      w       : array of weights, one per voxel
      r_total : effective outer radius used (r0 + r_max_eff)
      info    : dict with some diagnostics (r0, r_max_used, zeroed_count, ...)
    """
    N = len(voxels_xyz)
    d = np.empty(N, dtype=float)

    # For each voxel, compute its true Euclidean distance 'd' to the robot collision surface.
    # Chunked for memory/performance
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        _, di, _ = closest_point(robot_mesh, voxels_xyz[start:end])  # distance from voxel to closest point on mesh
        d[start:end] = di

    # Inner saturation radius r0 ("robot_radius"):
    # We shift distances by r0 so that everything within r0 of the mesh maps to d_eff=0, which results in weight = 1
    r0 = max(0.0, float(robot_radius))
    d_eff = np.clip(d - r0, 0.0, None)

    # Effective max distance for normalization:
    # If r_max is not provided, use maximum observed d_eff in this pose.
    # Otherwise interpret r_max as an absolute distance in the original distance domain,
    # and convert it to the shifted domain by subtracting r0.
    if r_max is None:
        r_max_eff = float(np.max(d_eff)) if np.any(d_eff > 0) else 1e-6
    else:
        r_max_eff = max(1e-6, float(r_max - r0))

    # Normalize shifted distances into [0,1] and apply the chosen falloff (e.g., gamma).
    # For d <= r0: d_eff=0 -> norm=0 -> falloff(norm)=1 -> plateau at weight=1.
    norm = np.clip(d_eff / r_max_eff, 0.0, 1.0)
    w = _falloff(norm, mode, gamma, alpha)

    # Remap falloff output into [min_w, 1]:
    # ensures far voxels do not drop below min_w (except where we explicitly zero them).
    min_w = float(np.clip(min_w, 0.0, 1.0))
    w = min_w + (1.0 - min_w) * w
    w = np.clip(w, min_w, 1.0)

    # Optional: explicitly zero voxels extremely close to/inside the mesh:
    # this overrides the plateau and sets w=0 for d <= zero_inside.
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

# ---------------- Helpers ----------------

def _compute_T_req_base(urdf: URDF, joint_cfg: dict, base_frame: str | None):
    """
    Returns T_req_base = (base_frame -> base_link) transform matrix, or None if base_frame is base_link / unspecified.
    This matches the convention used in build_robot_mesh_world_combined().
    """
    if base_frame is None or base_frame == urdf.base_link.name:
        return None

    link_fk = urdf.link_fk(cfg=joint_cfg)
    link_by_name = {L.name: L for L in urdf.links}
    if base_frame not in link_by_name:
        raise ValueError(f"--base-frame '{base_frame}' is not a URDF link name.")
    req_link = link_by_name[base_frame]
    T_base_req = link_fk[req_link]          # base_link -> base_frame
    T_req_base = np.linalg.inv(T_base_req)  # base_frame -> base_link
    return T_req_base


def _link_position_in_base_frame(urdf: URDF, joint_cfg: dict, link_name: str, base_frame: str | None) -> np.ndarray:
    """Return link origin position (xyz) expressed in base_frame coordinates."""
    link_by_name = {L.name: L for L in urdf.links}
    if link_name not in link_by_name:
        raise ValueError(f"Link '{link_name}' not found in URDF. Available example: {list(link_by_name)[:10]} ...")

    link_fk = urdf.link_fk(cfg=joint_cfg)   # base_link -> link
    T_base_link = link_fk[link_by_name[link_name]]

    T_req_base = _compute_T_req_base(urdf, joint_cfg, base_frame)
    if T_req_base is None:
        T_req_link = T_base_link            # base_link frame (== base_frame)
    else:
        T_req_link = T_req_base @ T_base_link  # base_frame -> link

    return T_req_link[:3, 3].astype(float)


def _pick_first_existing_link(urdf: URDF, candidates: list[str]) -> str | None:
    names = {L.name for L in urdf.links}
    for c in candidates:
        if c in names:
            return c
    return None


def _default_ur_chain_links(urdf: URDF, tcp_link: str) -> list[str]:
    """
    UR-ish defaults. We only keep those that exist in the provided URDF.
    Ensures last element is tcp_link if possible.
    """
    names = {L.name for L in urdf.links}
    # Typical UR10 link names
    candidates = [
        "base_link",
        "shoulder_link",
        "upper_arm_link",
        "forearm_link",
        "wrist_1_link",
        "wrist_2_link",
        "wrist_3_link",
        "tool0",
        "ee_link",
    ]
    chain = [n for n in candidates if n in names]

    # Force tcp_link to be last (and present)
    if tcp_link in names:
        if tcp_link in chain:
            chain = [x for x in chain if x != tcp_link] + [tcp_link]
        else:
            chain = chain + [tcp_link]
    return chain


def _multiplier_from_polyline_projection(
    voxels_xyz: np.ndarray,
    points_xyz: np.ndarray,
    m_min: float,
    m_max: float,
) -> np.ndarray:
    """
    points_xyz: (K,3) polyline points ordered from base->tcp.
    For each voxel, project to closest segment; convert arc-length position to multiplier.
    """
    points_xyz = np.asarray(points_xyz, dtype=float)
    if points_xyz.shape[0] < 2:
        return np.ones(len(voxels_xyz), dtype=float) * float(m_max)

    segs = points_xyz[1:] - points_xyz[:-1]            # (K-1,3)
    seglen = np.linalg.norm(segs, axis=1)              # (K-1,)
    total = float(np.sum(seglen))
    if total <= 1e-12:
        return np.ones(len(voxels_xyz), dtype=float) * float(m_max)

    cum = np.concatenate([[0.0], np.cumsum(seglen)])   # (K,)

    # For each segment, compute closest point param t for all voxels (vectorized over voxels, loop over segments)
    best_dist2 = np.full(len(voxels_xyz), np.inf, dtype=float)
    best_s = np.zeros(len(voxels_xyz), dtype=float)

    for i in range(len(segs)):
        d = segs[i]
        L2 = float(np.dot(d, d))
        if L2 <= 1e-18:
            continue

        p0 = points_xyz[i]
        v = voxels_xyz - p0                 # (N,3)
        t = (v @ d) / L2                    # (N,)
        t = np.clip(t, 0.0, 1.0)
        closest = p0 + t[:, None] * d[None, :]
        dist2 = np.sum((voxels_xyz - closest) ** 2, axis=1)

        improved = dist2 < best_dist2
        if np.any(improved):
            best_dist2[improved] = dist2[improved]
            # arc-length position along chain (0..1)
            s = (cum[i] + t * seglen[i]) / total
            best_s[improved] = s[improved]

    m_min = float(m_min)
    m_max = float(m_max)
    s = np.clip(best_s, 0.0, 1.0)
    mult = m_min + (m_max - m_min) * s
    return np.clip(mult, min(m_min, m_max), max(m_min, m_max))


def _multiplier_dual_spheres(
    voxels_xyz: np.ndarray,
    base_xyz: np.ndarray,
    elbow_xyz: np.ndarray,
    tcp_xyz: np.ndarray,
    m_min: float,
    m_max: float,
) -> np.ndarray:
    """
    Two spheres centered at elbow and tcp:
      mult(center)=m_max
      mult(at radius = ||center-base||) = m_min
      linear falloff.
    Combine by max() so proximity to either center boosts multiplier.
    """
    base_xyz = np.asarray(base_xyz, dtype=float).reshape(3)
    elbow_xyz = np.asarray(elbow_xyz, dtype=float).reshape(3)
    tcp_xyz = np.asarray(tcp_xyz, dtype=float).reshape(3)

    def sphere_mult(center):
        R = float(np.linalg.norm(center - base_xyz))
        R = max(R, 1e-6)
        dist = np.linalg.norm(voxels_xyz - center[None, :], axis=1)
        # 1 at dist=0, -> m_min at dist=R, clamp beyond
        t = np.clip(dist / R, 0.0, 1.0)
        return m_max - (m_max - m_min) * t

    m1 = sphere_mult(elbow_xyz)
    m2 = sphere_mult(tcp_xyz)
    mult = np.maximum(m1, m2)
    return np.clip(mult, min(m_min, m_max), max(m_min, m_max))


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
    ap.add_argument("--out-dir", default="ur_sensor_sim/tmp/weighted_poses_tcp_10",
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
    ap.add_argument("--zero-inside", type=float, default=0.1,
                    help="Meters: if voxel is within this distance to robot surface, set weight=0")
    
        # --- weighting multiplier (along robot / fallback circles) ---
    ap.add_argument("--extra-mult-mode", choices=["off", "chain", "dual"], default="chain",
                    help="Extra multiplier on top of distance weights: "
                         "'chain' = along robot polyline base->tcp, "
                         "'dual' = spheres around elbow+tcp, "
                         "'off' = disabled.")
    ap.add_argument("--extra-mult-min", type=float, default=0.2,
                    help="Minimum multiplier at the base-most region (default 0.2).")
    ap.add_argument("--extra-mult-max", type=float, default=1.0,
                    help="Maximum multiplier near TCP / joint centers (default 1.0).")

    ap.add_argument("--chain-links", type=str, default="",
                    help="Comma-separated link names to define the robot polyline (base->...->tcp). "
                         "If empty, tries a UR-style default chain.")
    ap.add_argument("--elbow-link", type=str, default="",
                    help="Link whose origin is at the elbow joint (used for --extra-mult-mode=dual). "
                         "If empty, tries common UR names.")
    ap.add_argument("--tcp-link", type=str, default="",
                    help="Link whose origin is at the TCP/end-effector (used for chain end and dual mode). "
                         "If empty, tries common UR names (tool0/ee_link/etc.).")
    ap.add_argument("--tcp-offset-tool0", type=float, default=0.0,
                help="Meters: virtual TCP point at (0,0,tcp_offset_tool0) in tool0 frame (default 0.10 = 10cm).")

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

        # --- multiplier (0.2 -> 1.0) based on base->tcp position ---

        extra_mult = None
        extra_mult_info = {}

        if args.extra_mult_mode != "off":
            m_min = float(args.extra_mult_min)
            m_max = float(args.extra_mult_max)

            # Determine TCP link
            tcp_link = args.tcp_link.strip()
            if not tcp_link:
                tcp_link = _pick_first_existing_link(urdf, ["tool0", "ee_link", "tool_link", "wrist_3_link"]) \
                           or urdf.links[-1].name  # last-resort fallback

            base_link_name = urdf.base_link.name
            base_xyz = _link_position_in_base_frame(urdf, joint_cfg, base_link_name, base_frame=args.base_frame)

            # virtual TCP = (0,0,offset) in tool0 frame
            tcp_xyz = _point_in_link_frame_to_base_frame(
                urdf=urdf,
                joint_cfg=joint_cfg,
                link_name=tcp_link,                # typically "tool0"
                base_frame=args.base_frame,
                p_link=np.array([0.0, 0.0, float(args.tcp_offset_tool0)])
            )
            # tcp_xyz = _link_position_in_base_frame(urdf, joint_cfg, tcp_link, base_frame=args.base_frame)


            if args.extra_mult_mode == "chain":
                # Chain definition
                if args.chain_links.strip():
                    chain_links = [s.strip() for s in args.chain_links.split(",") if s.strip()]
                else:
                    chain_links = _default_ur_chain_links(urdf, tcp_link=tcp_link)

                # Convert chain link origins to points
                points = []
                for ln in chain_links:
                    try:
                        points.append(_link_position_in_base_frame(urdf, joint_cfg, ln, base_frame=args.base_frame))
                    except Exception:
                        # skip missing/problem links silently
                        pass
                points = np.asarray(points, dtype=float)

                if points.size == 0:
                    points = tcp_xyz.reshape(1, 3)
                else:
                    # If the last point is already tool0 origin, keep it and append the virtual TCP
                    points = np.vstack([points, tcp_xyz.reshape(1, 3)])

                if len(points) < 2:
                    extra_mult = np.ones(len(voxels), dtype=float) * m_max
                else:
                    extra_mult = _multiplier_from_polyline_projection(voxels, points, m_min=m_min, m_max=m_max)

                extra_mult_info = {
                    "mode": "chain",
                    "m_min": m_min,
                    "m_max": m_max,
                    "tcp_link": tcp_link,
                    "chain_links_used": chain_links,
                    "chain_points_count": int(len(points)),
                }

            elif args.extra_mult_mode == "dual":
                elbow_link = args.elbow_link.strip()
                if not elbow_link:
                    elbow_link = _pick_first_existing_link(urdf, ["forearm_link", "elbow_link", "upper_arm_link"]) \
                                 or base_link_name

                elbow_xyz = _link_position_in_base_frame(urdf, joint_cfg, elbow_link, base_frame=args.base_frame)
                extra_mult = _multiplier_dual_spheres(
                    voxels_xyz=voxels,
                    base_xyz=base_xyz,
                    elbow_xyz=elbow_xyz,
                    tcp_xyz=tcp_xyz,
                    m_min=m_min,
                    m_max=m_max,
                )
                extra_mult_info = {
                    "mode": "dual",
                    "m_min": m_min,
                    "m_max": m_max,
                    "elbow_link": elbow_link,
                    "tcp_link": tcp_link,
                }

            # Apply multiplier (keeps zeroed voxels at 0)
            w = np.clip(w * extra_mult, 0.0, 1.0)


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
        if args.extra_mult_mode != "off":
            out_doc["weighting"]["extra_multiplier"] = extra_mult_info

        out_path = out_dir / f"{pose_name}.yaml"
        out_path.write_text(yaml.safe_dump(out_doc, sort_keys=False))
        print(f"[OK] {pose_name:10s} → {out_path} | w∈[{w.min():.3f},{w.max():.3f}] | zeroed={extra['zeroed_count']}")

if __name__ == "__main__":
    main()
