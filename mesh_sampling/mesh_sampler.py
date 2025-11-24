#!/usr/bin/env python3
"""
mesh_sampler.py — Generate evenly spaced surface candidates (>=1 cm) with normals
for mounting exteroceptive sensors on a robot described by URDF/Xacro.

Outputs a YAML with candidate poses expressed in each link frame
(xyz + rpy), plus the surface normal (unit, in link frame).

Tested with ROS 2 Humble, UR family URDFs, Gazebo Classic meshes.

Dependencies:
  pip install trimesh urdfpy numpy pyyaml scipy transforms3d ament-index-python

Example:



  python3 mesh_sampler.py \
    --urdf ~/gazebo_ws/src/ur_sensor_sim/ur_tof_description/urdf/ur_with_tof.urdf.xacro \
    --spacing 0.01 \
    --offset 0.005 \
    --links base_link shoulder_link upper_arm_link forearm_link wrist_1_link wrist_2_link wrist_3_link \
    --out candidates.yaml

Optional masks (exclusions) as spheres/cylinders per link can be provided via --mask-yaml.
See function `point_masked` for schema.
"""

import argparse
import os
import sys
import math
import yaml
import numpy as np
from pathlib import Path

import trimesh
from scipy.spatial import cKDTree
from transforms3d.quaternions import mat2quat, quat2mat
from transforms3d.euler import mat2euler
from transforms3d.axangles import axangle2mat

# ROS 2 (ament) for package:// resolution
try:
    from ament_index_python.packages import get_package_share_directory
except Exception:
    get_package_share_directory = None

# urdfpy for robust URDF parsing and mesh loading
# --- NumPy compatibility shim for packages expecting np.float/np.int/np.bool (e.g., urdfpy on NumPy>=1.24/2.0)
try:
    _ = np.float  # type: ignore[attr-defined]
except AttributeError:  # NumPy >=1.24 removed aliases
    np.float = float  # type: ignore[attr-defined]
    np.int = int      # type: ignore[attr-defined]
    np.bool = bool    # type: ignore[attr-defined]

from urdfpy import URDF


def resolve_package_uri(uri: str) -> str:
    """Resolve package:// URIs using ament index (ROS 2).
    Fallback: look into ROS_PACKAGE_PATH.
    """
    if not uri.startswith('package://'):
        return uri
    rest = uri[len('package://'):]
    parts = rest.split('/', 1)
    pkg = parts[0]
    rel = parts[1] if len(parts) > 1 else ''

    # Try ament index
    if get_package_share_directory is not None:
        try:
            share = get_package_share_directory(pkg)
            return os.path.join(share, rel)
        except Exception:
            pass

    # Fallback: ROS_PACKAGE_PATH search
    rpp = os.environ.get('ROS_PACKAGE_PATH', '')
    for base in rpp.split(':'):
        cand = os.path.join(base, pkg, rel)
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(f"Could not resolve package URI: {uri}")


def load_urdf(urdf_path: str) -> URDF:
    """Load URDF. If a .xacro is given, invoke xacro to expand it.
    Additionally sanitize mesh filenames with "file://" scheme which confuses
    downstream loaders (e.g., results like /tmp/file:///opt/ros/...).
    """
    path = Path(urdf_path)
    if path.suffix == '.xacro':
        import subprocess, tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.urdf') as tmp:
            out_path = tmp.name
        cmd = ['xacro', str(path), '-o', out_path]
        try:
            subprocess.check_call(cmd)
        except Exception as e:
            print(f"[ERROR] xacro expansion failed: {e}", file=sys.stderr)
            sys.exit(2)
        # Sanitize: strip file:// and file:/// schemes in mesh filenames
        try:
            txt = Path(out_path).read_text()
            txt = txt.replace('file:///', '/').replace('file://', '/')
            Path(out_path).write_text(txt)
        except Exception as e:
            print(f"[WARN] failed to sanitize file:// URIs: {e}")
        urdf = URDF.load(out_path)
        os.unlink(out_path)
        return urdf
    else:
        # If loading a plain URDF, sanitize a copied temp file as well
        if 'file://' in Path(urdf_path).read_text(errors='ignore'):
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.urdf', mode='w') as tmp:
                txt = Path(urdf_path).read_text()
                txt = txt.replace('file:///', '/').replace('file://', '/')
                tmp.write(txt)
                tmp_path = tmp.name
            urdf = URDF.load(tmp_path)
            os.unlink(tmp_path)
            return urdf
        return URDF.load(str(path))


def even_sample_on_mesh(mesh: trimesh.Trimesh, spacing: float, oversample: float = 1.4):
    """Blue-noise-ish even sampling on mesh surface.
    Uses trimesh.sample.sample_surface_even when available; otherwise
    Poisson disc by oversampling then voxel pruning.
    Returns (points [N,3], face_indices [N], normals [N,3]).
    """
    area = mesh.area
    if area <= 0:
        return np.zeros((0,3)), np.array([], dtype=int), np.zeros((0,3))

    # target count ~ area / spacing^2
    n_target = max(1, int(area / (spacing**2)))

    try:
        pts, face_idx = trimesh.sample.sample_surface_even(mesh, n_target)
    except Exception:
        # Fallback: uniform then prune
        n0 = int(n_target * oversample)
        pts, face_idx = trimesh.sample.sample_surface(mesh, n0)

    # Compute normals per face, then per point
    face_normals = mesh.face_normals
    nrm = face_normals[face_idx]

    # Poisson/voxel pruning to enforce >= spacing
    if len(pts) == 0:
        return pts, face_idx, nrm

    voxel = spacing / math.sqrt(3)
    # quantize points to grid
    q = np.floor(pts / voxel)
    # keep first point per grid cell
    _, uniq_idx = np.unique(q, axis=0, return_index=True)
    uniq_idx = np.sort(uniq_idx)
    pts = pts[uniq_idx]
    nrm = nrm[uniq_idx]

    # normalize normals
    nrm = nrm / (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12)
    return pts, face_idx[uniq_idx], nrm


def normal_to_quat(n: np.ndarray) -> np.ndarray:
    """Quaternion [w,x,y,z] rotating +Z to n (robust to numerical noise)."""
    n = n / (np.linalg.norm(n) + 1e-12)
    z = np.array([0.0, 0.0, 1.0])
    c = float(np.dot(z, n))
    if c > 0.999999:
        R = np.eye(3)
    elif c < -0.999999:
        # 180° around any axis orthogonal to z; choose x
        R = axangle2mat([1, 0, 0], math.pi)
    else:
        v = np.cross(z, n)
        s = math.sqrt((1 + c) * 2)
        vx, vy, vz = v / (np.linalg.norm(v) + 1e-12)
        K = np.array([[0, -vz, vy], [vz, 0, -vx], [-vy, vx, 0]])
        R = np.eye(3) + K + K @ K * ((1 - c) / (np.linalg.norm(v)**2 + 1e-12))
    q = mat2quat(R)
    return q


def rpy_from_quat(q):
    """Convert wxyz quaternion to roll-pitch-yaw using transforms3d (stable)."""
    R = quat2mat(q)
    # Clamp numerical issues inside mat2euler not necessary, but safe
    # Use static XYZ convention (roll,pitch,yaw)
    r, p, y = mat2euler(R, axes='sxyz')
    return float(r), float(p), float(y)


def point_masked(p, link_name, masks):
    """Return True if point p [3] should be excluded by mask spec.
    Mask YAML schema:
    link_masks:
      <link_name>:
        spheres:
          - [cx, cy, cz, r]
        cylinders:
          - [cx, cy, cz, ax, ay, az, r, h]  # axis unit, radius, half-height
    All coordinates are in the *link frame*.
    """
    if not masks:
        return False
    m = masks.get('link_masks', {}).get(link_name, {})
    for s in m.get('spheres', []):
        cx, cy, cz, r = s
        if np.linalg.norm(p - np.array([cx, cy, cz])) <= r:
            return True
    for c in m.get('cylinders', []):
        cx, cy, cz, ax, ay, az, r, h = c
        c0 = np.array([cx, cy, cz])
        a = np.array([ax, ay, az])
        a = a / (np.linalg.norm(a) + 1e-12)
        v = p - c0
        t = np.dot(v, a)
        t = np.clip(t, -h, h)
        closest = c0 + t * a
        if np.linalg.norm(p - closest) <= r:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--urdf', default="ur_sensor_sim/tmp/ur10.urdf", help='Path to URDF or Xacro file')
    ap.add_argument('--spacing', type=float, default=0.02, help='Target spacing (m) between candidates')
    ap.add_argument('--offset', type=float, default=0.005, help='Offset (m) to place sensor origin along normal')
    ap.add_argument('--links', nargs='*', default=None, help='Subset of link names to sample (default: all visual/collision links)')
    ap.add_argument('--mask-yaml', default=None, help='Optional mask YAML to exclude areas per link')
    ap.add_argument('--out', default="ur_sensor_sim/mesh_sampling/big_candidates.yaml", help='Output YAML path for candidates')
    ap.add_argument('--use-collision', action='store_true', help='Sample collision meshes instead of visual if available')
    args = ap.parse_args()

    urdf = load_urdf(args.urdf)

    masks = None
    if args.mask_yaml:
        with open(args.mask_yaml, 'r') as f:
            masks = yaml.safe_load(f)

    all_candidates = []

    # Build link name filter
    link_filter = set(args.links) if args.links else None

    # For each link, pick geometry (visual or collision)
    for link in urdf.links:
        if link_filter and link.name not in link_filter:
            continue

        geoms = []
        elems = link.collisions if args.use_collision and link.collisions else link.visuals
        if not elems:
            continue

        for vis in elems:
            o = vis.origin if vis.origin is not None else np.eye(4)
            if vis.geometry.mesh is None:
                continue
            # urdfpy may already provide in-memory Trimesh/Scene objects in `meshes`.
            for item in vis.geometry.mesh.meshes:
                try:
                    if isinstance(item, trimesh.Trimesh):
                        mesh = item.copy()
                    elif hasattr(item, 'dump'):  # trimesh.Scene
                        mesh = trimesh.util.concatenate(tuple(m for m in item.dump().values()))
                    elif isinstance(item, str):
                        path = resolve_package_uri(item)
                        loaded = trimesh.load_mesh(path, process=True)
                        if isinstance(loaded, trimesh.Trimesh):
                            mesh = loaded
                        else:
                            mesh = trimesh.util.concatenate(tuple(m for m in loaded.dump().values()))
                    else:
                        print(f"[WARN] Unsupported mesh entry type for link {link.name}: {type(item)}")
                        continue
                    geoms.append((mesh, o))
                except Exception as e:
                    print(f"[WARN] Failed to load mesh for link {link.name}: {item}: {e}")

        if not geoms:
            continue

        # Merge into a single Trimesh in link frame
        meshes_tf = []
        for mesh, T in geoms:
            m = mesh.copy()
            m.apply_transform(T)
            meshes_tf.append(m)
        merged = trimesh.util.concatenate(meshes_tf)
        merged.remove_unreferenced_vertices()
        merged.remove_degenerate_faces()
        merged.remove_duplicate_faces()
        merged.fix_normals()

        # Sample
        pts, face_idx, nrm = even_sample_on_mesh(merged, args.spacing)
        if len(pts) == 0:
            continue

        # Optionally exclude via masks
        keep = []
        for i, p in enumerate(pts):
            if not point_masked(p, link.name, masks):
                keep.append(i)
        if len(keep) != len(pts):
            pts = pts[keep]
            nrm = nrm[keep]

        # Create candidate entries
        for p, n in zip(pts, nrm):
            # Sensor origin offset along normal (outward). Ensure normal is outward; if mesh normals flipped, keep both possibilities? Here we trust fix_normals.
            pos = p + args.offset * n
            q = normal_to_quat(n)  # wxyz
            # rpy for URDF convenience
            rpy = rpy_from_quat(q)
            cand = {
                'link': link.name,
                'xyz': [float(pos[0]), float(pos[1]), float(pos[2])],
                'rpy': [float(rpy[0]), float(rpy[1]), float(rpy[2])],
                'normal': [float(n[0]), float(n[1]), float(n[2])],
                'offset': float(args.offset),
                'note': 'Sensor +Z aligned with surface normal'
            }
            all_candidates.append(cand)

        print(f"[INFO] {link.name}: {len(pts)} candidates")

    # Sort by link name for stability
    all_candidates.sort(key=lambda c: (c['link'], c['xyz']))

    out = {
        'spacing_m': float(args.spacing),
        'offset_m': float(args.offset),
        'candidate_count': len(all_candidates),
        'candidates': all_candidates,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        yaml.safe_dump(out, f, sort_keys=False)

    print(f"[OK] Wrote {len(all_candidates)} candidates to {args.out}")


if __name__ == '__main__':
    main()
