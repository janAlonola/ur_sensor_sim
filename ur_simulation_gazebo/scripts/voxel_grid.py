#!/usr/bin/env python3
"""
workspace_rounded_triangle_voxels.py – Generate a rounded (Reuleaux) triangular prism workspace
for a UR10-like robot.

Top view:   rounded triangle (intersection of 3 circles, each centered at triangle vertex)
Side view:  rectangular extrusion of height 'height'.
"""

import numpy as np
import yaml
from pathlib import Path
import argparse
import math


def generate_reuleaux_prism(reach=1.3, height=1.2, step=0.05, width=None):
    """
    Generate voxel centers inside a rounded (Reuleaux) triangular prism.
    - reach: distance from center to triangle vertices (m)
    - height: prism height (m)
    - step: voxel spacing (m)
    - width: optional; if given, inflates each circle radius (reach + width/2)
    """
    R = reach
    if width is None:
        width = 0.0
    r = R + width / 2.0  # effective circle radius

    # Equilateral triangle vertices
    V1 = np.array([ R,  0.0])
    V2 = np.array([-R/2,  math.sqrt(3)/2 * R])
    V3 = np.array([-R/2, -math.sqrt(3)/2 * R])

    # Bounding box
    xmin, xmax = -R - width, R + width
    ymin, ymax = -R - width, R + width
    zmin, zmax = 0.0, height

    xs = np.arange(xmin, xmax + step, step)
    ys = np.arange(ymin, ymax + step, step)
    zs = np.arange(zmin, zmax + step, step)

    voxels = []
    for z in zs:
        for x in xs:
            for y in ys:
                p = np.array([x, y])
                d1 = np.linalg.norm(p - V1)
                d2 = np.linalg.norm(p - V2)
                d3 = np.linalg.norm(p - V3)
                # Reuleaux condition: inside all 3 circles
                if d1 <= r and d2 <= r and d3 <= r:
                    voxels.append((x, y, z))

    return np.array(voxels, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voxel-size", type=float, default=0.05, help="Voxel edge length (m)")
    ap.add_argument("--reach", type=float, default=1.3, help="UR10 reach (m)")
    ap.add_argument("--height", type=float, default=1.2, help="Workspace height (m)")
    ap.add_argument("--width", type=float, default=1.5, help="Additional rounding width (m)")
    ap.add_argument("--out", default="tmp/workspace_prism.yaml", help="Output YAML file")
    args = ap.parse_args()

    voxels = generate_reuleaux_prism(
        reach=args.reach,
        height=args.height,
        step=args.voxel_size,
        width=args.width
    )

    print(f"[INFO] Generated {len(voxels):,} voxel centers "
          f"({args.voxel_size*100:.0f} mm resolution)")

    data = {
        "shape": "rounded_triangle_prism",
        "voxel_size_m": float(args.voxel_size),
        "reach_m": float(args.reach),
        "height_m": float(args.height),
        "width_m": float(args.width),
        "voxel_count": int(len(voxels)),
        "voxels": voxels.tolist(),
    }

    Path(args.out).write_text(yaml.safe_dump(data, sort_keys=False))
    print(f"[OK] Wrote voxel workspace to {args.out}")


if __name__ == "__main__":
    main()
