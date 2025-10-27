#!/usr/bin/env python3
import argparse, math
import numpy as np
import yaml
from shapely.geometry import Polygon, Point
from pathlib import Path

# ---------- geometry functions ----------

def circle_from_3_points(p1, p2, p3):
    (x1, z1), (x2, z2), (x3, z3) = p1, p2, p3
    temp = x2**2 + z2**2
    bc = (x1**2 + z1**2 - temp) / 2.0
    cd = (temp - x3**2 - z3**2) / 2.0
    det = (x1 - x2) * (z2 - z3) - (x2 - x3) * (z1 - z2)
    if abs(det) < 1e-12:
        raise ValueError("Points are collinear; cannot define a circle.")
    cx = (bc * (z2 - z3) - cd * (z1 - z2)) / det
    cz = ((x1 - x2) * cd - (x2 - x3) * bc) / det
    r = math.hypot(x1 - cx, z1 - cz)
    return cx, cz, r

def ang(cx, cz, xz):
    x, z = xz
    return math.atan2(z - cz, x - cx)

def wrap_to(t, base):
    """Wrap angle t into [base, base+2π)."""
    while t < base: t += 2*math.pi
    while t >= base + 2*math.pi: t -= 2*math.pi
    return t

def sample_short_arc_through_mid(cx, cz, r, start, mid, end, n=40):
    """
    Return the shorter arc from 'start' to 'end' that passes through 'mid'.
    All are (x,z).
    """
    t1 = ang(cx, cz, start)
    tm = ang(cx, cz, mid)
    t2 = ang(cx, cz, end)

    # Candidate CCW arc: from t1 -> t2 (increasing)
    t2_ccw = wrap_to(t2, t1)
    tm_ccw = wrap_to(tm, t1)
    ccw_contains_mid = (t1 <= tm_ccw <= t2_ccw)
    ccw_len = t2_ccw - t1  # in [0, 2π)

    # Candidate CW arc: from t1 -> t2 (decreasing)
    # Represent as going from t1 down to t2; map to increasing by swapping roles
    t1_cw_base = t2  # wrap relative to t2
    t1_cw = wrap_to(t1, t1_cw_base)
    tm_cw = wrap_to(tm, t1_cw_base)
    cw_contains_mid = (t2 <= tm_cw <= t1_cw)
    cw_len = t1_cw - t2  # in [0, 2π)

    # Keep only arcs that contain the mid; among those choose the shorter
    choices = []
    if ccw_contains_mid:
        choices.append(("ccw", ccw_len))
    if cw_contains_mid:
        choices.append(("cw", cw_len))
    if not choices:
        # Fallback: choose the absolutely shorter arc
        if ccw_len <= cw_len:
            mode = "ccw"; arc_len = ccw_len
        else:
            mode = "cw";  arc_len = cw_len
    else:
        mode, arc_len = min(choices, key=lambda x: x[1])

    # Sample the chosen arc
    if mode == "ccw":
        thetas = np.linspace(t1, t1 + arc_len, n)
    else:
        # CW from t1 to t2 with length cw_len => decreasing; sample by reversing CCW
        thetas = np.linspace(t1, t1 - arc_len, n)

    return [(cx + r*math.cos(t), cz + r*math.sin(t)) for t in thetas]

def buffer_voxels_3d(voxels, voxel_size, radius):
    """
    Approximate 3D buffer for voxel centers.
    Adds all grid points within `radius` of each voxel center.
    """
    from itertools import product

    grid = set(map(tuple, np.round(voxels / voxel_size).astype(int)))
    offset_range = int(math.ceil(radius / voxel_size))

    new_voxels = set(grid)
    for dx, dy, dz in product(range(-offset_range, offset_range + 1),
                              range(-offset_range, offset_range + 1),
                              range(-offset_range, offset_range + 1)):
        if dx == dy == dz == 0:
            continue
        if math.sqrt(dx**2 + dy**2 + dz**2) * voxel_size <= radius:
            new_voxels.update((gx + dx, gy + dy, gz + dz) for gx, gy, gz in grid)
        print(len(new_voxels))

    new_voxels = np.array(list(new_voxels), dtype=float) * voxel_size
    return new_voxels


def generate_voxels_from_polygon(poly, voxel, width_y, zmin, zmax):
    minx, minz, maxx, maxz = poly.bounds
    xs = np.arange(minx, maxx + voxel, voxel)
    zs = np.arange(zmin, zmax + voxel, voxel)
    ys = np.arange(-width_y/2, width_y/2 + voxel, voxel)
    voxels = []
    for x in xs:
        for z in zs:
            if not poly.contains(Point(x, z)):
                continue
            for y in ys:
                voxels.append((x, y, z))
    return np.asarray(voxels, dtype=np.float32)

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voxel-size", type=float, default=0.05, help="Voxel edge (m)")
    ap.add_argument("--buffer-size", type=float, default=1.0, help="Buffer size (m)")
    ap.add_argument("--width-y", type=float, default=0.25, help="Extrusion width (m, ±y/2)")
    ap.add_argument("--out", default="ur_sensor_sim/tmp/capsule.yaml", help="Output YAML")
    ap.add_argument("--arc-samples", type=int, default=40, help="Samples along quarter-arc")
    args = ap.parse_args()

    # Key x–z points (y ignored)
    base  = (0.0, 0.25)
    top   = (0.0, 1.677201)
    mid45 = (0.978414, 1.239016)
    floor = (1.301858, 0.25)

    # Build true quarter-circle (short arc that passes through mid45)
    cx, cz, r = circle_from_3_points(top, mid45, floor)
    outer_arc = sample_short_arc_through_mid(cx, cz, r, start=top, mid=mid45, end=floor, n=args.arc_samples)

    # Inner edge (you can also curve this similarly if needed)
    inner_pts = [
        (-0.353692, 0.876735),
        (0.0, 1.677201)
    ]

    # Assemble polygon outline (x,z). Order matters; keep it non-self-intersecting.
    xz = []
    xz += [base]
    xz += outer_arc
    xz += [base]
    xz += inner_pts

    poly = Polygon(xz).buffer(0)
    if not poly.is_valid:
        raise SystemExit("Invalid polygon (self-intersecting). Try reordering points.")

    minx, minz, maxx, maxz = poly.bounds
    voxels = generate_voxels_from_polygon(poly, args.voxel_size, args.width_y, minz, maxz)
    voxels = buffer_voxels_3d(voxels, args.voxel_size, args.buffer_size)


    data = {
        "shape": "polygon_fill_extruded",
        "frame": "world",
        "voxel_size_m": float(args.voxel_size),
        "extrusion_width_y_m": float(args.width_y),
        "anchors_xz": xz,
        "circle_center_xz": [float(cx), float(cz)],
        "circle_radius": float(r),
        "voxel_count": int(len(voxels)),
        "voxels": voxels.tolist(),
    }
    Path(args.out).write_text(yaml.safe_dump(data, sort_keys=False))
    print(f"[OK] Wrote {args.out} | voxels={len(voxels):,} | arc_len≈{r*abs(math.atan2(outer_arc[-1][1]-cz, outer_arc[-1][0]-cx) - math.atan2(outer_arc[0][1]-cz, outer_arc[0][0]-cx)):.3f} m")

if __name__ == "__main__":
    main()
