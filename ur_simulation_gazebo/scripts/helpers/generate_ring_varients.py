#!/usr/bin/env python3
"""
generate_ring_variants.py  (FIXED: forearm horizontal/vertical swapped back)

Now matches your naming:

- Forearm **vertical** ring  : XY-plane around Z axis  (your first forearm script, CENTER=[-0.01,0,-0.02])
  -> we generate tilt variants here (0/15/30/45) + alternating tilt

- Forearm **horizontal** ring: YZ-plane around X axis  (your "horizontal ring" script, CENTER=[-0.10,0,0.048])
  -> we sweep CENTER.x from current x down to x=-0.5 (no tilt by default)

Upperarm parts unchanged (as requested):
- Upperarm tilt variants (all + alternating)
- Upperarm no-tilt swept along x down to x=-0.5

Outputs:
  - ring_variants_candidates.yaml   (flat list of sensors, in index order)
  - ring_variants_index_map.json    (variant_name -> {start, count, indices, params...})
"""

import math
import json
from dataclasses import dataclass, asdict
from pathlib import Path
import yaml
import numpy as np


# -------------------- small math helpers --------------------

def normalize(v):
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    return (v / n) if n > 1e-12 else v

def rpy_from_R_zyx(R):
    pitch = math.atan2(-R[2,0], math.sqrt(R[0,0]**2 + R[1,0]**2))
    cp = math.cos(pitch)
    if abs(cp) < 1e-8:
        roll = 0.0
        yaw  = math.atan2(-R[0,1], R[1,1])
    else:
        roll = math.atan2(R[2,1], R[2,2])
        yaw  = math.atan2(R[1,0], R[0,0])
    return float(roll), float(pitch), float(yaw)

def rot_about_axis(axis, angle):
    axis = normalize(axis)
    x, y, z = axis
    ca, sa = math.cos(angle), math.sin(angle)
    C = 1.0 - ca
    return np.array([
        [ca + x*x*C,     x*y*C - z*sa, x*z*C + y*sa],
        [y*x*C + z*sa,   ca + y*y*C,   y*z*C - x*sa],
        [z*x*C - y*sa,   z*y*C + x*sa, ca + z*z*C],
    ], dtype=float)

def build_basis_from_z_and_ref(z_axis, ref_axis):
    z_axis = normalize(z_axis)
    ref_axis = normalize(ref_axis)

    if abs(float(np.dot(z_axis, ref_axis))) > 0.98:
        ref_axis = np.array([0.0, 0.0, 1.0], dtype=float) if abs(z_axis[2]) < 0.98 else np.array([1.0, 0.0, 0.0], dtype=float)

    y_axis = normalize(np.cross(z_axis, ref_axis))
    x_axis = normalize(np.cross(y_axis, z_axis))
    R = np.column_stack([x_axis, y_axis, z_axis])
    return R


# -------------------- ring generators --------------------

def make_ring_YZ_about_X(parent_link, center_xyz, radius, N, offset,
                         tilt_deg=0.0, tilt_every_second=False, tilt_sign=+1):
    """
    Ring around X axis (arm axis): points in YZ plane.
    Base z_axis is radial outward.
    Tilt rotates z_axis around tangential axis (arm_axis x radial).
    """
    center = np.asarray(center_xyz, dtype=float)
    arm_axis = np.array([1.0, 0.0, 0.0], dtype=float)

    sensors = []
    for i in range(N):
        theta = i * (2.0 * math.pi / N)

        xyz = center + np.array([0.0,
                                 radius * math.cos(theta),
                                 radius * math.sin(theta)], dtype=float)

        radial = normalize([0.0, xyz[1] - center[1], xyz[2] - center[2]])
        z_axis = radial

        do_tilt = (abs(tilt_deg) > 1e-9) and ((i % 2 == 1) if tilt_every_second else True)
        if do_tilt:
            tangent = normalize(np.cross(arm_axis, radial))
            z_axis = normalize(rot_about_axis(tangent, math.radians(tilt_deg) * float(tilt_sign)) @ z_axis)

        R = build_basis_from_z_and_ref(z_axis, arm_axis)
        rpy = rpy_from_R_zyx(R)
        normal = normalize(R @ np.array([0.0, 0.0, 1.0], dtype=float))

        sensors.append({
            "link": parent_link,
            "xyz": [float(xyz[0]), float(xyz[1]), float(xyz[2])],
            "rpy": [float(rpy[0]), float(rpy[1]), float(rpy[2])],
            "normal": [float(normal[0]), float(normal[1]), float(normal[2])],
            "offset": float(offset),
        })
    return sensors

def make_ring_XY_about_Z(parent_link, center_xyz, radius, N, offset,
                         tilt_deg=0.0, tilt_every_second=False, tilt_sign=+1):
    """
    Ring around Z axis: points in XY plane.
    Base z_axis is radial outward in XY.
    Tilt rotates z_axis around tangential axis (arm_axis x radial), with arm_axis = +Z.
    """
    center = np.asarray(center_xyz, dtype=float)
    arm_axis = np.array([0.0, 0.0, 1.0], dtype=float)

    sensors = []
    for i in range(N):
        theta = i * (2.0 * math.pi / N)

        xyz = center + np.array([radius * math.cos(theta),
                                 radius * math.sin(theta),
                                 0.0], dtype=float)

        radial = normalize([xyz[0] - center[0], xyz[1] - center[1], 0.0])
        z_axis = radial

        do_tilt = (abs(tilt_deg) > 1e-9) and ((i % 2 == 1) if tilt_every_second else True)
        if do_tilt:
            tangent = normalize(np.cross(arm_axis, radial))
            z_axis = normalize(rot_about_axis(tangent, math.radians(tilt_deg) * float(tilt_sign)) @ z_axis)

        R = build_basis_from_z_and_ref(z_axis, np.array([0.0, 0.0, 1.0], dtype=float))
        rpy = rpy_from_R_zyx(R)
        normal = normalize(R @ np.array([0.0, 0.0, 1.0], dtype=float))

        sensors.append({
            "link": parent_link,
            "xyz": [float(xyz[0]), float(xyz[1]), float(xyz[2])],
            "rpy": [float(rpy[0]), float(rpy[1]), float(rpy[2])],
            "normal": [float(normal[0]), float(normal[1]), float(normal[2])],
            "offset": float(offset),
        })
    return sensors


# -------------------- variant plan --------------------

@dataclass
class VariantSpec:
    name: str
    parent_link: str
    kind: str               # "YZ_X" or "XY_Z"
    center: list
    radius: float
    N: int
    offset: float
    tilt_deg: float
    tilt_every_second: bool
    tilt_sign: int = +1


def main():
    OFFSET = 0.005

    N_FOREARM = 7
    N_UPPERARM = 8

    # --- Forearm rings (your originals) ---
    # This one was your "forearm ring" script (XY around Z)
    FOREARM_VERTICAL_CENTER_BASE = [-0.01, 0.0, -0.02]
    FOREARM_VERTICAL_RADIUS = 0.055

    # This one was your "horizontal ring" script (YZ around X)
    FOREARM_HORIZONTAL_CENTER_BASE = [-0.10, 0.0, 0.048]
    FOREARM_HORIZONTAL_RADIUS = 0.05

    # --- Upperarm ring (your geared script) ---
    UPPERARM_CENTER_BASE = [-0.11, 0.0, 0.176]
    UPPERARM_RADIUS = 0.055

    # requested tilts
    TILTS = [0.0, 15.0, 30.0, 45.0]

    # sweep x down to -0.5
    X_MIN = -0.55
    X_STEP = 0.05

    def sweep_x(x0, x_min, step):
        xs = []
        x = float(x0)
        while x >= x_min - 1e-9:
            xs.append(round(x, 6))
            x -= step
        return xs

    # IMPORTANT FIX:
    # - sweep the FOREARM HORIZONTAL ring along X
    # - tilt the FOREARM VERTICAL ring
    forearm_horizontal_xs = sweep_x(FOREARM_HORIZONTAL_CENTER_BASE[0], X_MIN, X_STEP)

    upperarm_xs = sweep_x(UPPERARM_CENTER_BASE[0], X_MIN, X_STEP)

    variants: list[VariantSpec] = []

    # 1) Forearm VERTICAL ring: different tilts + alternating tilt
    for t in TILTS:
        variants.append(VariantSpec(
            name=f"forearm_vertical_tilt_all_{int(t)}",
            parent_link="forearm_link",
            kind="XY_Z",
            center=FOREARM_VERTICAL_CENTER_BASE,
            radius=FOREARM_VERTICAL_RADIUS,
            N=N_FOREARM,
            offset=OFFSET,
            tilt_deg=t,
            tilt_every_second=False,
        ))
        variants.append(VariantSpec(
            name=f"forearm_vertical_tilt_alt_{int(t)}",
            parent_link="forearm_link",
            kind="XY_Z",
            center=FOREARM_VERTICAL_CENTER_BASE,
            radius=FOREARM_VERTICAL_RADIUS,
            N=N_FOREARM,
            offset=OFFSET,
            tilt_deg=t,
            tilt_every_second=True,
        ))

    # 2) Forearm HORIZONTAL ring: different X positions (no tilt)
    for x in forearm_horizontal_xs:
        c = [x, FOREARM_HORIZONTAL_CENTER_BASE[1], FOREARM_HORIZONTAL_CENTER_BASE[2]]
        variants.append(VariantSpec(
            name=f"forearm_horizontal_x_{x:.2f}",
            parent_link="forearm_link",
            kind="YZ_X",
            center=c,
            radius=FOREARM_HORIZONTAL_RADIUS,
            N=N_FOREARM,
            offset=OFFSET,
            tilt_deg=0.0,
            tilt_every_second=False,
        ))

    # 3) Upperarm rings with different tilts + alternating tilt
    for t in TILTS:
        variants.append(VariantSpec(
            name=f"upperarm_tilt_all_{int(t)}",
            parent_link="upper_arm_link",
            kind="YZ_X",
            center=UPPERARM_CENTER_BASE,
            radius=UPPERARM_RADIUS,
            N=N_UPPERARM,
            offset=OFFSET,
            tilt_deg=t,
            tilt_every_second=False,
        ))
        variants.append(VariantSpec(
            name=f"upperarm_tilt_alt_{int(t)}",
            parent_link="upper_arm_link",
            kind="YZ_X",
            center=UPPERARM_CENTER_BASE,
            radius=UPPERARM_RADIUS,
            N=N_UPPERARM,
            offset=OFFSET,
            tilt_deg=t,
            tilt_every_second=True,
        ))

    # 4) Upperarm rings with no tilt, moved on x axis up to x=-0.5
    for x in upperarm_xs:
        c = [x, UPPERARM_CENTER_BASE[1], UPPERARM_CENTER_BASE[2]]
        variants.append(VariantSpec(
            name=f"upperarm_notilt_x_{x:.2f}",
            parent_link="upper_arm_link",
            kind="YZ_X",
            center=c,
            radius=UPPERARM_RADIUS,
            N=N_UPPERARM,
            offset=OFFSET,
            tilt_deg=0.0,
            tilt_every_second=False,
        ))

    # ---------- build one big sensor list + index map ----------
    all_sensors = []
    index_map = {}

    for spec in variants:
        start = len(all_sensors)

        if spec.kind == "YZ_X":
            sensors = make_ring_YZ_about_X(
                parent_link=spec.parent_link,
                center_xyz=spec.center,
                radius=spec.radius,
                N=spec.N,
                offset=spec.offset,
                tilt_deg=spec.tilt_deg,
                tilt_every_second=spec.tilt_every_second,
                tilt_sign=spec.tilt_sign,
            )
        elif spec.kind == "XY_Z":
            sensors = make_ring_XY_about_Z(
                parent_link=spec.parent_link,
                center_xyz=spec.center,
                radius=spec.radius,
                N=spec.N,
                offset=spec.offset,
                tilt_deg=spec.tilt_deg,
                tilt_every_second=spec.tilt_every_second,
                tilt_sign=spec.tilt_sign,
            )
        else:
            raise ValueError(f"Unknown kind: {spec.kind}")

        all_sensors.extend(sensors)
        end = len(all_sensors)

        index_map[spec.name] = {
            "start": start,
            "count": end - start,
            "indices": list(range(start, end)),
            "params": asdict(spec),
        }

    # ---------- write outputs ----------
    out_yaml = Path("ring_variants_candidates.yaml")
    out_json = Path("ring_variants_index_map.json")

    # If your pipeline expects {"candidates":[...]} instead of a raw list, use this line:
    # yaml.safe_dump({"candidates": all_sensors}, out_yaml.open("w"), sort_keys=False)

    yaml.safe_dump(all_sensors, out_yaml.open("w"), sort_keys=False)
    out_json.write_text(json.dumps(index_map, indent=2))

    print(f"[OK] wrote {out_yaml} with {len(all_sensors)} sensors")
    print(f"[OK] wrote {out_json} with {len(index_map)} variants")
    first = next(iter(index_map.keys()))
    print(f"Example: '{first}' -> indices {index_map[first]['indices'][:5]} ...")

if __name__ == "__main__":
    main()
