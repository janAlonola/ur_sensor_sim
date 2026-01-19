#!/usr/bin/env python3
"""
generate_sensor_variants_configurable.py

Config-driven sensor variant generator.

Enhancement:
- Sensor families can optionally add a "center sensor" (e.g. for cap rings),
  placed at the ring center with configurable orientation.
"""

import math
import json
from dataclasses import dataclass, field
from pathlib import Path
import yaml
import numpy as np


# -------------------- small math helpers --------------------

def normalize(v):
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    return (v / n) if n > 1e-12 else v

def rpy_from_R_zyx(R):
    pitch = math.atan2(-R[2, 0], math.sqrt(R[0, 0]**2 + R[1, 0]**2))
    cp = math.cos(pitch)
    if abs(cp) < 1e-8:
        roll = 0.0
        yaw = math.atan2(-R[0, 1], R[1, 1])
    else:
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])
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
    """
    Build rotation matrix whose 3rd column is z_axis, and x/y are chosen stably using ref_axis.
    """
    z_axis = normalize(z_axis)
    ref_axis = normalize(ref_axis)

    # If ref axis is nearly parallel, pick a different one
    if abs(float(np.dot(z_axis, ref_axis))) > 0.98:
        ref_axis = np.array([0.0, 0.0, 1.0], dtype=float) if abs(z_axis[2]) < 0.98 else np.array([1.0, 0.0, 0.0], dtype=float)

    y_axis = normalize(np.cross(z_axis, ref_axis))
    x_axis = normalize(np.cross(y_axis, z_axis))
    R = np.column_stack([x_axis, y_axis, z_axis])
    return R


# -------------------- center sensor helper --------------------

def make_center_sensor(parent_link, center_xyz, offset,
                       z_axis=(0.0, 0.0, -1.0), ref_axis=(1.0, 0.0, 0.0)):
    """
    Add one sensor at center_xyz, oriented so sensor +Z points along z_axis.
    ref_axis defines the "x-ish" direction (must not be parallel to z_axis).
    """
    center = np.asarray(center_xyz, dtype=float)
    z_axis = normalize(np.asarray(z_axis, dtype=float))
    ref_axis = normalize(np.asarray(ref_axis, dtype=float))

    R = build_basis_from_z_and_ref(z_axis, ref_axis)
    rpy = rpy_from_R_zyx(R)
    normal = normalize(R @ np.array([0.0, 0.0, 1.0], dtype=float))

    return {
        "link": parent_link,
        "xyz": [float(center[0]), float(center[1]), float(center[2])],
        "rpy": [float(rpy[0]), float(rpy[1]), float(rpy[2])],
        "normal": [float(normal[0]), float(normal[1]), float(normal[2])],
        "offset": float(offset),
    }


# -------------------- ring generators --------------------

def make_ring_YZ_about_X(parent_link, center_xyz, radius, N, offset,
                         tilt_deg=0.0, tilt_every_second=False, tilt_sign=+1,
                         add_center_sensor=False,
                         center_z_axis=(0.0, 0.0, -1.0),
                         center_ref_axis=(1.0, 0.0, 0.0)):
    """
    Ring around X axis: points in YZ plane.
    Base z_axis is radial outward.
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

    if add_center_sensor:
        sensors.append(make_center_sensor(
            parent_link=parent_link,
            center_xyz=center,
            offset=offset,
            z_axis=center_z_axis,
            ref_axis=center_ref_axis,
        ))

    return sensors

def make_ring_XY_about_Z(parent_link, center_xyz, radius, N, offset,
                         tilt_deg=0.0, tilt_every_second=False, tilt_sign=+1,
                         add_center_sensor=False,
                         center_z_axis=(0.0, 0.0, -1.0),
                         center_ref_axis=(1.0, 0.0, 0.0)):
    """
    Ring around Z axis: points in XY plane.
    Base z_axis is radial outward in XY.
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

    if add_center_sensor:
        sensors.append(make_center_sensor(
            parent_link=parent_link,
            center_xyz=center,
            offset=offset,
            z_axis=center_z_axis,
            ref_axis=center_ref_axis,
        ))

    return sensors

def make_ring_XZ_about_Y(parent_link, center_xyz, radius, N, offset,
                         tilt_deg=0.0, tilt_every_second=False, tilt_sign=+1,
                         add_center_sensor=False,
                         center_z_axis=(0.0, 0.0, -1.0),
                         center_ref_axis=(1.0, 0.0, 0.0)):
    """
    Ring around Y axis: points in XZ plane.
    Base z_axis is radial outward in XZ.
    Tilt rotates z_axis around tangential axis (arm_axis x radial), with arm_axis = +Y.
    """
    center = np.asarray(center_xyz, dtype=float)
    arm_axis = np.array([0.0, 1.0, 0.0], dtype=float)

    sensors = []
    for i in range(N):
        theta = i * (2.0 * math.pi / N)

        # ring in XZ plane around Y
        xyz = center + np.array([radius * math.cos(theta),
                                 0.0,
                                 radius * math.sin(theta)], dtype=float)

        radial = normalize([xyz[0] - center[0], 0.0, xyz[2] - center[2]])
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

    if add_center_sensor:
        sensors.append(make_center_sensor(
            parent_link=parent_link,
            center_xyz=center,
            offset=offset,
            z_axis=center_z_axis,
            ref_axis=center_ref_axis,
        ))

    return sensors


# -------------------- config model --------------------

@dataclass
class TiltConfig:
    tilts_deg: list[float] = field(default_factory=list)
    include_all: bool = True
    include_alt: bool = True
    tilt_sign: int = +1

@dataclass
class SweepXConfig:
    x_min: float = -0.55
    x_step: float = 0.05
    name_prefix: str = "x"

@dataclass
class CenterSensorConfig:
    enabled: bool = False
    z_axis: list[float] = field(default_factory=lambda: [0.0, 0.0, -1.0])  # like your example
    ref_axis: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0])

@dataclass
class SensorFamily:
    name: str
    link: str
    kind: str            # "YZ_X" or "XY_Z"
    center: list[float]  # [x,y,z]
    radius: float
    N: int
    offset: float

    tilts: TiltConfig | None = None
    sweep_x: SweepXConfig | None = None
    center_sensor: CenterSensorConfig = field(default_factory=CenterSensorConfig)


# -------------------- variant expansion --------------------

def sweep_x_values(x0: float, x_min: float, step: float) -> list[float]:
    xs = []
    x = float(x0)
    while x >= x_min - 1e-9:
        xs.append(round(x, 6))
        x -= float(step)
    return xs

def build_variants_from_family(fam: SensorFamily) -> list[dict]:
    """
    Returns list of concrete variant specs dicts.
    """
    base = {
        "family": fam.name,
        "parent_link": fam.link,
        "kind": fam.kind,
        "center": list(map(float, fam.center)),
        "radius": float(fam.radius),
        "N": int(fam.N),
        "offset": float(fam.offset),
        "tilt_deg": 0.0,
        "tilt_every_second": False,
        "tilt_sign": +1,
        "center_sensor_enabled": bool(fam.center_sensor.enabled),
        "center_sensor_z_axis": list(map(float, fam.center_sensor.z_axis)),
        "center_sensor_ref_axis": list(map(float, fam.center_sensor.ref_axis)),
    }

    specs = []

    # Tilt variants
    if fam.tilts and fam.tilts.tilts_deg:
        for t in fam.tilts.tilts_deg:
            if fam.tilts.include_all:
                s = dict(base)
                s["tilt_deg"] = float(t)
                s["tilt_every_second"] = False
                s["tilt_sign"] = int(fam.tilts.tilt_sign)
                s["name"] = f"{fam.name}_tilt_all_{int(round(t))}"
                specs.append(s)
            if fam.tilts.include_alt:
                s = dict(base)
                s["tilt_deg"] = float(t)
                s["tilt_every_second"] = True
                s["tilt_sign"] = int(fam.tilts.tilt_sign)
                s["name"] = f"{fam.name}_tilt_alt_{int(round(t))}"
                specs.append(s)
    else:
        s = dict(base)
        s["name"] = fam.name
        specs.append(s)

    # X-sweep (notilt) variants
    if fam.sweep_x:
        xs = sweep_x_values(fam.center[0], fam.sweep_x.x_min, fam.sweep_x.x_step)
        for x in xs:
            s = dict(base)
            s["center"] = [float(x), float(fam.center[1]), float(fam.center[2])]
            s["tilt_deg"] = 0.0
            s["tilt_every_second"] = False
            s["tilt_sign"] = +1
            s["name"] = f"{fam.name}_{fam.sweep_x.name_prefix}_{x:.2f}"
            specs.append(s)

    # Dedup
    seen = set()
    uniq = []
    for s in specs:
        if s["name"] in seen:
            continue
        seen.add(s["name"])
        uniq.append(s)
    return uniq


# -------------------- USER CONFIG: edit only this block --------------------

def build_config():
    OFFSET = 0.005
    TILTS = [0.0, 10.0, 15.0, 30.0, 45.0, 60.0]
    TILTS_BIG = [0.0, -10.0, -15.0, -30.0, -45.0, -60.0, -75.0, -90.0]
    TILTS_NEG = [-t for t in TILTS]
    SWEEP = SweepXConfig(x_min=-0.55, x_step=0.05, name_prefix="x")

    families: list[SensorFamily] = [
        # Example: make "cap" families add a center sensor like your snippet.
        # Use center_sensor.enabled=True and optionally override z_axis/ref_axis.

        # base_cap (1) ??
        # upperarm_cap_1 (2)    ✅
        # upperarm_ring         ✅
        # upperarm_cap_2 (3)    ✅
        # forearm_cap_1 (4)     ✅
        # forearm_ring          ✅
        # forearm_cap_2 (5)     ✅
        # wrist_cap_1 (6) 
        # wrist_cap_2 (7)
        # flange_ring (F)

        #2 72,2 92,2 120

        SensorFamily(
            name="flange_ring",                # horizontal ring
            link="wrist_3_link",
            kind="XY_Z",
            center=[0.0, 0.0, 0.0],
            radius=0.05,
            N=7,
            offset=OFFSET,
            tilts=TiltConfig(tilts_deg=TILTS_BIG, include_all=True, include_alt=False),
            center_sensor=CenterSensorConfig(
                enabled=True,
                z_axis=[0.0, 0.0, 1.0],        # exactly like your example center sensor
                ref_axis=[1.0, 0.0,  0.0],
            ),
        ),
        SensorFamily(
            name="wrist_cap_2",                # horizontal ring
            link="wrist_2_link",
            kind="XZ_Y",
            center=[0.0, -0.06, 0.0],
            radius=0.035,
            N=7,
            offset=OFFSET,
            tilts=TiltConfig(tilts_deg=TILTS, include_all=True, include_alt=False),
            center_sensor=CenterSensorConfig(
                enabled=True,
                z_axis=[0.0, -1.0, 0.0],        # exactly like your example center sensor
                ref_axis=[1.0, 0.0,  0.0],
            ),
        ),
        SensorFamily(
            name="wrist_cap_1",                # horizontal ring
            link="wrist_1_link",
            kind="XZ_Y",
            center=[0.0, 0.058, 0.0],
            radius=0.035,
            N=7,
            offset=OFFSET,
            tilts=TiltConfig(tilts_deg=TILTS_NEG, include_all=True, include_alt=False),
            center_sensor=CenterSensorConfig(
                enabled=True,
                z_axis=[0.0, 1.0, 0.0],        # exactly like your example center sensor
                ref_axis=[1.0, 0.0,  0.0],
            ),
        ),

        SensorFamily(
            name="forearm_cap_2",                # horizontal ring
            link="forearm_link",
            kind="XY_Z",
            center=[-0.572, 0.0, -0.013],
            radius=0.045,
            N=7,
            offset=OFFSET,
            tilts=TiltConfig(tilts_deg=TILTS, include_all=True, include_alt=False),
            center_sensor=CenterSensorConfig(
                enabled=True,
                z_axis=[0.0, 0.0, -1.0],        # exactly like your example center sensor
                ref_axis=[1.0, 0.0,  0.0],
            ),
        ),

        SensorFamily(
            name="forearm_ring",  # horizontal ring
            link="forearm_link",
            kind="YZ_X",
            center=[-0.10, 0.0, 0.048],
            radius=0.05,
            N=8,
            offset=OFFSET,
            sweep_x=SWEEP,
            center_sensor=CenterSensorConfig(enabled=False),
        ),
        
        SensorFamily(
            name="forearm_cap_1", # vertical ring
            link="forearm_link",
            kind="XY_Z",
            center=[-0.01, 0.0, -0.02],
            radius=0.045,
            N=7,
            offset=OFFSET,
            tilts=TiltConfig(tilts_deg=TILTS, include_all=True, include_alt=False),
            center_sensor=CenterSensorConfig(
                enabled=True,
                z_axis=[0.0, 0.0, -1.0],     
                ref_axis=[1.0, 0.0,  0.0],
            ),
        ),

        SensorFamily(
            name="upperarm_cap_2",        
            link="upper_arm_link",
            kind="XY_Z",
            center=[-0.612, 0.0, 0.245],
            radius=0.055,
            N=7,
            offset=OFFSET,
            tilts=TiltConfig(tilts_deg=TILTS_NEG, include_all=True, include_alt=False),
            center_sensor=CenterSensorConfig(
                enabled=True,
                z_axis=[0.0, 0.0, 1.0],    
                ref_axis=[1.0, 0.0,  0.0],
            ),
        ),

        SensorFamily(
            name="upperarm_ring",
            link="upper_arm_link",
            kind="YZ_X",
            center=[-0.11, 0.0, 0.176],
            radius=0.055,
            N=8,
            offset=OFFSET,
            tilts=TiltConfig(tilts_deg=TILTS, include_all=True, include_alt=False),
            sweep_x=SWEEP,
            center_sensor=CenterSensorConfig(enabled=False),
        ),
        
        SensorFamily(
            name="upperarm_cap_1",
            link="upper_arm_link",
            kind="XY_Z",
            center=[0.0, 0.0, 0.27],
            radius=0.055,
            N=7,
            offset=OFFSET,
            tilts=TiltConfig(tilts_deg=TILTS_NEG, include_all=True, include_alt=False),
            center_sensor=CenterSensorConfig(
                enabled=True,
                z_axis=[0.0, 0.0, 1.0],     
                ref_axis=[1.0, 0.0,  0.0],
            ),
        ),

        SensorFamily(
            name="base_cap",
            link="base_link",
            kind="XY_Z",
            center=[0.0, 0.0, 0.216],
            radius=0.055,
            N=7,
            offset=OFFSET,
            tilts=TiltConfig(tilts_deg=TILTS_NEG, include_all=True, include_alt=False),
            center_sensor=CenterSensorConfig(
                enabled=True,
                z_axis=[0.0, 0.0, 1.0],     
                ref_axis=[1.0, 0.0,  0.0],
            ),
        ),
    ]

    return families


# -------------------- main --------------------

def main():
    families = build_config()

    variants = []
    for fam in families:
        variants.extend(build_variants_from_family(fam))

    all_sensors = []
    index_map = {}

    for spec in variants:
        start = len(all_sensors)

        if spec["kind"] == "YZ_X":
            sensors = make_ring_YZ_about_X(
                parent_link=spec["parent_link"],
                center_xyz=spec["center"],
                radius=spec["radius"],
                N=spec["N"],
                offset=spec["offset"],
                tilt_deg=spec["tilt_deg"],
                tilt_every_second=spec["tilt_every_second"],
                tilt_sign=spec["tilt_sign"],
                add_center_sensor=spec["center_sensor_enabled"],
                center_z_axis=spec["center_sensor_z_axis"],
                center_ref_axis=spec["center_sensor_ref_axis"],
            )
        elif spec["kind"] == "XY_Z":
            sensors = make_ring_XY_about_Z(
                parent_link=spec["parent_link"],
                center_xyz=spec["center"],
                radius=spec["radius"],
                N=spec["N"],
                offset=spec["offset"],
                tilt_deg=spec["tilt_deg"],
                tilt_every_second=spec["tilt_every_second"],
                tilt_sign=spec["tilt_sign"],
                add_center_sensor=spec["center_sensor_enabled"],
                center_z_axis=spec["center_sensor_z_axis"],
                center_ref_axis=spec["center_sensor_ref_axis"],
            )
        elif spec["kind"] == "XZ_Y":
            sensors = make_ring_XZ_about_Y(
                parent_link=spec["parent_link"],
                center_xyz=spec["center"],
                radius=spec["radius"],
                N=spec["N"],
                offset=spec["offset"],
                tilt_deg=spec["tilt_deg"],
                tilt_every_second=spec["tilt_every_second"],
                tilt_sign=spec["tilt_sign"],
                add_center_sensor=spec["center_sensor_enabled"],
                center_z_axis=spec["center_sensor_z_axis"],
                center_ref_axis=spec["center_sensor_ref_axis"],
            )
        else:
            raise ValueError(f"Unknown kind: {spec['kind']} (expected 'YZ_X', 'XY_Z', or 'XZ_Y')")

        all_sensors.extend(sensors)
        end = len(all_sensors)

        index_map[spec["name"]] = {
            "start": start,
            "count": end - start,
            "indices": list(range(start, end)),
            "params": dict(spec),
        }

    out_yaml = Path("ring_variants_candidates.yaml")
    out_json = Path("ring_variants_index_map.json")

    yaml.safe_dump(all_sensors, out_yaml.open("w"), sort_keys=False)
    out_json.write_text(json.dumps(index_map, indent=2))

    print(f"[OK] wrote {out_yaml} with {len(all_sensors)} sensors")
    print(f"[OK] wrote {out_json} with {len(index_map)} variants")


if __name__ == "__main__":
    main()
