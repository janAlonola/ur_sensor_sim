#!/usr/bin/env python3
import math
import yaml
import numpy as np

def rpy_from_R_zyx(R):
    pitch = math.atan2(-R[2,0], math.sqrt(R[0,0]**2 + R[1,0]**2))
    cp = math.cos(pitch)
    if abs(cp) < 1e-8:
        roll = 0.0
        yaw = math.atan2(-R[0,1], R[1,1])
    else:
        roll = math.atan2(R[2,1], R[2,2])
        yaw  = math.atan2(R[1,0], R[0,0])
    return float(roll), float(pitch), float(yaw)

def normalize(v):
    n = float(np.linalg.norm(v))
    return (v / n) if n > 1e-12 else v

# ---- Upper arm ring config ----
PARENT_LINK = "upper_arm_link"   # <- upper arm
N = 7
OFFSET = 0.005

# Ring center in upper_arm_link frame (you must set this)
# x = along link axis in your setup; y/z define the cylinder center
CENTER = np.array([-0.11, 0.0, 0.176], dtype=float)   # <-- adjust x (axial placement)
RADIUS = 0.055                                       # <-- adjust to match cylinder radius

# Ring plane: around the "arm axis".
# If your arm axis is X (typical for UR links), use axis="x".
# If it is Z, use axis="z".
RING_AXIS = "x"   # "x" or "z"

out = []
for i in range(N):
    theta = i * (2.0 * math.pi / N)

    if RING_AXIS == "x":
        # ring in YZ plane around X axis (arm axis = X)
        xyz_i = CENTER + np.array([0.0,
                                   RADIUS * math.cos(theta),
                                   RADIUS * math.sin(theta)], dtype=float)
        radial = normalize(np.array([0.0,
                                     xyz_i[1] - CENTER[1],
                                     xyz_i[2] - CENTER[2]], dtype=float))
        x_ref = np.array([1.0, 0.0, 0.0], dtype=float)   # stable reference = arm axis
    elif RING_AXIS == "z":
        # ring in XY plane around Z axis
        xyz_i = CENTER + np.array([RADIUS * math.cos(theta),
                                   RADIUS * math.sin(theta),
                                   0.0], dtype=float)
        radial = normalize(np.array([xyz_i[0] - CENTER[0],
                                     xyz_i[1] - CENTER[1],
                                     0.0], dtype=float))
        x_ref = np.array([0.0, 0.0, 1.0], dtype=float)   # stable reference = ring axis
    else:
        raise ValueError("RING_AXIS must be 'x' or 'z'")

    # Sensor convention in your pipeline: local +Z = viewing direction / normal
    z_axis = radial                 # point outward from cylinder
    y_axis = normalize(np.cross(z_axis, x_ref))
    x_axis = normalize(np.cross(y_axis, z_axis))
    R_i = np.column_stack([x_axis, y_axis, z_axis])

    rpy_i = rpy_from_R_zyx(R_i)
    normal_i = normalize(R_i @ np.array([0.0, 0.0, 1.0], dtype=float))

    out.append({
        "link": PARENT_LINK,
        "xyz": [float(xyz_i[0]), float(xyz_i[1]), float(xyz_i[2])],
        "rpy": [float(rpy_i[0]), float(rpy_i[1]), float(rpy_i[2])],
        "normal": [float(normal_i[0]), float(normal_i[1]), float(normal_i[2])],
        "offset": float(OFFSET),
    })

with open("selected_candidates_upper_arm_ring.yaml", "w") as f:
    yaml.safe_dump(out, f, sort_keys=False)

print("Wrote selected_candidates_upper_arm_ring.yaml with", len(out), "sensors")
