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

def rot_about_axis(axis, angle):
    """Rodrigues rotation matrix for rotating around 'axis' by 'angle'."""
    axis = normalize(np.asarray(axis, dtype=float))
    x, y, z = axis
    ca, sa = math.cos(angle), math.sin(angle)
    C = 1.0 - ca
    return np.array([
        [ca + x*x*C,     x*y*C - z*sa, x*z*C + y*sa],
        [y*x*C + z*sa,   ca + y*y*C,   y*z*C - x*sa],
        [z*x*C - y*sa,   z*y*C + x*sa, ca + z*z*C],
    ], dtype=float)

# ---- Upper arm "geared" ring config ----
PARENT_LINK = "upper_arm_link"
N = 8
OFFSET = 0.005

# Center and radius in upper_arm_link frame
CENTER = np.array([-0.11, 0.0, 0.176], dtype=float)  # adjust axial placement (likely x)
RADIUS = 0.055                                      # adjust to cylinder radius

# Ring axis (arm axis). UR-style links typically: X is along the link.
RING_AXIS = "x"   # "x" or "z"

# Every 2nd sensor gets an extra tilt "up" (towards +arm_axis by default)
TILT_EVERY = 2
TILT_DEG = 45.0

# Tilt sign: +1 tilts towards +arm axis, -1 tilts towards -arm axis
TILT_SIGN = +1

out = []

for i in range(N):
    theta = i * (2.0 * math.pi / N)

    if RING_AXIS == "x":
        # ring in YZ plane around X axis
        xyz_i = CENTER + np.array([0.0,
                                   RADIUS * math.cos(theta),
                                   RADIUS * math.sin(theta)], dtype=float)
        radial = normalize(np.array([0.0,
                                     xyz_i[1] - CENTER[1],
                                     xyz_i[2] - CENTER[2]], dtype=float))
        arm_axis = np.array([1.0, 0.0, 0.0], dtype=float)    # arm axis
    elif RING_AXIS == "z":
        # ring in XY plane around Z axis
        xyz_i = CENTER + np.array([RADIUS * math.cos(theta),
                                   RADIUS * math.sin(theta),
                                   0.0], dtype=float)
        radial = normalize(np.array([xyz_i[0] - CENTER[0],
                                     xyz_i[1] - CENTER[1],
                                     0.0], dtype=float))
        arm_axis = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        raise ValueError("RING_AXIS must be 'x' or 'z'")

    # Base: sensor +Z points outward from cylinder
    z_axis = radial

    # Apply "gearing": rotate the view direction z_axis about the tangential direction
    # Tangent is along increasing theta; tangent = arm_axis x radial
    tangent = normalize(np.cross(arm_axis, radial))

    if TILT_EVERY > 0 and (i % TILT_EVERY == 0):
        tilt = math.radians(TILT_DEG) * float(TILT_SIGN)
        z_axis = normalize(rot_about_axis(tangent, tilt) @ z_axis)

    # Build orthonormal basis for sensor frame:
    # keep x_axis as close as possible to arm_axis for stable roll
    x_axis = arm_axis
    y_axis = normalize(np.cross(z_axis, x_axis))
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
        # Optional debug fields you can delete:
        # "theta_deg": float(math.degrees(theta)),
        # "tilted": bool(TILT_EVERY > 0 and (i % TILT_EVERY == 1)),
    })

with open("selected_candidates_upper_arm_ring_geared.yaml", "w") as f:
    yaml.safe_dump(out, f, sort_keys=False)

print("Wrote selected_candidates_upper_arm_ring_geared.yaml with", len(out), "sensors")
