#!/usr/bin/env python3
import math
import yaml
import numpy as np

def Rx(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[1,0,0],[0,ca,-sa],[0,sa,ca]], dtype=float)

def Rz(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[ca,-sa,0],[sa,ca,0],[0,0,1]], dtype=float)

def Ry(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[ca,0,sa],[0,1,0],[-sa,0,ca]], dtype=float)

def R_from_rpy(roll, pitch, yaw):
    # ROS convention: R = Rz(yaw) * Ry(pitch) * Rx(roll)
    return Rz(yaw) @ Ry(pitch) @ Rx(roll)

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

# ---- Seed pose ----
PARENT_LINK = "forearm_link"        # <- set to your actual forearm link name (e.g. "forearm_link")

N = 7
OFFSET = 0.005   # use whatever you use elsewhere

out = []
CENTER = np.array([-0.01, 0.0, -0.02], dtype=float)   # [xc, yc, zc]  <-- yc/zc are what you need
RADIUS = 0.055

for i in range(N):
    theta = i * (2.0 * math.pi / N)

    # ring in XY plane around Z axis
    xyz_i = CENTER + np.array([RADIUS * math.cos(theta), RADIUS * math.sin(theta), 0.0], dtype=float)

    # radial in XY (outward from CENTER)
    radial = normalize(np.array([ xyz_i[0] - CENTER[0], xyz_i[1] - CENTER[1], 0.0], dtype=float))

    z_axis = radial                    # sensor +Z points outward (radial)
    x_axis = np.array([0.0, 0.0, 1.0]) # choose +Z as a stable reference axis
    y_axis = normalize(np.cross(z_axis, x_axis))
    x_axis = normalize(np.cross(y_axis, z_axis))

    R_i = np.column_stack([x_axis, y_axis, z_axis])
    rpy_i = rpy_from_R_zyx(R_i)
    normal_i = normalize(R_i @ np.array([0.0, 0.0, 1.0]))

    out.append({
        "link": PARENT_LINK,
        "xyz": [float(xyz_i[0]), float(xyz_i[1]), float(xyz_i[2])],
        "rpy": [float(rpy_i[0]), float(rpy_i[1]), float(rpy_i[2])],
        "normal": [float(normal_i[0]), float(normal_i[1]), float(normal_i[2])],
        "offset": float(OFFSET),
    })

xyz_c = CENTER.copy()

z_axis = np.array([0.0, 0.0, -1.0], dtype=float)      # sensor +Z points "outward" = -Z
x_axis = np.array([1.0, 0.0,  0.0], dtype=float)      # any axis not parallel to z_axis
y_axis = normalize(np.cross(z_axis, x_axis))
x_axis = normalize(np.cross(y_axis, z_axis))

R_c = np.column_stack([x_axis, y_axis, z_axis])
rpy_c = rpy_from_R_zyx(R_c)
normal_c = z_axis  # same as R_c @ [0,0,1]

out.append({
    "link": PARENT_LINK,
    "xyz": [float(xyz_c[0]), float(xyz_c[1]), float(xyz_c[2])],
    "rpy": [float(rpy_c[0]), float(rpy_c[1]), float(rpy_c[2])],
    "normal": [float(normal_c[0]), float(normal_c[1]), float(normal_c[2])],
    "offset": float(OFFSET),
})

with open("selected_candidates_forearm_ring.yaml", "w") as f:
    yaml.safe_dump(out, f, sort_keys=False)

print("Wrote selected_candidates_forearm_ring.yaml with", len(out), "sensors")
