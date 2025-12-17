#!/usr/bin/env python3
"""
Generate a Xacro file that attaches N ToF sensors at poses from candidates.yaml.

Assumptions:
- candidates.yaml has a top-level key 'candidates', each with:
  { link: str, xyz: [x,y,z], rpy: [r,p,y] }  # as produced by your mesh_sampler
- The sensor looks along +Z in its local frame (sampler aligned RPY accordingly).

Usage examples:
  python3 gen_sensors_xacro.py --yaml candidates.yaml --out sensors_ur_with_tof.urdf.xacro
  python3 gen_sensors_xacro.py --yaml candidates.yaml --pick-n 10 --stride 7 > ur_with_tof.urdf.xacro
  python3 gen_sensors_xacro.py --yaml candidates.yaml --indices 12,37,88,101,233,377,400,512,777,900

Optional:
  --rate 15 --max-range 3.0 --name-prefix tof
"""

import argparse, sys, math, json
from pathlib import Path
import yaml
import numpy as np, math
from transforms3d.axangles import axangle2mat
from transforms3d.euler import mat2euler, euler2mat

def fmt3(v):
    return f"{v[0]:.9g} {v[1]:.9g} {v[2]:.9g}"

def select_indices(num, pick_n, stride, indices_csv):
    if indices_csv:
        idx = [int(x.strip()) for x in indices_csv.split(",") if x.strip()!=""]
    else:
        idx = list(range(0, num, max(1, stride)))
    return idx[:pick_n]
    
def rpy_from_normal(normal, beam_axis='z'):
    """
    Make RPY so that the sensor's forward axis points along `normal`.
    - If beam_axis='z'  → +Z points along normal
    - If beam_axis='x'  → +X points along normal (Gazebo ray default)
    """
    n = np.asarray(normal, dtype=float)
    n /= (np.linalg.norm(n) + 1e-12)
    z = np.array([0.0, 0.0, 1.0])

    c = float(np.dot(z, n))
    if c > 0.999999:
        R = np.eye(3)
    elif c < -0.999999:
        R = axangle2mat([1, 0, 0], math.pi)   # 180° about X maps +Z→-Z
    else:
        axis = np.cross(z, n)
        axis /= (np.linalg.norm(axis) + 1e-12)
        ang = math.acos(max(-1.0, min(1.0, c)))
        R = axangle2mat(axis, ang)

    # If your sensor beam is along +X (Gazebo ray), rotate -90° about Y so +Z→+X
    if beam_axis.lower() == 'x':
        R = R @ euler2mat(0.0, -math.pi/2.0, 0.0, axes='sxyz')

    r, p, y = mat2euler(R, axes='sxyz')
    return [float(r), float(p), float(y)]

TEMPLATE_HEAD = """<?xml version="1.0"?>
<robot name="ur" xmlns:xacro="http://www.ros.org/wiki/xacro">

  <!-- include UR base macros -->
  <xacro:include filename="$(find ur_description)/urdf/ur_macro.xacro"/>

  <!-- include sensor macro -->
  <xacro:include filename="$(find ur_tof_description)/urdf/sensors/tof_sensor.xacro"/>

  <!-- passthrough args -->
  <xacro:arg name="ur_type" default="ur10"/>
  <xacro:arg name="safety_limits" default="true"/>
  <xacro:arg name="prefix" default=""/>

  <!-- define dummy world link -->
  <link name="world"/>

  <!-- Instantiate the UR robot -->
  <xacro:ur_robot
      name="ur"
      tf_prefix="$(arg prefix)"
      parent="world"
      joint_limits_parameters_file="$(find ur_description)/config/$(arg ur_type)/joint_limits.yaml"
      kinematics_parameters_file="$(find ur_description)/config/$(arg ur_type)/default_kinematics.yaml"
      physical_parameters_file="$(find ur_description)/config/$(arg ur_type)/physical_parameters.yaml"
      visual_parameters_file="$(find ur_description)/config/$(arg ur_type)/visual_parameters.yaml"
      safety_limits="$(arg safety_limits)"
      safety_pos_margin="0.15"
      safety_k_position="20"
      sim_gazebo="true">
    <origin xyz="0 0 0.25" rpy="0 0 0"/>
  </xacro:ur_robot>

  <!-- Attach ToF sensors (auto-generated) -->
"""

TEMPLATE_TAIL = """
  <!-- Gazebo ROS 2 Control -->
  <gazebo>
    <plugin name="gazebo_ros2_control" filename="libgazebo_ros2_control.so">
      <parameters>$(find ur_simulation_gazebo)/config/ur_controllers.yaml</parameters>
    </plugin>
  </gazebo>

</robot>
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", default="ur_sensor_sim/mesh_sampling/big_candidates_with_ring.yaml", help="Path to candidates.yaml")
    ap.add_argument("--out", default="ur_sensor_sim/ur_tof_description/urdf/ur_with_tof.urdf.xacro", help="Write to file (default: stdout)")
    ap.add_argument("--pick-n", type=int, default=10, help="How many sensors to place")
    ap.add_argument("--stride", type=int, default=40, help="Take every stride-th candidate (ignored if --indices given)")
    ap.add_argument("--indices", default="", help="Explicit comma-separated candidate indices (overrides stride)")
    ap.add_argument("--rate", type=float, default=15.0)
    ap.add_argument("--max-range", type=float, default=3.0)
    ap.add_argument("--name-prefix", default="tof")
    args = ap.parse_args()

    data = yaml.safe_load(Path(args.yaml).read_text())
    cands = data.get("candidates", [])
    if not cands:
        print("ERROR: no candidates found in YAML", file=sys.stderr)
        sys.exit(2)


    sel = [266, 275, 640, 647, 668, 876, 1440, 1456, 1471, 1921, 1933, 2022, 2629, 2658, 2659, 2660, 2661, 2662, 2663, 2664]
    
  #[28, 58, 62, 65, 225, 247, 304, 324, 488, 501, 519, 601, 625, 645, 683, 684, 700, 702, 799, 812] #[11, 114, 317, 626, 1062, 286, 104, 824, 296, 241, 649, 1128, 860, 367, 377, 1124, 826, 858, 560, 898]#[6, 118, 338, 375, 491, 560, 566, 581, 942, 1118] # [452, 28, 117, 0, 566, 338, 104, 581, 371, 1128, 942, 856, 855, 382, 824, 843, 367, 285, 314, 860 ]
    #= select_indices(len(cands), args.pick_n, args.stride, args.indices)

    # Build Xacro
    parts = [TEMPLATE_HEAD]
    for i, ci in enumerate(sel):
        c = cands[ci]
        link = c["link"]
        xyz = c.get("xyz")
        rpy = rpy_from_normal(c.get("normal"), beam_axis='x')
        if xyz is None or rpy is None:
            print(f"WARNING: candidate {ci} missing xyz/rpy; skipping", file=sys.stderr)
            continue
        name = f"{args.name_prefix}_{ci}"
        parts.append(
            f'  <xacro:tof_sensor name="{name}" parent="{link}" '
            f'xyz="{fmt3(xyz)}" rpy="{fmt3(rpy)}" '
            f'max_range="{args.max_range:.9g}" rate="{args.rate:.9g}"/>\n'
        )

    parts.append(TEMPLATE_TAIL)
    xacro_text = "".join(parts)

    if args.out:
        Path(args.out).write_text(xacro_text)
        print(f"[OK] wrote {args.out} with {len(sel)} sensors from {args.yaml}")
    else:
        print(xacro_text)

if __name__ == "__main__":
    main()
