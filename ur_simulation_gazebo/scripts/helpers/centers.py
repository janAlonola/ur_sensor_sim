#!/usr/bin/env python3
import time, yaml
from pathlib import Path
from typing import List

import rclpy
from rclpy.node import Node
from rclpy.clock import ClockType
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException

def lookup_xyz(buf: Buffer, target: str, source: str, node: Node, retries=60, sleep=0.05):
    for _ in range(retries):
        try:
            tf = buf.lookup_transform(target, source, rclpy.time.Time())
            t = tf.transform.translation
            return [float(t.x), float(t.y), float(t.z)]
        except (LookupException, ConnectivityException, ExtrapolationException):
            rclpy.spin_once(node, timeout_sec=sleep)
            time.sleep(sleep)
    return None

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Publish joint poses -> record centers from TF")
    ap.add_argument("--poses-yaml", default="ur_sensor_sim/tmp/poses.yaml", help="poses.yaml with joint_names & poses[].joints_rad")
    ap.add_argument("--out", default="ur_sensor_sim/tmp/centers.yaml", help="Output centers.yaml")
    ap.add_argument("--tf-target", default="world", help="Target frame for centers")
    ap.add_argument("--forearm-frame", default="forearm_link", help="Forearm/Elbow frame name")
    ap.add_argument("--tcp-frame", default="tool0", help="TCP frame name")
    ap.add_argument("--base-fixed", default="0,0,0.25", help="Fixed base center 'x,y,z' in target frame")
    ap.add_argument("--rate", type=float, default=50.0, help="JointState publish rate (Hz)")
    ap.add_argument("--hold-sec", type=float, default=0.5, help="Hold duration per pose (s)")
    ap.add_argument("--wait-tf-sec", type=float, default=0.5, help="Extra wait for TF to settle (s)")
    args = ap.parse_args()

    poses_doc = yaml.safe_load(Path(args.poses_yaml).read_text())
    if not isinstance(poses_doc, dict) or "joint_names" not in poses_doc or "poses" not in poses_doc:
        raise SystemExit("poses.yaml must contain {joint_names: [...], poses: [{joints_rad:[...]}...]}")

    joint_names: List[str] = poses_doc["joint_names"]
    poses = poses_doc["poses"]

    # Fixed base parse
    bx, by, bz = [float(v) for v in args.base_fixed.split(",")]

    rclpy.init()
    node = rclpy.create_node("record_centers_from_joints")
    # make sure node time runs even without sim time
    node.set_parameters([node.get_parameter_or("use_sim_time", False)])

    js_pub = node.create_publisher(JointState, "joint_states", 10)
    buf = Buffer()
    _listener = TransformListener(buf, node)

    # Warm up TF
    for _ in range(40):
        rclpy.spin_once(node, timeout_sec=0.05)
        time.sleep(0.01)

    dt = 1.0 / max(1e-6, args.rate)
    centers = []
    labels = []

    node.get_logger().info(f"Publishing {len(poses)} poses to /joint_states at {args.rate:.1f} Hz…")
    for idx, p in enumerate(poses, 1):
        name = p.get("name", f"pose_{idx}") if isinstance(p, dict) else f"pose_{idx}"
        input(f"[{idx}/{len(poses)}] Stelle Roboter auf '{name}' und drücke ENTER … ")
        q = p.get("joints_rad", None)
        if q is None or len(q) != len(joint_names):
            raise SystemExit(f"Pose #{idx} missing joints_rad or wrong length (got {q})")

        # Publish joints for hold-sec duration
        t_end = time.time() + args.hold-sec if False else time.time() + args.hold_sec  # avoid hyphen typo
        while time.time() < t_end:
            msg = JointState()
            msg.name = joint_names
            msg.position = list(map(float, q))
            msg.velocity = []
            msg.effort = []
            msg.header.stamp = node.get_clock().now().to_msg()
            js_pub.publish(msg)
            rclpy.spin_once(node, timeout_sec=0.0)
            time.sleep(dt)

        # Small extra wait to let robot_state_publisher push TF
        time.sleep(args.wait_tf_sec)
        rclpy.spin_once(node, timeout_sec=0.05)

        # Record 3 centers
        # 1) fixed base
        centers.append([bx, by, bz])
        labels.append({"pose": p.get("name", f"pose_{idx}"), "type": "base_fixed"})

        # 2) forearm
        forearm = lookup_xyz(buf, args.tf_target, args.forearm_frame, node)
        if forearm:
            centers.append(forearm)
            labels.append({"pose": p.get("name", f"pose_{idx}"), "type": "forearm", "frame": args.forearm_frame})
            node.get_logger().info(f"[{idx:02d}] forearm: {forearm}")
        else:
            node.get_logger().warn(f"[{idx:02d}] No TF {args.forearm_frame}->{args.tf_target}")

        # 3) tcp
        tcp = lookup_xyz(buf, args.tf_target, args.tcp_frame, node)
        if tcp:
            centers.append(tcp)
            labels.append({"pose": p.get("name", f"pose_{idx}"), "type": "tcp", "frame": args.tcp_frame})
            node.get_logger().info(f"[{idx:02d}] tcp    : {tcp}")
        else:
            node.get_logger().warn(f"[{idx:02d}] No TF {args.tcp_frame}->{args.tf_target}")

    # Done
    try:
        node.destroy_node()
    finally:
        rclpy.shutdown()

    out = {"centers": centers, "labels": labels, "target_frame": args.tf_target}
    Path(args.out).write_text(yaml.safe_dump(out, sort_keys=False))
    print(f"\n[OK] wrote {args.out}  centers={len(centers)}  (~3 per pose)")

if __name__ == "__main__": main()