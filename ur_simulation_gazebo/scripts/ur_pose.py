#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import MotionPlanRequest, Constraints, JointConstraint, RobotState, MoveItErrorCodes
from trajectory_msgs.msg import JointTrajectory
from sensor_msgs.msg import JointState
import math


def deg2rad(vals):
    return [math.radians(v) for v in vals]


class MoveItPoseSelector(Node):
    """
    Node that allows interactive selection of one of 15 predefined robot poses
    (5 base configurations × 3 wrist configurations).
    """

    def __init__(self):
        super().__init__("moveit_pose_selector")

        self.joint_names = [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
        ]

        # 5 base configurations (deg)
        base_sets = [
            (0, -90, 0),
            (0, 0, 0),
            (0, -90, 90),
            (0, -125, 90),
            (0, -90, 160),
        ]

        # 3 wrist configurations (deg)
        wrist_sets = [
            (-90, 0, 0),        # x
            (0, 90, 0),         # z     # -90 90 0 for 13    
            (-90, 90, 0),       # y
        ]

        # Combine into 15 poses
        self.poses = []
        for b in base_sets:
            for w in wrist_sets:
                if b == (0, -90, 160) and w == (0, 90, 0):
                    w = (-180, -90, 0)
                pose = deg2rad([*b, *w])
                self.poses.append(pose)

        # Action client for MoveIt
        self.client = ActionClient(self, MoveGroup, "move_action")
        self.client.wait_for_server()

        # Trajectory publisher
        self.traj_pub = self.create_publisher(JointTrajectory, "/scaled_joint_trajectory_controller/joint_trajectory", 10)

        # Prompt user
        self.print_pose_table()
        self.select_pose()

    def print_pose_table(self):
        print("\n=== Available Robot Poses (index: [J1–J6] in deg) ===")
        for i, pose in enumerate(self.poses):
            print(f"{i:2d}: " + " ".join(f"{math.degrees(a):6.1f}" for a in pose))
        print("=====================================================\n")

    def select_pose(self):
        while True:
            try:
                idx = int(input(f"Select pose [0–{len(self.poses)-1}] (or -1 to exit): "))
                if idx == -1:
                    print("Exiting.")
                    rclpy.shutdown()
                    return
                if 0 <= idx < len(self.poses):
                    self.send_goal(idx)
                    return
                else:
                    print("Invalid index.")
            except ValueError:
                print("Please enter an integer.")

    def send_goal(self, idx):
        pose = self.poses[idx]
        self.get_logger().info(f"📡 Sending pose #{idx}: {[round(math.degrees(a), 1) for a in pose]}")

        goal_msg = MoveGroup.Goal()
        goal_msg.request = MotionPlanRequest()
        goal_msg.request.group_name = "ur_manipulator"
        goal_msg.request.max_velocity_scaling_factor = 0.1
        goal_msg.request.max_acceleration_scaling_factor = 0.1
        goal_msg.request.allowed_planning_time = 5.0
        goal_msg.request.num_planning_attempts = 10
        goal_msg.request.planner_id = "BiTRRTkConfigDefault"

        # Define start state (use current)
        start_state = RobotState()
        start_state.joint_state.name = self.joint_names
        start_state.joint_state.position = [0.0] * 6
        goal_msg.request.start_state = start_state

        # Define goal constraints
        constraints = [
            JointConstraint(
                joint_name=name,
                position=val,
                tolerance_above=0.01,
                tolerance_below=0.01,
                weight=1.0,
            )
            for name, val in zip(self.joint_names, pose)
        ]
        goal_msg.request.goal_constraints.append(Constraints(joint_constraints=constraints))

        goal_msg.planning_options.plan_only = False
        goal_msg.planning_options.replan = False
        goal_msg.planning_options.look_around = False

        send_future = self.client.send_goal_async(goal_msg)
        send_future.add_done_callback(self.goal_response_cb)

    def goal_response_cb(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn("❌ Goal rejected by MoveIt.")
            self.select_pose()
            return

        self.get_logger().info("✅ Goal accepted, waiting for result...")
        goal_handle.get_result_async().add_done_callback(self.result_cb)

    def result_cb(self, future):
        result = future.result().result
        traj = result.planned_trajectory.joint_trajectory
        if result.error_code.val != MoveItErrorCodes.SUCCESS or not traj.points:
            self.get_logger().warn(f"❌ Planning failed (error_code={result.error_code.val}).")
            self.select_pose()
            return

        self.get_logger().info(
            f"✅ Plan successful: {len(traj.points)} points, {result.planning_time:.2f}s planning time."
        )
        self.traj_pub.publish(traj)
        self.get_logger().info("▶️ Trajectory published to controller.")
        self.select_pose()


def main(args=None):
    rclpy.init(args=args)
    node = MoveItPoseSelector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
