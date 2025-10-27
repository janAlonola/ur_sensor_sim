#!/usr/bin/env python3
import yaml, math, rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray

class NormalsViz(Node):
    def __init__(self, yaml_path, topic="/sensor_normals", length=0.10):
        super().__init__("normals_viz")
        qos = QoSProfile(depth=1,
                         reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub = self.create_publisher(MarkerArray, topic, qos)
        self.length = float(length)

        with open(yaml_path, "r") as f:
            y = yaml.safe_load(f)
        self.cands = y["candidates"]

        self.timer = self.create_timer(0.5, self.tick)
        self.once = False

    def tick(self):
        if self.once: return
        arr = MarkerArray()
        mid = 0
        for c in self.cands:
            link = c["link"]
            x,y,z = c["xyz"]
            nx,ny,nz = c["normal"]
            # Arrow from two points: start=xyz, end=xyz+L*normal
            m = Marker()
            m.header.frame_id = link
            m.ns = "sensor_normal"
            m.id = mid; mid += 1
            m.type = Marker.ARROW
            m.action = Marker.ADD
            m.scale.x = 0.02   # shaft diameter
            m.scale.y = 0.04   # head diameter
            m.scale.z = 0.04   # head length
            m.color.r, m.color.g, m.color.b, m.color.a = (0.1, 0.7, 0.9, 1.0)
            m.points = [
                Point(x=x, y=y, z=z),
                Point(x=x+nx*self.length, y=y+ny*self.length, z=z+nz*self.length)
            ]
            arr.markers.append(m)
        self.pub.publish(arr)
        self.get_logger().info(f"Published {len(arr.markers)} normal arrows")
        self.once = True

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", default="mesh_sampling/candidates.yaml")
    ap.add_argument("--topic", default="/sensor_normals")
    ap.add_argument("--length", type=float, default=0.10)
    args = ap.parse_args()
    rclpy.init()
    node = NormalsViz(args.yaml, args.topic, args.length)
    rclpy.spin(node)
    rclpy.shutdown()
