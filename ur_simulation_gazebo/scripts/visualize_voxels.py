#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
import yaml
import numpy as np

from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2


class VoxelPointCloudPublisher(Node):
    def __init__(self):
        super().__init__('voxel_pc2_publisher')

        # Params
        self.declare_parameter('yaml_file', 'tmp/workspace_prism.yaml')
        self.declare_parameter('topic', '/workspace_voxels')
        self.declare_parameter('frame', 'base_link')
        self.declare_parameter('publish_rate', 1.0)
        self.declare_parameter('heatmap', False)

        yaml_path = self.get_parameter('yaml_file').get_parameter_value().string_value
        topic = self.get_parameter('topic').get_parameter_value().string_value
        self.frame = self.get_parameter('frame').get_parameter_value().string_value
        rate = float(self.get_parameter('publish_rate').get_parameter_value().double_value)
        use_heat = self.get_parameter('heatmap').get_parameter_value().bool_value

        # Load YAML
        self.get_logger().info(f"Loading voxels from: {yaml_path}")
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        vox = np.asarray(data['voxels'], dtype=np.float32)
        self.get_logger().info(f"Loaded {vox.shape[0]:,} voxels")

        # Heatmap/intensity values
        if 'values' in data:
            vals = np.asarray(data['values'], dtype=np.float32)
        elif use_heat:
            # Example: higher intensity near origin
            vals = np.exp(-np.linalg.norm(vox, axis=1)).astype(np.float32)
        else:
            vals = np.ones((vox.shape[0],), dtype=np.float32)

        # Prepare points as list of tuples (x,y,z,intensity)
        self.points = [ (float(x), float(y), float(z), float(i))
                        for (x,y,z), i in zip(vox, vals) ]

        # QoS: transient local so late subscribers (RViz) still get the last cloud
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
        self.pub = self.create_publisher(PointCloud2, topic, qos)

        # Publish timer
        self.timer = self.create_timer(max(1e-3, 1.0 / rate), self.publish_cloud)

        # Prebuild fields
        self.fields = [
            PointField(name='x',         offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y',         offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z',         offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        self.get_logger().info(f"Publishing PointCloud2 on {topic} (frame={self.frame})")

    def publish_cloud(self):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.frame

        msg = pc2.create_cloud(header, self.fields, self.points)
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = VoxelPointCloudPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
