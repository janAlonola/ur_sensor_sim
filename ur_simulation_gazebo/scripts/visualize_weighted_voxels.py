#!/usr/bin/env python3
import struct
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

        # ---------------- params ----------------
        self.declare_parameter('yaml_file', 'ur_sensor_sim/tmp/weighted_poses/b1_w1.yaml')
        self.declare_parameter('topic', '/workspace_voxels')
        self.declare_parameter('frame', 'world')
        self.declare_parameter('publish_rate', 1.0)

        # Show weights in intensity (default). Optional: also publish RGB from weights.
        self.declare_parameter('colorize', False)          # add packed 'rgb' field
        self.declare_parameter('normalize', True)          # normalize weights to [0,1] if not already
        self.declare_parameter('weight_key', 'weights')    # YAML key to read weights from (default produced by your script)
        self.declare_parameter('fallback_key', 'values')   # secondary key if weights missing
        self.declare_parameter('heatmap', False)           # last-resort synthetic heatmap if neither key exists

        yaml_path = self.get_parameter('yaml_file').get_parameter_value().string_value
        topic = self.get_parameter('topic').get_parameter_value().string_value
        self.frame = self.get_parameter('frame').get_parameter_value().string_value
        rate = float(self.get_parameter('publish_rate').get_parameter_value().double_value)

        colorize = self.get_parameter('colorize').get_parameter_value().bool_value
        normalize = self.get_parameter('normalize').get_parameter_value().bool_value
        weight_key = self.get_parameter('weight_key').get_parameter_value().string_value
        fallback_key = self.get_parameter('fallback_key').get_parameter_value().string_value
        use_heat = self.get_parameter('heatmap').get_parameter_value().bool_value

        # ---------------- load YAML ----------------
        self.get_logger().info(f"Loading voxels from: {yaml_path}")
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        vox = np.asarray(data['voxels'], dtype=np.float32)
        N = vox.shape[0]
        self.get_logger().info(f"Loaded {N:,} voxels")

        # ---------------- choose weights ----------------
        weights = None
        if weight_key in data:
            weights = np.asarray(data[weight_key], dtype=np.float32)
            if weights.shape[0] != N:
                self.get_logger().warn(f"'{weight_key}' length {weights.shape[0]} != voxels {N}; falling back…")
                weights = None

        if weights is None and fallback_key in data:
            self.get_logger().info(f"Using '{fallback_key}' from YAML as intensity/weights")
            weights = np.asarray(data[fallback_key], dtype=np.float32)
            weights = weights[:N]  # in case it's longer

        if weights is None and use_heat:
            self.get_logger().info("No weights found; generating synthetic heatmap (exp(-||p||))")
            weights = np.exp(-np.linalg.norm(vox, axis=1)).astype(np.float32)

        if weights is None:
            self.get_logger().warn("No weights/values found; using ones.")
            weights = np.ones((N,), dtype=np.float32)

        # Optional normalization to [0,1]
        if normalize:
            wmin, wmax = float(np.min(weights)), float(np.max(weights))
            if wmax > wmin:
                weights = (weights - wmin) / (wmax - wmin)
            else:
                weights = np.zeros_like(weights)

        # ---------------- build fields & points ----------------
        # Always publish intensity = weights
        fields = [
            PointField(name='x',         offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y',         offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z',         offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        points = None

        if colorize:
            # Pack grayscale RGB from weights into a single float32 'rgb' field (RViz expects FLOAT32 packed RGB)
            # grayscale = round(255 * w), pack as 0xRRGGBB
            fields.append(PointField(name='rgb', offset=16, datatype=PointField.FLOAT32, count=1))

            def pack_rgb_float(gray: np.ndarray) -> np.ndarray:
                g = np.clip((gray * 255.0).round().astype(np.uint8), 0, 255)
                rgb_uint32 = (g.astype(np.uint32) << 16) | (g.astype(np.uint32) << 8) | g.astype(np.uint32)
                # reinterpret as float32
                return rgb_uint32.view(np.float32)

            rgb_f32 = pack_rgb_float(weights)

            # Interleave x,y,z,intensity,rgb
            pts = np.empty((N, 5), dtype=np.float32)
            pts[:, 0:3] = vox
            pts[:, 3] = weights
            pts[:, 4] = rgb_f32
            points = [tuple(row) for row in pts]
        else:
            # Interleave x,y,z,intensity
            pts = np.empty((N, 4), dtype=np.float32)
            pts[:, 0:3] = vox
            pts[:, 3] = weights
            points = [tuple(row) for row in pts]

        # QoS: transient local so RViz late subscribers still get last cloud
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
        self.pub = self.create_publisher(PointCloud2, topic, qos)

        # Publish timer
        self.timer = self.create_timer(max(1e-3, 1.0 / rate), self.publish_cloud)

        self.fields = fields
        self.points = points
        self.get_logger().info(
            f"Publishing PointCloud2 on {topic} (frame={self.frame}) "
            f"with intensity=weights{' and rgb=grayscale(weights)' if colorize else ''}"
        )

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
