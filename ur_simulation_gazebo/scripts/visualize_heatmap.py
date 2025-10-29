#!/usr/bin/env python3
"""
visualize_heatmap.py
--------------------
Visualize sensor coverage heatmap (from compute_visibility.py)
as a colored PointCloud2 in RViz.

Each voxel -> one point.
Color encodes number of sensors that can see that voxel (coverage).

"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
import numpy as np
import struct, yaml
from pathlib import Path
from collections import Counter

def create_cloud(points, intensities):
    """Build PointCloud2 message from Nx3 points and N intensity values."""
    msg = PointCloud2()
    msg.height = 1
    msg.width = len(points)
    msg.is_bigendian = False
    msg.is_dense = True

    msg.fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    msg.point_step = 16
    msg.row_step = msg.point_step * len(points)

    buf = b''.join(
        [struct.pack('ffff', *p, float(i)) for p, i in zip(points, intensities)]
    )
    msg.data = buf
    return msg


class HeatmapViz(Node):
    def __init__(self, yaml_path, topic="/heatmap_points"):
        super().__init__("heatmap_viz")

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1,
        )
        self.pub = self.create_publisher(PointCloud2, topic, qos)

        data = yaml.safe_load(Path(yaml_path).read_text())
        pts = np.array(data["voxels"], dtype=np.float32)
        cov = np.array(data["coverage"], dtype=np.float32)

        # mask 0 coverage turn on off 
        #mask = cov > 0
        #pts = pts[mask]
        #cov = cov[mask]

        # Normalize for color mapping
        cov_norm = cov / (cov.max() + 1e-9)
        colors = self.coverage_to_color(cov_norm)
        self.msg = self.make_colored_cloud(pts, colors)

        self.timer = self.create_timer(1.0, self.tick)
        self.once = False
        self.get_logger().info(f"Loaded {len(pts)} voxels from {yaml_path}")

        self.statistics(pts, cov)

    def statistics(self, pts, cov):
        """
        Print and return heatmap statistics:
          - min/max/mean coverage
          - voxel counts per coverage level
          - coordinates of least and most visible voxels
        """

        min_cov = np.min(cov)
        max_cov = np.max(cov)
        mean_cov = np.mean(cov)

        # Find indices of min and max coverage
        idx_min = np.where(cov == min_cov)[0]
        idx_max = np.where(cov == max_cov)[0]

        least_visible_pts = pts[idx_min]
        most_visible_pts = pts[idx_max]

        # Histogram / count of coverage levels
        counts = Counter(cov.astype(int))
        sorted_hist = dict(sorted(counts.items()))

        # Print summary
        print("──────────────────────────────")
        print("📊 HEATMAP STATISTICS")
        print(f"  ▸ Voxels total:       {len(cov):,}")
        print(f"  ▸ Coverage range:     {min_cov:.0f} – {max_cov:.0f}")
        print(f"  ▸ Average coverage:   {mean_cov:.2f}")
        print(f"  ▸ Unseen voxels:      {np.count_nonzero(cov==0)}")
        print("\n  ▸ Coverage histogram (count per coverage level):")
        for k, v in sorted_hist.items():
            print(f"     {k:3.0f}: {v:6d}")

        print("\n  ▸ Least visible voxel(s):")
        for p in least_visible_pts[:3]:
            print(f"     ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})")

        print("\n  ▸ Most visible voxel(s):")
        for p in most_visible_pts[:3]:
            print(f"     ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})")

        print("──────────────────────────────")

        # Optional: return dictionary if you want to use it programmatically
        return {
            "min": float(min_cov),
            "max": float(max_cov),
            "mean": float(mean_cov),
            "histogram": sorted_hist,
            "least_visible_pts": least_visible_pts.tolist(),
            "most_visible_pts": most_visible_pts.tolist(),
        }

    def coverage_to_color(self, norm_vals):
        """
        Map [0,1] -> RGB (blue→green→red)
        """
        r = np.clip(2 * norm_vals, 0, 1)
        g = np.clip(2 - 4 * np.abs(norm_vals - 0.5), 0, 1)
        b = np.clip(2 * (1 - norm_vals), 0, 1)
        return np.stack([r, g, b], axis=1)

    def make_colored_cloud(self, points, colors):
        """
        Convert Nx3 points + Nx3 colors to a PointCloud2 with rgb field
        """
        msg = PointCloud2()
        msg.header.frame_id = "world"
        msg.height = 1
        msg.width = len(points)
        msg.is_bigendian = False
        msg.is_dense = True

        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
        ]
        msg.point_step = 16
        msg.row_step = msg.point_step * len(points)

        def pack_rgb(r, g, b):
            return struct.unpack('I', struct.pack('BBBB',
                int(255*b), int(255*g), int(255*r), 0))[0]

        buf = b''.join(
            [struct.pack('fffI', *p, pack_rgb(*c)) for p, c in zip(points, colors)]
        )
        msg.data = buf
        return msg

    def tick(self):
        if self.once:
            return
        self.msg.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(self.msg)
        self.once = True
        self.get_logger().info(f"Published heatmap PointCloud2 with {self.msg.width} points.")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", default="heatmap.yaml", help="Path to heatmap.yaml")
    ap.add_argument("--topic", default="/heatmap_points", help="Output topic name")
    args = ap.parse_args()

    rclpy.init()
    node = HeatmapViz(args.yaml, args.topic)
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
