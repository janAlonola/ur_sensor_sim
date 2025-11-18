#!/usr/bin/env python3
import math
import rclpy
import yaml
import numpy as np
from rclpy.node import Node
from pathlib import Path

# Interactive markers / RViz
from visualization_msgs.msg import InteractiveMarker, InteractiveMarkerControl, Marker
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from interactive_markers.menu_handler import MenuHandler
from geometry_msgs.msg import Point
from std_msgs.msg import Int32MultiArray

# TF2
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException
from geometry_msgs.msg import TransformStamped
from rclpy.duration import Duration

AXES = {
    "+X": np.array([1.0, 0.0, 0.0]), "-X": np.array([-1.0, 0.0, 0.0]),
    "+Y": np.array([0.0, 1.0, 0.0]), "-Y": np.array([0.0, -1.0, 0.0]),
    "+Z": np.array([0.0, 0.0, 1.0]), "-Z": np.array([0.0, 0.0, -1.0]),
}

def q_to_R(qx, qy, qz, qw) -> np.ndarray:
    """Quaternion (x,y,z,w) -> 3x3 rotation matrix."""
    x, y, z, w = qx, qy, qz, qw
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return np.array([
        [1 - 2*(yy + zz),     2*(xy - wz),         2*(xz + wy)],
        [    2*(xy + wz),  1 - 2*(xx + zz),        2*(yz - wx)],
        [    2*(xz - wy),     2*(yz + wx),     1 - 2*(xx + yy)]
    ], dtype=float)

def load_candidates(path: str):
    d = yaml.safe_load(Path(path).read_text())
    cands = d["candidates"]
    xyz = np.array([c["xyz"] for c in cands], dtype=float)                    # (N,3) in link frame
    normals = np.array([c.get("normal", [0,0,1]) for c in cands], dtype=float)
    nrm = np.linalg.norm(normals, axis=1, keepdims=True); nrm[nrm == 0.0] = 1.0
    normals = normals / nrm
    links = [c.get("link", "") for c in cands]
    offs_default = float(d.get("offset_m", 0.0))
    offsets = np.array([c.get("offset", offs_default) for c in cands], dtype=float)
    return xyz, normals, links, offsets, d

def normal_in_cone(n, axis_vec, deg):
    cos_th = np.clip(np.dot(n, axis_vec) / (np.linalg.norm(n)*np.linalg.norm(axis_vec)), -1.0, 1.0)
    th = math.degrees(math.acos(cos_th))
    return th <= deg

class SensorPicker(Node):
    def __init__(self, yaml_path: str,
                 frame: str = "world",
                 sphere_d: float = 0.03,
                 arrow_len: float = 0.06,
                 exclude_links=None,
                 normal_exclude=None,
                 save_path: str | None = None):
        super().__init__("sensor_picker")

        # Load candidate data (link-frame)
        self.frame = frame
        self.xyz_link, self.normals_link, self.links, self.offsets, self.meta = load_candidates(yaml_path)
        self.N = self.xyz_link.shape[0]
        self.enabled = np.ones(self.N, dtype=bool)
        self.save_path = save_path

        # TF2: create buffer & listener, then let TF messages arrive
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)  # background thread
        for _ in range(20):  # ~2s total
            rclpy.spin_once(self, timeout_sec=0.1)
            frames_yaml = self.tf_buffer.all_frames_as_yaml()
            if frames_yaml and frames_yaml != "[]":
                break

        # Optional auto-excludes
        if exclude_links:
            bad = {i for i, L in enumerate(self.links) if L in exclude_links}
            if bad:
                self.get_logger().info(f"Auto-excluding {len(bad)} by link: {sorted(exclude_links)}")
                self.enabled[list(bad)] = False
        if normal_exclude:
            axis_key, deg = normal_exclude
            axis_vec = AXES[axis_key]
            bad_idx = [i for i, n in enumerate(self.normals_link) if normal_in_cone(n, axis_vec, deg)]
            if bad_idx:
                self.get_logger().info(f"Auto-excluding {len(bad_idx)} by normal cone {axis_key} ≤ {deg}°")
                self.enabled[bad_idx] = False

        # Transform candidates to target frame
        self.xyz_world, self.normals_world = self.transform_candidates_to_frame()

        # Interactive Marker server (absolute name so Update Topic is fixed)
        self.server = InteractiveMarkerServer(self, "/sensor_picker_server")
        self.menu = MenuHandler()
        self.menu_enable = self.menu.insert("Enable", callback=self._menu_enable)
        self.menu_disable = self.menu.insert("Disable", callback=self._menu_disable)

        # Publisher of enabled indices
        self.pub = self.create_publisher(Int32MultiArray, "sensors_enabled_indices", 10)

        # Visual sizing
        self.sphere_d = float(sphere_d)
        self.arrow_len = float(arrow_len)

        # Build UI
        self._build_markers()
        self._publish_enabled()

    # ---- TF transform (link -> target frame) ----
    def transform_candidates_to_frame(self):
        target = self.frame
        unique_links = sorted(set(self.links))
        self.get_logger().info(f"Transforming {self.N} candidates into frame '{target}' from links: {unique_links}")

        # Log frames once (helpful for debugging)
        try:
            frames_yaml = self.tf_buffer.all_frames_as_yaml()
            self.get_logger().info(f"TF frames known to this node:\n{frames_yaml}")
        except Exception:
            pass

        link_T = {}
        timeout = Duration(seconds=1.0)
        for link in unique_links:
            if not link:
                link_T[link] = (np.eye(3), np.zeros(3))
                continue
            try:
                if not self.tf_buffer.can_transform(target, link, rclpy.time.Time(), timeout):
                    raise LookupException("timeout waiting for TF")
                tf: TransformStamped = self.tf_buffer.lookup_transform(target, link, rclpy.time.Time(), timeout)
                R = q_to_R(tf.transform.rotation.x,
                           tf.transform.rotation.y,
                           tf.transform.rotation.z,
                           tf.transform.rotation.w)
                t = np.array([tf.transform.translation.x,
                              tf.transform.translation.y,
                              tf.transform.translation.z], dtype=float)
                link_T[link] = (R, t)
            except (LookupException, ConnectivityException, ExtrapolationException) as e:
                self.get_logger().warn(f"[TF] No transform {link} -> {target} ({e}). Using identity; markers may be misplaced.")
                link_T[link] = (np.eye(3), np.zeros(3))

        xyz_w = np.zeros_like(self.xyz_link)
        nrm_w = np.zeros_like(self.normals_link)
        for i, link in enumerate(self.links):
            R, t = link_T[link]
            n_link = self.normals_link[i]
            p_link = self.xyz_link[i] + n_link * self.offsets[i]  # apply offset along link normal
            p_w = R @ p_link + t
            n_w = R @ n_link
            n_w = n_w / (np.linalg.norm(n_w) + 1e-12)
            xyz_w[i] = p_w
            nrm_w[i] = n_w

        self.get_logger().info("TF transform complete.")
        return xyz_w, nrm_w

    # ---- Marker builders (use world-frame data) ----
    def _mk_sphere_marker(self, i):
        m = Marker()
        m.type = Marker.SPHERE
        m.scale.x = m.scale.y = m.scale.z = self.sphere_d
        m.color.a = 1.0
        if self.enabled[i]:
            m.color.r, m.color.g, m.color.b = 0.10, 0.80, 0.15
        else:
            m.color.r, m.color.g, m.color.b = 0.75, 0.20, 0.20
        m.pose.position.x, m.pose.position.y, m.pose.position.z = self.xyz_world[i].tolist()
        return m

    def _mk_arrow_marker(self, i):
        start = self.xyz_world[i]
        end = start + self.normals_world[i] * self.arrow_len
        m = Marker()
        m.type = Marker.ARROW
        m.points = [
            Point(x=float(start[0]), y=float(start[1]), z=float(start[2])),
            Point(x=float(end[0]),   y=float(end[1]),   z=float(end[2]))
        ]
        m.scale.x = self.sphere_d * 0.25  # shaft diameter
        m.scale.y = self.sphere_d * 0.45  # head diameter
        m.scale.z = self.sphere_d * 0.45  # head length
        m.color.a = 0.9
        m.color.r, m.color.g, m.color.b = (0.2, 0.4, 0.9) if self.enabled[i] else (0.5, 0.5, 0.5)
        return m

    def _add_marker(self, i):
        name = f"sensor_{i}"
        im = InteractiveMarker()
        im.header.frame_id = self.frame
        im.name = name
        im.description = f"{i} | {self.links[i]}"
        im.scale = max(self.sphere_d * 6.0, 0.1)
        # leave IM pose at origin in target frame; geometry carries positions
        im.pose.position.x = im.pose.position.y = im.pose.position.z = 0.0

        ctrl = InteractiveMarkerControl()
        ctrl.interaction_mode = InteractiveMarkerControl.BUTTON
        ctrl.always_visible = True
        ctrl.markers.append(self._mk_sphere_marker(i))
        ctrl.markers.append(self._mk_arrow_marker(i))
        im.controls.append(ctrl)

        self.server.insert(im)
        self.server.setCallback(name, self._cb_click)
        self.menu.apply(self.server, name)

    def _build_markers(self):
        for i in range(self.N):
            self._add_marker(i)
        self.server.applyChanges()

    # ---- Interactions ----
    def _cb_click(self, fb):
        i = int(fb.marker_name.split("_")[-1])
        self.enabled[i] = ~self.enabled[i]
        self._refresh_marker(i)
        self._publish_enabled()

    def _menu_enable(self, fb):  self._set_state(fb, True)
    def _menu_disable(self, fb): self._set_state(fb, False)

    def _set_state(self, fb, state):
        i = int(fb.marker_name.split("_")[-1])
        self.enabled[i] = state
        self._refresh_marker(i)
        self._publish_enabled()

    def _refresh_marker(self, i):
        name = f"sensor_{i}"
        self.server.erase(name)
        self._add_marker(i)
        self.server.applyChanges()

    def _publish_enabled(self):
        msg = Int32MultiArray()
        enabled_idx = np.where(self.enabled)[0].astype(int).tolist()
        msg.data = enabled_idx
        self.pub.publish(msg)
        self.get_logger().info(f"Enabled: {len(enabled_idx)}/{self.N}")

        if self.save_path:
            data = {
                "include": list(enabled_idx),
                "exclude": np.where(~self.enabled)[0].astype(int).tolist(),
            }
            save_p = Path(self.save_path)
            save_p.parent.mkdir(parents=True, exist_ok=True)
            save_p.write_text(yaml.safe_dump(data, sort_keys=False))

def main():
    import argparse, time
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates-yaml", default="ur_sensor_sim/mesh_sampling/candidates.yaml")
    ap.add_argument("--frame", default="base_link_inertia")          # target frame for RViz & transforms
    ap.add_argument("--sphere", type=float, default=0.03)
    ap.add_argument("--arrow",  type=float, default=0.06)
    ap.add_argument("--exclude-links", type=str, default=None)
    ap.add_argument("--normal-exclude", type=str, default=None)  # e.g. "+Z,30"
    ap.add_argument("--save", type=str, default="ur_sensor_sim/mesh_sampling/sel_candidates.yaml")
    args = ap.parse_args()

    excl_links = set(map(str.strip, args.exclude_links.split(","))) if args.exclude_links else None
    norm_exc = None
    if args.normal_exclude:
        axis_key, deg = args.normal_exclude.split(",")
        axis_key = axis_key.strip().upper()
        if axis_key not in AXES:
            raise SystemExit("normal-exclude axis must be one of: " + ", ".join(AXES.keys()))
        norm_exc = (axis_key, float(deg))

    rclpy.init()
    node = SensorPicker(args.candidates_yaml, frame=args.frame,
                        sphere_d=args.sphere, arrow_len=args.arrow,
                        exclude_links=excl_links, normal_exclude=norm_exc,
                        save_path=args.save)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.server.clear()
            node.server.applyChanges()
        except Exception:
            pass
        node.get_logger().info("Shutting down cleanly…")
        node.destroy_node()
        rclpy.shutdown()
        time.sleep(0.1)

if __name__ == "__main__":
    main()
