"""ROS2 bridge for the RallyCourt mecanum robot (MotrixSim).

Implements the ROS2 interface of rally_ctrl/MECANUM_SENSORS.md on top of the
MotrixSim MJCF, in one process: physics, drive, sensors, publishers.

Topics
------
subscribe  /mecanum/cmd_vel    geometry_msgs/Twist        vehicle frame
subscribe  /mecanum/wheel_cmd  std_msgs/Float64MultiArray 4 wheel targets [rad/s]
                                                          FL, FR, RL, RR (bypasses cmd_vel)
publish    /mecanum/state      std_msgs/Float64MultiArray [x, y, theta, vx, vy]
publish    /imu/data_raw       sensor_msgs/Imu           6-axis, frame imu_link
publish    /odom              nav_msgs/Odometry          odom -> base_link
publish    /tf                tf2_msgs/TFMessage         odom->base_link, base_link->imu_link
publish    /joint_states      sensor_msgs/JointState     four wheel joints
publish    /clock             rosgraph_msgs/Clock

service    /rally/serve        std_srvs/Trigger            spawn one legal serve

Frames
------
odom        the pose the robot had when the bridge finished settling: origin at
            the robot, +X along the camera axis (toward the court net), +Y left,
            +Z up, so theta = 0 at power-on
base_link   the vehicle frame: +X forward (camera axis), +Y left, +Z up. The MJCF
            base body is rotated 90 deg from it (+X left, +Y rearward), which is
            folded into the published poses and IMU axes
imu_link    the `imu` site, coincident with base_link

cmd_vel is in the *vehicle* frame and follows the body as it turns:
linear.x = forward along the camera axis [m/s], linear.y = left [m/s],
angular.z = CCW [rad/s]. 0.5 s without a message stops the robot (deadman).

Run (needs ROS2 sourced):
    source /opt/ros/jazzy/setup.bash
    .venv/bin/python ros2_bridge.py            # viewer window
    .venv/bin/python ros2_bridge.py --headless # no window
"""
from __future__ import annotations

import argparse
import contextlib
import math
import random
import time

import motrixsim
import numpy as np
import rclpy
from builtin_interfaces.msg import Time
from geometry_msgs.msg import TransformStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger
from tf2_msgs.msg import TFMessage

XML = "rally_court.xml"
WHEELS = ("FL", "FR", "RL", "RR")

# measured on this model: body speed per wheel rad/s (all four at 5 rad/s ->
# 0.4979 m/s), lateral (FL-RL+ 5 rad/s -> 0.4856 m/s), yaw (5 rad/s -> 0.9307 rad/s)
G_FORE = 0.09958          # base_link x [m/s] per rad/s
G_LAT = 0.09712           # base_link y [m/s] per rad/s
G_YAW = 0.18613           # base_link z [rad/s] per rad/s

VMAX_FORE, VMAX_LAT, WMAX = 4.0, 4.5, 3.0    # spec limits, clamp on the command
WHEEL_KP, TORQUE_LIMIT = 10.0, 12.0
BODY_KP, YAW_KP = 0.30, 0.30                 # trim on the measured body velocity
DEADMAN = 0.5                                 # s without cmd_vel -> stop
PUB_HZ = 120.0

# base_link -> vehicle(forward,left,up) as a rotation matrix
M = np.array([[0.0, -1.0, 0.0],
              [1.0, 0.0, 0.0],
              [0.0, 0.0, 1.0]])


# ------------------------------------------------------------------ ball model
BALL_R = 0.0335                  # m, ball radius (MJCF)
COURT_X, SINGLES_Y = 11.885, 4.115
SERVICE_X = 6.4                  # service line, |x|
NET_TOP = 1.07                   # net collision box top (flat in the MJCF)
GRAV = np.array([0.0, 0.0, -9.81])
# aerodynamic drag of a tennis ball: 0.5 * rho * Cd * A / m  [1/m]
# rho 1.2 kg/m3, Cd 0.6, A = pi r^2, m 0.057 kg.  The MJCF has no fluid model,
# so the bridge applies this to the ball every step (and the serve solver uses it)
DRAG_K = 0.5 * 1.2 * 0.6 * math.pi * BALL_R ** 2 / 0.057


def fly_step(p, v, h):
    """One explicit step of the drag-affected flight."""
    speed = float(np.linalg.norm(v))
    return p + v * h, v + (GRAV - DRAG_K * speed * v) * h


def flight(p0, v0, dt=0.0005, max_t=3.0):
    """Fly ballistically (drag included).  Returns (landing xy, net clearance).

    Landing is the first floor touch; clearance is the ball surface height over
    the net box at x = 0 (None if the flight never crosses the net).
    """
    p, v = np.array(p0, float), np.array(v0, float)
    clear = None
    t = 0.0
    while t < max_t:
        prev = p
        p, v = fly_step(p, v, dt)
        t += dt
        if clear is None and prev[0] > 0.0 >= p[0] and p[0] != prev[0]:
            u = (0.0 - prev[0]) / (p[0] - prev[0])
            clear = float(prev[2] + u * (p[2] - prev[2])) - NET_TOP - BALL_R
        if p[2] <= BALL_R and v[2] < 0.0:
            return p[:2].copy(), clear
    return None, clear


def _launch(speed, azimuth, elevation):
    """Launch velocity of magnitude `speed`; azimuth is measured from the -X
    axis (the direction the serve travels) toward +Y."""
    return np.array([-speed * math.cos(elevation) * math.cos(azimuth),
                     speed * math.cos(elevation) * math.sin(azimuth),
                     speed * math.sin(elevation)])


def solve_speed(spawn, target, speed, elevation=0.15):
    """Launch velocity of magnitude `speed` that lands on `target` (xy)."""
    azimuth = 0.0
    for _ in range(6):
        land, _ = flight(spawn, _launch(speed, azimuth, elevation))
        if land is None:
            return None
        err = land - target[:2]
        if np.linalg.norm(err) < 5e-3:
            break
        J = np.empty((2, 2))
        for i, (da, de) in enumerate(((1e-3, 0.0), (0.0, 1e-3))):
            land_i, _ = flight(spawn, _launch(speed, azimuth + da, elevation + de))
            J[:, i] = (land_i - land) / 1e-3
        step = np.linalg.solve(J, err)
        azimuth -= step[0]
        elevation -= step[1]
    return _launch(speed, azimuth, elevation)


def plan_serve(rng, spawn, speed, clearance_min, clearance_spread=0.35):
    """A legal serve at the requested speed and minimum net clearance.

    Returns (v0, target, clearance) or (None, reason, None) if the speed and
    geometry cannot produce a legal serve from this spawn.
    """
    want = clearance_min + rng.uniform(0.0, clearance_spread)
    scan = []
    for elev_deg in np.arange(1.0, 40.0, 0.5):
        land, clear = flight(spawn, _launch(speed, 0.0, math.radians(elev_deg)), dt=0.002)
        if land is not None and clear is not None:
            scan.append((clear, land[0], elev_deg))
    legal = [c for c in scan if c[0] >= clearance_min and -SERVICE_X + 0.4 < c[1] < -0.8]
    if not legal:
        return None, (f"speed {speed:.1f} m/s cannot clear the net by {clearance_min:.2f} m "
                      f"and bounce inside the service box"), None
    clear_t, x_land, elev_deg = min(legal, key=lambda c: abs(c[0] - want))
    side = 1.0 if spawn[1] >= 0.0 else -1.0
    target = np.array([x_land, -side * rng.uniform(0.5, 3.6), BALL_R])
    v0 = solve_speed(spawn, target, speed, math.radians(elev_deg))
    if v0 is None:
        return None, "no launch solution for that speed", None
    _, clear = flight(spawn, v0)
    return v0, target, clear


def quat_to_mat(q) -> np.ndarray:
    i, j, k, w = (float(v) for v in q)
    return np.array([
        [1 - 2 * (j * j + k * k), 2 * (i * j - w * k), 2 * (i * k + w * j)],
        [2 * (i * j + w * k), 1 - 2 * (i * i + k * k), 2 * (j * k - w * i)],
        [2 * (i * k - w * j), 2 * (j * k + w * i), 1 - 2 * (i * i + j * j)],
    ])


def mat_to_quat(R: np.ndarray):
    """Rotation matrix -> (i, j, k, w), matching get_link_poses."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        i = (R[2, 1] - R[1, 2]) / s
        j = (R[0, 2] - R[2, 0]) / s
        k = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        i = 0.25 * s
        j = (R[0, 1] + R[1, 0]) / s
        k = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        i = (R[0, 1] + R[1, 0]) / s
        j = 0.25 * s
        k = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        i = (R[0, 2] + R[2, 0]) / s
        j = (R[1, 2] + R[2, 1]) / s
        k = 0.25 * s
    return np.array([i, j, k, w])


def stamp(sim_time: float) -> Time:
    ns = round(sim_time * 1e9)
    return Time(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)


class Bridge:
    """Physics-side state: reset, drive, sensor reads, ROS publishers."""

    def __init__(self, model, data, node: Node):
        self.m, self.d, self.node = model, data, node
        self.dt = float(model.options.timestep)
        self.base = list(model.link_names).index("base")
        self.wheels = [model.get_joint(f"{p}_wheel_joint") for p in WHEELS]
        self.wheel_names = [f"{p}_wheel_joint" for p in WHEELS]
        self.body = model.get_body("base")
        self.pub_imu = node.create_publisher(Imu, "/imu/data_raw", 10)
        self.pub_odom = node.create_publisher(Odometry, "/odom", 10)
        self.pub_tf = node.create_publisher(TFMessage, "/tf", 10)
        self.pub_js = node.create_publisher(JointState, "/joint_states", 10)
        self.pub_clock = node.create_publisher(Clock, "/clock", 10)
        self.pub_state = node.create_publisher(Float64MultiArray, "/mecanum/state", 10)
        self.sub_cmd = node.create_subscription(Twist, "/mecanum/cmd_vel", self.on_cmd, 10)
        self.sub_wheel = node.create_subscription(
            Float64MultiArray, "/mecanum/wheel_cmd", self.on_wheel, 10)
        self.cmd = np.zeros(3)              # vehicle (forward, left, yaw)
        self.cmd_time = -1e9
        self.wheel_cmd = np.zeros(4)        # wheel rad/s, FL FR RL RR
        self.wheel_time = -1e9
        self.log = node.get_logger()
        self.ball = model.get_body("ball")
        self.ball_fb = next(f for f in model.floating_bases if f.name == "ball_free")
        self.rng = random.Random()
        node.declare_parameter("serve_speed", 20.0)        # m/s at launch
        node.declare_parameter("serve_clearance", 0.45)    # m over the net box
        self.srv_serve = node.create_service(Trigger, "/rally/serve", self.on_serve)
        self.t = 0.0
        self.v_cmd = np.zeros(3)
        self.p_start = np.zeros(3)
        self.R_start = np.eye(3)

    # ------------------------------------------------------------------ setup
    def reset_to_origin(self, settle: float = 1.5) -> None:
        """Put the robot at the court origin (baseline centre), camera at the net."""
        q = self.d.dof_pos.copy()
        q[:7] = (-11.885, 0.0, 0.156, 0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))
        self.d.set_dof_pos(q, self.m)
        self.d.set_dof_vel(np.zeros(self.d.dof_vel.shape[0], np.float32))
        self.d.actuator_ctrls = np.zeros(4, np.float32)
        self.m.forward_kinematic(self.d)
        for _ in range(int(settle / self.dt)):
            self.step(control=False)
        pose = np.ravel(self.m.get_link_poses(self.d)[self.base])
        self.p_start = pose[:3].copy()
        self.R_start = quat_to_mat(pose[3:7])
        self.cmd = np.zeros(3)
        self.cmd_time = -1e9

    def on_wheel(self, msg: Float64MultiArray) -> None:
        """Raw wheel speed targets [rad/s] (FL, FR, RL, RR); takes over from cmd_vel."""
        if len(msg.data) != 4:
            self.node.get_logger().warn(f"wheel_cmd needs 4 values, got {len(msg.data)}")
            return
        self.wheel_cmd = np.array(msg.data, dtype=float)
        self.wheel_time = self.t

    def on_cmd(self, msg: Twist) -> None:
        self.cmd = np.array([msg.linear.x, msg.linear.y, msg.angular.z], dtype=float)
        self.cmd_time = self.t

    def on_serve(self, request, response):
        try:
            ok, msg = self.serve()
        except Exception as exc:            # never let a bad request kill the sim
            ok, msg = False, f"serve failed: {exc!r}"
        if ok:
            self.log.info(f"serve: {msg}")
        else:
            self.log.warning(f"serve: {msg}")
        response.success = ok
        response.message = msg
        return response

    def spawn_point(self, rng) -> np.ndarray:
        """Random ball toss behind the opposite baseline, between centre mark
        and the singles sideline (where a server is allowed to stand)."""
        side = 1.0 if rng.random() < 0.5 else -1.0
        return np.array([COURT_X + rng.uniform(0.2, 1.4),
                         side * rng.uniform(0.3, 3.8),
                         rng.uniform(2.0, 2.6)])

    def serve(self, seed: int | None = None) -> tuple[bool, str]:
        """Spawn one legal serve at the configured speed and clearance."""
        rng = random.Random(seed) if seed is not None else self.rng
        speed = float(self.node.get_parameter("serve_speed").value)
        clearance = float(self.node.get_parameter("serve_clearance").value)
        spawn = self.spawn_point(rng)
        v0, target, clear = plan_serve(rng, spawn, speed, clearance)
        if v0 is None:
            return False, str(target)
        self.ball_fb.set_translation(self.d, spawn.astype(np.float32))
        self.ball_fb.set_rotation(self.d, np.array([1.0, 0.0, 0.0, 0.0], np.float32))
        self.ball_fb.set_global_linear_velocity(self.d, v0.astype(np.float32))
        self.ball_fb.set_global_angular_velocity(self.d, np.zeros(3, np.float32))
        self.m.forward_kinematic(self.d)
        return True, (f"spawn ({spawn[0]:.2f}, {spawn[1]:.2f}, {spawn[2]:.2f}) -> "
                      f"target ({target[0]:.2f}, {target[1]:.2f}) "
                      f"speed {np.linalg.norm(v0):.1f} m/s clearance {clear:.2f} m")

    def _drag(self) -> None:
        """Quadratic air drag on the ball (the MJCF has no fluid model)."""
        v = self.ball_fb.get_global_linear_velocity(self.d)
        speed = float(np.linalg.norm(v))
        if speed > 1e-3:
            self.ball_fb.set_global_linear_velocity(
                self.d, (v - (DRAG_K * speed * v) * self.dt).astype(np.float32))

    # ------------------------------------------------------------------ state
    def pose(self):
        p = np.ravel(self.m.get_link_poses(self.d)[self.base])
        return p[:3], quat_to_mat(p[3:7])

    def body_velocity(self):
        """(body linear velocity, body angular velocity) in base_link."""
        R = self.pose()[1]
        v = R.T @ np.ravel(self.m.get_sensor_value("base_linvel", self.d))
        w = R.T @ np.ravel(self.m.get_sensor_value("imu_angvel_world", self.d))
        return v, w

    # ------------------------------------------------------------------- step
    def step(self, control: bool = True) -> None:
        if control:
            self.control()
        self._drag()
        self.m.step(self.d)
        self.t += self.dt

    def control(self) -> None:
        if (self.t - self.wheel_time) <= DEADMAN:          # raw wheel targets win
            wheel_w = np.array([float(np.ravel(j.get_dof_vel(self.d))[0]) for j in self.wheels])
            ctrl = np.clip(WHEEL_KP * (self.wheel_cmd - wheel_w), -TORQUE_LIMIT, TORQUE_LIMIT)
            self.d.actuator_ctrls = ctrl.astype(np.float32)
            return
        alive = (self.t - self.cmd_time) <= DEADMAN
        fwd, left, yaw = self.cmd if alive else (0.0, 0.0, 0.0)
        fwd = float(np.clip(fwd, -VMAX_FORE, VMAX_FORE))
        left = float(np.clip(left, -VMAX_LAT, VMAX_LAT))
        yaw = float(np.clip(yaw, -WMAX, WMAX))
        self.v_cmd = np.array([fwd, left, yaw])          # vehicle frame
        # vehicle (forward, left) -> base_link (left, -forward)
        target = np.array([left, -fwd, yaw])
        v, w = self.body_velocity()
        v_cmd = target[:2] + BODY_KP * (target[:2] - v[:2])
        w_cmd = target[2] + YAW_KP * (target[2] - w[2])
        common = -v_cmd[0] / G_FORE
        lat = v_cmd[1] / G_LAT
        yaw_c = w_cmd / G_YAW
        speeds = np.array([common + lat + yaw_c,
                           common - lat - yaw_c,
                           common - lat + yaw_c,
                           common + lat - yaw_c])
        wheel_w = np.array([float(np.ravel(j.get_dof_vel(self.d))[0]) for j in self.wheels])
        ctrl = np.clip(WHEEL_KP * (speeds - wheel_w), -TORQUE_LIMIT, TORQUE_LIMIT)
        self.d.actuator_ctrls = ctrl.astype(np.float32)

    # -------------------------------------------------------------- publishers
    def publish(self) -> None:
        t_stamp = stamp(self.t)
        p, R = self.pose()
        R_bl = M @ self.R_start.T @ R @ M.T          # base_link (= vehicle) in odom
        p_odom = M @ (self.R_start.T @ (p - self.p_start))
        q_odom = mat_to_quat(R_bl)
        v = M @ (R.T @ np.ravel(self.m.get_sensor_value("base_linvel", self.d)))
        w = M @ (R.T @ np.ravel(self.m.get_sensor_value("imu_angvel_world", self.d)))
        theta = math.atan2(2.0 * (q_odom[3] * q_odom[2] + q_odom[0] * q_odom[1]),
                           1.0 - 2.0 * (q_odom[1] ** 2 + q_odom[2] ** 2))

        imu = Imu()
        imu.header.stamp = t_stamp
        imu.header.frame_id = "imu_link"
        imu.orientation_covariance[0] = -1.0          # 6-axis only: no orientation
        # MuJoCo accelerometer is +g at rest; ROS (REP-145) wants -g
        acc = M @ (-np.ravel(self.m.get_sensor_value("imu_acc", self.d)))
        imu.linear_acceleration.x, imu.linear_acceleration.y, imu.linear_acceleration.z = (float(a) for a in acc)
        gyro = M @ np.ravel(self.m.get_sensor_value("imu_gyro", self.d))
        imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z = (float(g) for g in gyro)
        self.pub_imu.publish(imu)

        odom = Odometry()
        odom.header.stamp = t_stamp
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"
        odom.pose.pose.position.x = float(p_odom[0])
        odom.pose.pose.position.y = float(p_odom[1])
        odom.pose.pose.orientation.z = float(q_odom[2])
        odom.pose.pose.orientation.w = float(q_odom[3])
        odom.twist.twist.linear.x = float(v[0])
        odom.twist.twist.linear.y = float(v[1])
        self.pub_odom.publish(odom)

        tf = TFMessage()
        tf.transforms = [self._tf("odom", "base_link", (p_odom[0], p_odom[1], 0.0), q_odom, t_stamp),
                         self._tf("base_link", "imu_link", (0.0, 0.0, 0.0),
                                  np.array([0.0, 0.0, 0.0, 1.0]), t_stamp)]
        self.pub_tf.publish(tf)

        js = JointState()
        js.header.stamp = t_stamp
        js.name = list(self.wheel_names)
        js.position = [float(np.ravel(j.get_dof_pos(self.d))[0]) for j in self.wheels]
        js.velocity = [float(np.ravel(j.get_dof_vel(self.d))[0]) for j in self.wheels]
        self.pub_js.publish(js)

        clock = Clock()
        clock.clock = t_stamp
        self.pub_clock.publish(clock)

        state = Float64MultiArray()
        state.data = [float(p_odom[0]), float(p_odom[1]), float(theta), float(v[0]), float(v[1])]
        self.pub_state.publish(state)

    @staticmethod
    def _tf(parent: str, child: str, xyz, quat, t_stamp):
        tr = TransformStamped()
        tr.header.stamp = t_stamp
        tr.header.frame_id = parent
        tr.child_frame_id = child
        tr.transform.translation.x, tr.transform.translation.y, tr.transform.translation.z = (float(c) for c in xyz)
        tr.transform.rotation.x, tr.transform.rotation.y = float(quat[0]), float(quat[1])
        tr.transform.rotation.z, tr.transform.rotation.w = float(quat[2]), float(quat[3])
        return tr

    # -------------------------------------------------------------------- run
    def run(self, headless: bool, fps: int = 60, speed: float = 1.0) -> None:
        """Step physics in real time (x speed), publish at PUB_HZ, spin ROS."""
        app_cm = contextlib.nullcontext(None)
        if not headless:
            import motrixsim.render as mr

            app_cm = mr.RenderApp(headless=False, fps=fps)
        frame = 1.0 / fps
        max_steps = max(1, int(4 * frame / self.dt))
        with app_cm as app:
            if app is not None:
                app.launch(self.m)
            t0, sim0, pub_next = time.monotonic(), self.t, self.t
            while rclpy.ok():
                rclpy.spin_once(self.node, timeout_sec=0.0)
                target = sim0 + (time.monotonic() - t0) * speed
                n = 0
                while self.t < target and n < max_steps:
                    self.step()
                    n += 1
                    if self.t >= pub_next:
                        self.publish()
                        pub_next += 1.0 / PUB_HZ
                if app is not None:
                    app.sync(self.d)
                time.sleep(frame if app is not None else 0.002)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true", help="no viewer window")
    ap.add_argument("--xml", default=XML)
    ap.add_argument("--speed", type=float, default=1.0, help="real-time factor (>1 faster)")
    args = ap.parse_args()

    model = motrixsim.load_model(args.xml)
    data = motrixsim.SceneData(model)
    rclpy.init()
    node = Node("mecanum_drive")
    bridge = Bridge(model, data, node)
    bridge.reset_to_origin()
    node.get_logger().info(
        f"robot at court origin {np.round(bridge.p_start, 3)} - waiting for /mecanum/cmd_vel")
    try:
        bridge.run(args.headless, speed=args.speed)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
