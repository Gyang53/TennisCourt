"""Vectorised catch environment: a mecanum robot and a served tennis ball.

Built on one batched MotrixSim scene (SceneData(model, batch=[N])), so all
environments step in parallel in the Rust backend.  Control at CTRL_HZ; the
ball's flight needs the 0.5 ms physics step (see README).

Episode: the robot waits at the baseline centre, a legal serve comes from the
far half, and the robot must put the ball inside its cage.  A catch is a
geometric test in the robot's base frame: the ball inside the cage footprint,
below the rim.

The agent commands the same interface as the ROS bridge (vehicle frame
forward/left/yaw rate), the wheel IK and torque servo use the gains measured in
ros2_bridge.py.
"""
from __future__ import annotations

import math

import motrixsim
import numpy as np
from ball_physics import (  # noqa: F401  (sample_serve/clears_net/first_catchable live here)
    BALL_R, COURT_X, DRAG_K, G_FORE, G_LAT, G_YAW, SERVICE_X, SINGLES_Y,
    TANGENT_LOSS, TORQUE_LIMIT, WHEEL_CTRL_MAX, WHEEL_KP, bounce_cor, catch_window, clears_net,
    first_catchable, flight, fly_step, sample_serve,
)

GRAVITY = 9.81


XML = "rally_court.xml"
CTRL_HZ = 50.0
EPISODE_S = 4.0
from ball_physics import START  # baseline centre, robot facing the net
CAGE_HALF = 0.21                # cage footprint half width
CAGE_LOW, CAGE_HIGH = -0.06, 0.36   # base-frame z window of a catch
CATCH_Z = 0.30                      # world height where the approach is scored
VMAX_FORE, VMAX_LAT, WMAX = 4.0, 4.5, 3.0
# spec limits of the drive (measured in test_motion.py):
#   body X (wheel axis)  accel 3.5, decel 5.0, vmax 4.0
#   body Y (lateral)     accel 2.5, decel 5.0, vmax 4.5
ACCEL_FORE, DECEL_FORE = 3.5, 5.0
ACCEL_LAT, DECEL_LAT = 2.5, 5.0
WHEEL_SLEW = 10.0                # rad/s per control step on the wheel speed target
TRAJ_POINTS = 16                 # trajectory samples fed to the policy
OBS_MODES = {                     # what the policy sees
    "state": 12,                  # robot + current ball state (the shipped policy)
    "traj": 6 + TRAJ_POINTS * 3 + 1,      # robot + path since the serve
    "both": 12 + TRAJ_POINTS * 3 + 1,     # robot + ball state + path
}
OBS_MODE_FOR_DIM = {v: k for k, v in OBS_MODES.items()}   # checkpoint dim -> mode
OBS_DIM = OBS_MODES["both"]


class CatchEnv:
    def __init__(self, num_envs: int = 32, seed: int = 0, serve_speed=(16.0, 20.0),
                 serve_clearance=(0.45, 0.8), aim=(-4.9, 0.0), radius=(1.5, 1.0),
                 obs_mode="both"):
        self.n = num_envs
        self.model = motrixsim.load_model(XML)
        self.data = motrixsim.SceneData(self.model, batch=[num_envs])
        self.dt = float(self.model.options.timestep)
        self.substeps = max(1, int(round(1.0 / CTRL_HZ / self.dt)))
        self.rng = np.random.default_rng(seed)
        names = list(self.model.link_names)
        self.base_i = names.index("base")
        self.ball_i = names.index("ball")
        self.ball_fb = next(f for f in self.model.floating_bases if f.name == "ball_free")
        self.ball_p = np.asarray(self.ball_fb.dof_pos_indices)
        self.ball_v = np.asarray(self.ball_fb.dof_vel_indices)
        self.wheels = [self.model.get_joint(f"{p}_wheel_joint") for p in ("FL", "FR", "RL", "RR")]
        self.wheel_v = np.array([self.model.joint_dof_vel_indices[self.model.get_joint_index(f"{p}_wheel_joint")]
                                 for p in ("FL", "FR", "RL", "RR")])
        # motor order in the MJCF is FL, FR, RL, RR
        self.serve_speed = serve_speed
        self.serve_clearance = serve_clearance
        self.obs_mode = obs_mode
        self.aim = aim                    # curriculum: where the serve bounces
        self.radius = radius
        self.model.forward_kinematic(self.data)
        self.t = np.zeros(self.n)
        self.done = np.ones(self.n, dtype=bool)
        self.caught = np.zeros(self.n)
        self.prev_bz = np.zeros(self.n)
        self.catch_xy = np.zeros((self.n, 2))    # oracle catch point (teacher target)
        self.catch_t = np.zeros(self.n)
        self.traj = np.zeros((self.n, 400, 3), np.float32)   # ball path since the serve
        self.age = np.zeros(self.n, dtype=int)
        self._pred = None
        self._pred_age = 99
        self._minz = np.full(self.n, np.inf)
        self._v_cmd = None
        self._wheel_ref = None
        self.reset()

    # ------------------------------------------------------------- utilities
    def robot_pose(self):
        p = self.model.get_link_poses(self.data)[:, self.base_i]
        return p[:, :3].copy(), p[:, 3:7].copy()

    def ball_pose(self):
        return self.model.get_link_poses(self.data)[:, self.ball_i, :3].copy()

    def ball_vel(self):
        return self.model.get_link_velocities(self.data)[:, self.ball_i, :3].copy()

    def yaw_of(self, quat):
        i, j, k, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        return np.arctan2(2 * (w * k + i * j), 1 - 2 * (j * j + k * k))

    def base_frame_ball(self):
        """Ball position and velocity in the robot's base frame (xy only used)."""
        p, q = self.robot_pose()
        yaw = self.yaw_of(q)
        d = self.ball_pose() - p
        c, s = np.cos(-yaw), np.sin(-yaw)
        dx = c * d[:, 0] - s * d[:, 1]
        dy = s * d[:, 0] + c * d[:, 1]
        return dx, dy, d[:, 2]

    # ----------------------------------------------------------------- reset
    def reset(self, idx=None):
        idx = np.arange(self.n) if idx is None else np.atleast_1d(idx)
        if not len(idx):
            return
        pos = self.data.dof_pos.copy()
        vel = self.data.dof_vel.copy()
        for e in idx:
            # robot at the baseline centre, settled, facing the net
            pos[e, :7] = (START[0], START[1], 0.1555, 0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))
            vel[e, :6] = 0.0
            # random legal serve
            spawn, v0 = sample_serve(self.rng, self.serve_speed, self.serve_clearance,
                                     self.aim, self.radius)
            xy, t_c = first_catchable(spawn, v0)
            self.catch_xy[e] = xy if xy is not None else spawn[:2]
            self.catch_t[e] = t_c if t_c is not None else 3.0
            pos[e, self.ball_p[:3]] = spawn
            pos[e, self.ball_p[3:]] = (1.0, 0.0, 0.0, 0.0)
            vel[e, self.ball_v] = np.concatenate([v0, (0.0, 0.0, 0.0)])
            xy, t_c = catch_window(spawn, v0)
            self.catch_xy[e] = xy if xy is not None else spawn[:2]
            self.catch_t[e] = t_c if t_c is not None else 3.0
        self.data.set_dof_pos(pos, self.model)
        self.data.set_dof_vel(vel)
        self.data.actuator_ctrls = np.zeros((self.n, 4), np.float32)
        self.model.forward_kinematic(self.data)
        self.t[idx] = 0.0
        self.caught[idx] = 0.0
        self.done[idx] = False
        self.prev_bz[idx] = pos[idx, self.ball_p[2]]
        self.traj[idx] = 0.0
        self.age[idx] = 0
        if self._v_cmd is not None:
            self._v_cmd[idx] = 0.0
        if self._wheel_ref is not None:
            self._wheel_ref[idx] = 0.0
        self.traj[idx, 0] = pos[idx][:, self.ball_p[:3]]
        self._pred_age = 99

    # -------------------------------------------------------------- physics
    def _ball_impulse(self, pos, vel):
        """Drag and bounce for every env whose ball moved during this substep."""
        bp = pos[:, self.ball_p[:3]]
        bv = vel[:, self.ball_v]
        speed = np.linalg.norm(bv, axis=1)
        live = speed > 1e-3
        if live.any():
            bv[live] -= (DRAG_K * speed[live, None] * bv[live]) * self.dt
        # bounce where the ball is at floor level and descending
        hit = (bp[:, 2] <= BALL_R + 0.002) & (bv[:, 2] < 0.0)
        if hit.any():
            vn = -bv[hit, 2]
            bv[hit, 0] *= TANGENT_LOSS
            bv[hit, 1] *= TANGENT_LOSS
            bv[hit, 2] = vn * np.array([bounce_cor(x) for x in vn])
            # spin of a rolling bounce, about the axis perpendicular to travel
            roll = np.stack([-bv[hit, 1] / BALL_R, bv[hit, 0] / BALL_R, np.zeros(hit.sum())], axis=1)
            vel[hit][:, self.ball_v[3:]] = roll
        vel[:, self.ball_v] = bv
        return vel

    def _advance(self):
        for _ in range(self.substeps):
            pos = self.data.dof_pos.copy()
            vel = self._ball_impulse(pos, self.data.dof_vel.copy())
            self.data.set_dof_pos(pos, self.model)
            self.data.set_dof_vel(vel)
            self.model.step(self.data)
        self.t += self.substeps * self.dt

    def _drive(self, action):
        """Vehicle-frame (forward, left, yaw) -> wheel speed targets -> torques.

        The commanded body velocity is ramped (ACCEL_LIMIT m/s^2): the policy can
        ask for the full 4.5 m/s at t=0, and without the ramp the wheel servo slams
        the wheels to ~50 rad/s in one step, they slip, and the chassis pitches up
        30 mm - the visible ``jump`` at startup.
        """
        fwd = np.clip(action[:, 0], -VMAX_FORE, VMAX_FORE)
        left = np.clip(action[:, 1], -VMAX_LAT, VMAX_LAT)
        yaw = np.clip(action[:, 2], -WMAX, WMAX)
        target = np.stack([fwd, left], axis=1)
        if self._v_cmd is None:
            self._v_cmd = np.zeros_like(target)
        # accelerate along the drive spec instead of jumping to the commanded
        # speed: asking for 4.5 m/s laterally from rest makes the mecanum rollers
        # slip and pitches the chassis up 30 mm (the "hop" at startup).  Ramping at
        # the spec rates keeps the wheels from breaking traction at all.
        if self._v_cmd is None:
            self._v_cmd = np.zeros_like(target)
        dt_c = self.substeps * self.dt
        prev = self._v_cmd
        for axis, (a_acc, a_dec) in enumerate(((ACCEL_FORE, DECEL_FORE),
                                               (ACCEL_LAT, DECEL_LAT))):
            tgt = target[:, axis]
            speeding_up = (np.abs(tgt) > np.abs(prev[:, axis])) & \
                          ((prev[:, axis] == 0.0) | (np.sign(tgt) == np.sign(prev[:, axis])))
            rate = np.where(speeding_up, a_acc, a_dec) * dt_c
            prev[:, axis] += np.clip(tgt - prev[:, axis], -rate, rate)
        fwd = self._v_cmd[:, 0]
        left = self._v_cmd[:, 1]
        # vehicle (forward, left) -> base_link (left, -forward)
        vx_b, vy_b = left, -fwd
        c = -vx_b / G_FORE
        l = vy_b / G_LAT
        y = yaw / G_YAW
        targets = np.stack([c + l + y, c - l - y, c - l + y, c + l - y], axis=1)
        # slew-limit the *wheel* targets: an instantaneous jump to ~46 rad/s makes
        # the wheels slip and the chassis pitch up 30 mm at startup.  10 rad/s per
        # control step reaches full speed in ~0.1 s, so the motion is unaffected.
        if self._wheel_ref is None:
            self._wheel_ref = np.zeros_like(targets)
        self._wheel_ref += np.clip(targets - self._wheel_ref, -WHEEL_SLEW, WHEEL_SLEW)
        # the actuators are velocity servos: the command is the wheel speed
        self.data.actuator_ctrls = np.clip(self._wheel_ref, -WHEEL_CTRL_MAX,
                                           WHEEL_CTRL_MAX).astype(np.float32)

    # ------------------------------------------------------------ intercept
    def _predict(self, z_target=0.30, dt=0.005, horizon=3.0):
        """Forward-simulate the ball (same drag + bounce model as the sim) and
        return where/when it next descends through z_target after a bounce.

        This is the catch window the robot can actually reach: the serve itself
        bounces too far up the court.
        """
        p = self.ball_pose()
        v = self.ball_vel()
        n = self.n
        t = np.zeros(n)
        bounced = np.zeros(n, dtype=bool)
        tgt = p[:, :2].copy()
        tc = np.full(n, horizon)
        active = np.ones(n, dtype=bool)
        prev_z = p[:, 2].copy()
        for _ in range(int(horizon / dt)):
            speed = np.linalg.norm(v, axis=1)
            v = v + (-DRAG_K * speed[:, None] * v + np.array([0.0, 0.0, -9.81])) * dt
            p = p + v * dt
            t += dt
            low = p[:, 2] <= BALL_R
            if low.any():
                vn = -v[low, 2]
                v[low, 2] = vn * np.array([bounce_cor(x) for x in vn])
                v[low, 0] *= TANGENT_LOSS
                v[low, 1] *= TANGENT_LOSS
                p[low, 2] = BALL_R
                bounced |= low
            cross = active & bounced & (prev_z > z_target) & (p[:, 2] <= z_target) & (v[:, 2] < 0)
            if cross.any():
                tgt[cross] = p[cross, :2]
                tc[cross] = t[cross]
                active &= ~cross
            prev_z = p[:, 2].copy()
            if not active.any():
                break
        return tgt, tc

    def intercept(self, z_target=0.30):
        """Cached catch point (the prediction only changes slowly)."""
        if self._pred is None or self._pred_age >= 4:
            self._pred = self._predict(z_target)
            self._pred_age = 0
        self._pred_age += 1
        return self._pred

    def intercept_xy(self, z_target=0.30):
        return self.intercept(z_target)[0]

    # ---------------------------------------------------------------- step
    def step(self, action):
        phase0 = self._potential()
        self._drive(np.asarray(action, dtype=float))
        self._advance()
        bp = self.ball_pose()
        self.age = np.minimum(self.age + 1, self.traj.shape[1] - 1)
        self.traj[np.arange(self.n), self.age] = bp
        dx, dy, dz = self.base_frame_ball()
        inside = (np.abs(dx) < CAGE_HALF) & (np.abs(dy) < CAGE_HALF) \
            & (dz > CAGE_LOW) & (dz < CAGE_HIGH)
        caught_now = inside & ~self.done & (self.caught == 0)
        self.caught[caught_now] = 1.0
        out = (np.abs(bp[:, 1]) > 9.0) | (bp[:, 0] < -15.0) | (bp[:, 0] > 15.0)
        timeout = self.t >= EPISODE_S
        done = ~self.done & (caught_now | out | timeout)
        reward = 0.25 * (self._potential() - phase0)     # potential shaping
        # dense approach reward: the moment the ball drops past the cage rim the
        # miss distance is what decides the catch, so pay for it directly
        crossing = (self.prev_bz > CATCH_Z) & (bp[:, 2] <= CATCH_Z) & (self.ball_vel()[:, 2] < 0)
        if crossing.any():
            pr, _ = self.robot_pose()
            miss = np.linalg.norm(bp[crossing, :2] - pr[crossing, :2], axis=1)
            reward[crossing] += 0.6 * np.clip(1.0 - miss / 0.6, 0.0, 1.0)
        # final approach: pull toward the ball once it is low and falling
        p_robot, _ = self.robot_pose()
        dist = np.linalg.norm(bp[:, :2] - p_robot[:, :2], axis=1)
        falling = (bp[:, 2] < 1.0) & (self.ball_vel()[:, 2] < 0)
        if falling.any():
            reward[falling] -= 0.01 * np.minimum(dist[falling], 3.0)
        self.prev_bz = bp[:, 2].copy()
        reward[caught_now] = 1.0
        reward[out & ~caught_now] -= 0.5
        self.done = self.done | done
        info = {"caught": self.caught.copy(), "timeout": timeout, "out": out}
        self.reset(np.where(self.done)[0])
        return self._obs(), reward.astype(np.float32), done, info

    def _potential(self):
        """Negative normalised distance from the cage to the ball (dense shaping)."""
        p, _ = self.robot_pose()
        d = np.linalg.norm(self.ball_pose()[:, :2] - p[:, :2], axis=1)
        return -np.minimum(d, 12.0) / 12.0

    def set_serves(self, spawns, v0s, park=True):
        """Install explicit serves (spawn, v0 per env) and park the robots away,
        so one batched pass can record where each ball enters the cage window."""
        pos = self.data.dof_pos.copy()
        vel = np.zeros_like(self.data.dof_vel)      # wheel spin from a previous pass must not carry over
        for e in range(self.n):
            rx, ry = (-2.0, -8.0) if park else START
            pos[e, :7] = (rx, ry, 0.1555, 0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))
            vel[e, :6] = 0.0
            pos[e, self.ball_p[:3]] = spawns[e]
            pos[e, self.ball_p[3:]] = (1.0, 0.0, 0.0, 0.0)
            vel[e, self.ball_v] = np.concatenate([v0s[e], (0.0, 0.0, 0.0)])
        self.data.set_dof_pos(pos, self.model)
        self.data.set_dof_vel(vel)
        self.data.actuator_ctrls = np.zeros((self.n, 4), np.float32)
        self.model.forward_kinematic(self.data)
        self.t[:] = 0.0
        self.caught[:] = 0.0
        self.done[:] = False
        self.traj[:] = 0.0
        self.age[:] = 0
        self.traj[:, 0] = spawns          # same first trajectory point reset() writes
        for e in range(self.n):                 # analytic target: computable from the
            xy, t_c = first_catchable(spawns[e], v0s[e])   # trajectory the policy sees
            self.catch_xy[e] = xy if xy is not None else spawns[e, :2]
            self.catch_t[e] = t_c if t_c is not None else 3.0

    def record_catch_points(self, z_enter=0.52, v_eff=3.6, t_react=0.2, steps=240):
        """Pass 1: run the balls with the robots parked and record, per env, the
        first entry into the cage window that the robot could reach from START."""
        found = np.zeros(self.n, dtype=bool)
        for _ in range(steps):
            # keep the robot out of the ball's way every step: a parked robot that
            # the ball happens to hit would corrupt the record
            pos = self.data.dof_pos.copy()
            pos[:, 0] = -2.0
            pos[:, 1] = -8.0
            self.data.set_dof_pos(pos, self.model)
            self.model.forward_kinematic(self.data)
            obs, _rew, done, _info = self.step(np.zeros((self.n, 3)))
            bp = self.ball_pose()
            # reachability from START, using this env's serve time
            d = np.linalg.norm(bp[:, :2] - np.array(START), axis=1)
            # an env whose episode ended has been reset to a *new* serve: its ball
            # is not the one we are recording, so it can never be recorded
            ok = ((bp[:, 2] <= z_enter) & ~found & ~done
                  & (d <= np.minimum(v_eff * np.maximum(self.t - t_react, 1e-6), 6.0)))
            if ok.any():
                self.catch_xy[ok] = bp[ok, :2]
                self.catch_t[ok] = self.t[ok]
                found |= ok
            if found.all() or done.all():
                break
        return found

    def teacher_action(self, lead=0.6, kv=0.0, kp_far=1.5, kp_near=2.5,
                       brake=0.0, early=0.0):
        """Scripted teacher: drive to the recorded catch point, timed, with an
        inner velocity loop (pure position feedback leaves a mecanum strafe
        error of ~0.24 m at the arrival).  Returns normalised actions."""
        p, q = self.robot_pose()
        yaw = self.yaw_of(q)
        vw = self.model.get_link_velocities(self.data)[:, self.base_i, :2]
        err = self.catch_xy - p[:, :2]
        kp = np.where(self.t < self.catch_t - lead - early, kp_far, kp_near)
        v_ref = np.clip(kp[:, None] * err, -4.0, 4.0)
        if brake > 0.0:
            # never command more speed than can be shed in the remaining distance
            dist = np.linalg.norm(err, axis=1)
            v_ok = np.sqrt(2.0 * brake * np.maximum(dist, 1e-6))
            n = np.maximum(np.linalg.norm(v_ref, axis=1, keepdims=True), 1e-6)
            v_ref = np.minimum(v_ref, v_ref / n * np.minimum(v_ok, 4.0)[:, None])
        v_cmd = np.clip(v_ref + kv * (v_ref - vw), -4.5, 4.5)
        c, s = np.cos(yaw), np.sin(yaw)
        fwd = v_cmd[:, 0] * s - v_cmd[:, 1] * c
        left = v_cmd[:, 0] * c + v_cmd[:, 1] * s
        raw = np.stack([fwd, left, np.zeros(self.n)], axis=1)
        return raw / np.array([4.0, 4.5, 3.0])

    def _obs(self):
        p, q = self.robot_pose()
        v = self.model.get_link_velocities(self.data)[:, self.base_i, :3]
        yaw = self.yaw_of(q)
        parts = [
            (p[:, :2] - np.array(START)) / 5.0,      # robot offset from home
            v[:, :2] / 5.0,                           # robot velocity
            np.stack([np.sin(yaw), np.cos(yaw)], axis=1),
        ]
        if self.obs_mode in ("state", "both"):
            bp = self.ball_pose()
            bv = self.ball_vel()
            parts.append((bp - p) / 5.0)              # ball relative to the robot
            parts.append(bv / 20.0)                   # ball velocity
        if self.obs_mode in ("traj", "both"):
            frac = np.linspace(0.0, 1.0, TRAJ_POINTS)[None, :]
            idx = np.clip((frac * self.age[:, None]).astype(int), 0, self.traj.shape[1] - 1)
            path = self.traj[np.arange(self.n)[:, None], idx]      # (n, K, 3)
            parts.append((path[:, :, :2] - np.array(START)).reshape(self.n, -1) / 5.0)
            parts.append(path[:, :, 2].reshape(self.n, -1) / 3.0)
            parts.append((self.age * self.substeps * self.dt)[:, None] / 4.0)
        return np.concatenate(parts, axis=1).astype(np.float32)

    def obs(self):
        return self._obs()

    def close(self):
        pass


def make(num_envs=32, seed=0):
    return CatchEnv(num_envs=num_envs, seed=seed)
