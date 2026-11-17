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
from ball_physics import (
    BALL_R, COURT_X, DRAG_K, G_FORE, G_LAT, G_YAW, SERVICE_X, SINGLES_Y,
    TANGENT_LOSS, TORQUE_LIMIT, WHEEL_KP, bounce_cor, flight,
)

GRAVITY = 9.81


def sample_serve(rng, speed_range, clearance_range, aim=(-4.9, 0.0), radius=(1.5, 1.0)):
    """Fast approximate serve for training resets.

    The first bounce is sampled around `aim` within `radius`, which controls how
    far from the robot the ball comes down (the post-bounce flight lands ~7 m
    past the bounce).  The serve is solved drag-free and rejected unless the
    drag-free arc clears the net generously, so the real (drag affected) ball
    still passes over it.  Microseconds; the exact planner in ros2_bridge is
    used for the real serves.
    """
    v0 = np.array([-10.0, 0.0, 2.0])
    spawn = np.array([COURT_X + 1.0, 0.0, 2.2])
    for _ in range(60):
        bx = float(np.clip(aim[0] + rng.uniform(-radius[0], radius[0]), -6.2, -1.2))
        by = float(np.clip(aim[1] + rng.uniform(-radius[1], radius[1]), -3.6, 3.6))
        side = -1.0 if by >= 0.0 else 1.0          # diagonal: spawn on the other side
        spawn = np.array([COURT_X + rng.uniform(0.2, 1.4),
                          side * rng.uniform(0.3, 3.8),
                          rng.uniform(2.0, 2.6)])
        target = np.array([bx, by, BALL_R])
        speed = rng.uniform(*speed_range)
        dx, dy = target[0] - spawn[0], target[1] - spawn[1]
        L = math.hypot(dx, dy)
        dz = target[2] - spawn[2]
        disc = speed ** 4 - GRAVITY * (GRAVITY * L * L + 2.0 * dz * speed * speed)
        if disc < 0.0:
            continue
        tan_t = (speed * speed - math.sqrt(disc)) / (GRAVITY * L)   # flat arc
        elev = math.atan(tan_t)
        v0 = np.array([speed * math.cos(elev) * dx / L,
                       speed * math.cos(elev) * dy / L,
                       speed * math.sin(elev)])
        if spawn[0] * v0[0] >= 0.0:
            continue
        t_net = spawn[0] / -v0[0]
        z_net = spawn[2] + v0[2] * t_net - 0.5 * GRAVITY * t_net * t_net
        if z_net - 1.07 - BALL_R < rng.uniform(*clearance_range) + 0.3:
            continue
        return spawn, v0
    return spawn, v0

XML = "rally_court.xml"
CTRL_HZ = 50.0
EPISODE_S = 4.0
START = (-11.885, 0.0)          # baseline centre, robot facing the net
CAGE_HALF = 0.21                # cage footprint half width
CAGE_LOW, CAGE_HIGH = -0.06, 0.36   # base-frame z window of a catch
CATCH_Z = 0.30                      # world height where the approach is scored
VMAX_FORE, VMAX_LAT, WMAX = 4.0, 4.5, 3.0
OBS_DIM = 12


class CatchEnv:
    def __init__(self, num_envs: int = 32, seed: int = 0, serve_speed=(16.0, 20.0),
                 serve_clearance=(0.45, 0.8), aim=(-4.9, 0.0), radius=(1.5, 1.0)):
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
        self.aim = aim                    # curriculum: where the serve bounces
        self.radius = radius
        self.model.forward_kinematic(self.data)
        self.t = np.zeros(self.n)
        self.done = np.ones(self.n, dtype=bool)
        self.caught = np.zeros(self.n)
        self.prev_bz = np.zeros(self.n)
        self._pred = None
        self._pred_age = 99
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
            pos[e, self.ball_p[:3]] = spawn
            pos[e, self.ball_p[3:]] = (1.0, 0.0, 0.0, 0.0)
            vel[e, self.ball_v] = np.concatenate([v0, (0.0, 0.0, 0.0)])
        self.data.set_dof_pos(pos, self.model)
        self.data.set_dof_vel(vel)
        self.data.actuator_ctrls = np.zeros((self.n, 4), np.float32)
        self.model.forward_kinematic(self.data)
        self.t[idx] = 0.0
        self.caught[idx] = 0.0
        self.done[idx] = False
        self.prev_bz[idx] = pos[idx, self.ball_p[2]]
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
        """Vehicle-frame (forward, left, yaw) -> wheel speed targets -> torques."""
        fwd = np.clip(action[:, 0], -VMAX_FORE, VMAX_FORE)
        left = np.clip(action[:, 1], -VMAX_LAT, VMAX_LAT)
        yaw = np.clip(action[:, 2], -WMAX, WMAX)
        # vehicle (forward, left) -> base_link (left, -forward)
        vx_b, vy_b = left, -fwd
        c = -vx_b / G_FORE
        l = vy_b / G_LAT
        y = yaw / G_YAW
        targets = np.stack([c + l + y, c - l - y, c - l + y, c + l - y], axis=1)
        w = self.data.dof_vel[:, self.wheel_v]
        self.data.actuator_ctrls = np.clip(WHEEL_KP * (targets - w),
                                           -TORQUE_LIMIT, TORQUE_LIMIT).astype(np.float32)

    # ------------------------------------------------------------ intercept
    def _predict(self, z_target=0.30, dt=0.02, horizon=3.0):
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

    def _obs(self):
        p, q = self.robot_pose()
        v = self.model.get_link_velocities(self.data)[:, self.base_i, :3]
        yaw = self.yaw_of(q)
        bp = self.ball_pose()
        bv = self.ball_vel()
        d = bp - p
        obs = np.concatenate([
            (p[:, :2] - np.array(START)) / 5.0,      # robot offset from home
            v[:, :2] / 5.0,                           # robot velocity
            np.stack([np.sin(yaw), np.cos(yaw)], axis=1),
            d / 5.0,                                  # ball relative position
            bv / 20.0,                                # ball velocity
        ], axis=1)
        return obs.astype(np.float32)

    def obs(self):
        return self._obs()

    def close(self):
        pass


def make(num_envs=32, seed=0):
    return CatchEnv(num_envs=num_envs, seed=seed)
