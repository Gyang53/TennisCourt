"""Catch environment on MuJoCo Warp (GPU physics), batched over thousands of worlds.

Same interface as `catch_env.CatchEnv` (obs layout, action space, reward, catch
predicate, teacher), so `bc_train.py`/`dagger_train.py`-style training and the
evaluation tools can drive either backend.

Two physics differences to the MotrixSim environment, and they are *improvements*
(see UNILAB.md): air drag comes from MuJoCo's native fluid model
(`<option density viscosity>`) instead of a bridge-side velocity update, and the
ball bounces through the contact solver's restitution instead of an explicit
impulse.

    .venv/bin/python ...   (needs the ms conda env: mujoco 4.0.1 + mujoco-warp 4.0.1)
"""
from __future__ import annotations

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import warp as wp

XML = "rally_court.xml"
CTRL_HZ = 50.0
EPISODE_S = 4.0
START = (-11.885, 0.0)
CAGE_HALF = 0.21
CAGE_LOW, CAGE_HIGH = -0.06, 0.36
CATCH_Z = 0.30
VMAX = np.array([4.0, 4.5, 3.0])
WHEELS = ("FL", "FR", "RL", "RR")
DRAG_DENSITY = 1.2       # kg/m^3, MuJoCo fluid model
DRAG_VISCOSITY = 1.8e-5  # Pa s
BOUNCE_SOLREF = "0.002 0.2"   # contact restitution for ball <-> ground


def _patched_xml(path: str) -> str:
    """Add MuJoCo's native fluid drag, a bouncy ball/floor contact, and contact
    headroom: the 36 mecanum rollers generate far more than MuJoCo's default 64
    constraints per world (`nefc overflow` kills the physics otherwise)."""
    with open(path) as f:
        s = f.read()
    s = s.replace('<option timestep="0.0005"',
                  f'<option density="{DRAG_DENSITY}" viscosity="{DRAG_VISCOSITY}" timestep="0.0005"')
    s = s.replace('solref="0.002 1.0"', f'solref="{BOUNCE_SOLREF}"')   # ball <-> ground pair
    s = s.replace("  <option ", '  <size njmax="2000" nconmax="4000"/>\n  <option ', 1)
    return s


class CatchEnvGPU:
    def __init__(self, num_envs=4096, seed=0, serve_speed=(12.0, 16.0),
                 serve_clearance=(0.6, 1.0), radius=(1.5, 1.5), aim=(-4.9, 0.0),
                 xml=XML, ctrl_hz=CTRL_HZ):
        self.n = num_envs
        self.mjm = mujoco.MjModel.from_xml_string(_patched_xml(xml))
        self.m = mjwarp.put_model(self.mjm)
        self.mjd = mujoco.MjData(self.mjm)
        # the 36 mecanum rollers blow past MuJoCo's default 64 constraints per
        # world; mjwarp takes the limits per world here, not from MJCF <size>
        # 36 mecanum rollers per robot generate hundreds of candidate contacts per
        # world; the wide phase needs the *total* budget scaled by the batch
        self.d = mjwarp.put_data(self.mjm, self.mjd, nworld=num_envs,
                                 nconmax=256, njmax=1024,
                                 naconmax=256 * num_envs, naccdmax=64 * num_envs)
        self.dt = float(self.mjm.opt.timestep)
        self.substeps = max(1, int(round(1.0 / ctrl_hz / self.dt)))
        self.rng = np.random.default_rng(seed)
        self.bid = lambda name: mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_BODY, name)
        self.base_i = self.bid("base")
        self.ball_i = self.bid("ball")
        ball_j = mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
        self.ball_q = int(self.mjm.jnt_qposadr[ball_j])          # 7 dofs
        self.ball_v = int(self.mjm.jnt_dofadr[ball_j])           # 6 dofs
        self.base_v = 0                                          # free joint first
        self.wheel_dof = np.array([
            int(self.mjm.jnt_dofadr[mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_JOINT, f"{p}_wheel_joint")])
            for p in WHEELS])
        self.serve_speed = serve_speed
        self.serve_clearance = serve_clearance
        self.radius = radius
        self.aim = aim
        self.t = np.zeros(num_envs)
        self.done = np.zeros(num_envs, dtype=bool)
        self.caught = np.zeros(num_envs)
        self.catch_xy = np.zeros((num_envs, 2))
        self.catch_t = np.zeros(num_envs)
        self._host = np.zeros((num_envs, self.mjm.nq), np.float32)   # writable mirror
        self._hostv = np.zeros((num_envs, self.mjm.nv), np.float32)

    # ------------------------------------------------------------- host <-> gpu
    def _pull(self):
        self._host[:] = self.d.qpos.numpy()
        self._hostv[:] = self.d.qvel.numpy()

    def _push(self):
        wp.copy(self.d.qpos, wp.array(self._host, dtype=wp.float32, device=self.d.qpos.device))
        wp.copy(self.d.qvel, wp.array(self._hostv, dtype=wp.float32, device=self.d.qvel.device))

    def _body(self, i):
        x = self.d.xpos.numpy()[:, i]
        q = self.d.xquat.numpy()[:, i]      # w, x, y, z
        return x, q

    # ------------------------------------------------------------------ serving
    def set_serves(self, spawns, v0s):
        """Install explicit serves; the analytic catch target is computed locally."""
        self._pull()
        for e in range(self.n):
            self._host[e, :7] = (*START, 0.1555, 1.0, 0.0, 0.0, 0.0)
            self._hostv[e, :6] = 0.0
            self._host[e, self.ball_q:self.ball_q + 3] = spawns[e]
            self._host[e, self.ball_q + 3:self.ball_q + 7] = (1.0, 0.0, 0.0, 0.0)
            self._hostv[e, self.ball_v:self.ball_v + 3] = v0s[e]
            self._hostv[e, self.ball_v + 3:self.ball_v + 6] = 0.0
            self.catch_xy[e], self.catch_t[e] = _analytic_catch(spawns[e], v0s[e])
        self._hostv[:, self.wheel_dof] = 0.0
        self._push()
        self.t[:] = 0.0
        self.done[:] = False
        self.caught[:] = 0.0

    # ------------------------------------------------------------------ physics
    def step(self, action):
        """action: (n,3) normalised vehicle-frame (forward, left, yaw)."""
        act = np.asarray(action, dtype=np.float64) * VMAX
        _, q = self._body(self.base_i)
        yaw = _yaw(q)
        c, s = np.cos(yaw), np.sin(yaw)
        vx_w, vy_w = act[:, 0], act[:, 1]                  # world velocity command
        vx_b = c * vx_w + s * vy_w                          # -> base frame
        vy_b = -s * vx_w + c * vy_w
        common = -vx_b / 0.09958
        lat = vy_b / 0.09712
        yaw_c = act[:, 2] / 0.18613
        targets = np.stack([common + lat + yaw_c, common - lat - yaw_c,
                            common - lat + yaw_c, common + lat - yaw_c], axis=1)
        wheel_w = self.d.qvel.numpy()[:, self.wheel_dof]
        ctrl = np.clip(10.0 * (targets - wheel_w), -12.0, 12.0).astype(np.float32)
        wp.copy(self.d.ctrl, wp.array(ctrl, dtype=wp.float32, device=self.d.ctrl.device))
        for _ in range(self.substeps):
            mjwarp.step(self.m, self.d)
        self.t += self.substeps * self.dt

        bp = self.d.xpos.numpy()[:, self.ball_i]
        rp, rq = self._body(self.base_i)
        yaw = _yaw(rq)
        d = bp - rp
        c, s = np.cos(-yaw), np.sin(-yaw)
        dx = c * d[:, 0] - s * d[:, 1]
        dy = s * d[:, 0] + c * d[:, 1]
        inside = ((np.abs(dx) < CAGE_HALF) & (np.abs(dy) < CAGE_HALF)
                  & (d[:, 2] > CAGE_LOW) & (d[:, 2] < CAGE_HIGH))
        caught_now = inside & ~self.done & (self.caught == 0)
        self.caught[caught_now] = 1.0
        out = (np.abs(bp[:, 1]) > 9.0) | (bp[:, 0] < -15.0) | (bp[:, 0] > 15.0)
        timeout = self.t >= EPISODE_S
        done = ~self.done & (caught_now | out | timeout)
        reward = np.zeros(self.n)
        reward[caught_now] = 1.0
        reward[out & ~caught_now] -= 0.5
        self.done = self.done | done
        info = {"caught": self.caught.copy()}
        return done, reward, info

    # -------------------------------------------------------------- observation
    def obs(self):
        rp, rq = self._body(self.base_i)
        bp = self.d.xpos.numpy()[:, self.ball_i]
        v = self.d.qvel.numpy()
        yaw = _yaw(rq)
        rel = bp - rp
        return np.concatenate([
            (rp[:, :2] - np.array(START)) / 5.0,
            (v[:, self.base_v:self.base_v + 2]) / 5.0,
            np.stack([np.sin(yaw), np.cos(yaw)], axis=1),
            rel / 5.0,
            (v[:, self.ball_v:self.ball_v + 3]) / 20.0,
        ], axis=1).astype(np.float32)

    # ------------------------------------------------------------------ teacher
    def teacher_action(self, lead=0.6, kp_far=1.5, kp_near=2.5):
        rp, _ = self._body(self.base_i)
        err = self.catch_xy - rp[:, :2]
        kp = np.where(self.t < self.catch_t - lead, kp_far, kp_near)
        v_des = np.clip(kp[:, None] * err, -4.0, 4.0)
        v_des = np.clip(v_des, VMAX[:2] * -1, VMAX[:2])
        raw = np.concatenate([v_des, np.zeros((self.n, 1))], axis=1)
        return raw / VMAX


def _yaw(q):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _analytic_catch(spawn, v0, z_enter=0.52, dt=0.002, v_eff=3.6, t_react=0.2, max_t=4.0):
    """First reachable entry into the cage window (same rule as the CPU env)."""
    from ball_physics import BALL_R, bounce_cor, fly_step, TANGENT_LOSS

    p = np.array(spawn, float)
    v = np.array(v0, float)
    t = 0.0
    while t < max_t:
        p, v = fly_step(p, v, dt)
        t += dt
        if p[2] <= BALL_R:
            vn = -v[2]
            v = v.copy()
            v[2] = vn * bounce_cor(vn)
            v[0] *= TANGENT_LOSS
            v[1] *= TANGENT_LOSS
            p[2] = BALL_R
        if p[2] <= z_enter:
            d = float(np.hypot(p[0] - START[0], p[1] - START[1]))
            if d <= min(v_eff * max(t - t_react, 0.0), 6.0):
                return p[:2].copy(), t
    return np.array(spawn[:2], float), 3.0
