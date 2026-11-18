"""Two-pass teacher oracle.

The analytic forward model does not match the engine's contact solver: the
predicted post-bounce crossing was 0.8 s and 4.4 m off, so a teacher built on it
aims at the wrong place.  Instead, run the serve once with the robot parked away
(the ball never touches the robot before the crossing, so the record is valid),
then replay the same seeded serve with a teacher that drives to the recorded
point at the recorded time.
"""
import numpy as np
from catch_env import CatchEnv, CATCH_Z, BALL_R

def record(seed, sp, cl, radius, steps=240, z_enter=0.52, v_eff=3.6, t_react=0.2):
    """Run one episode with the robot parked away and record where/when the ball
    first enters the cage window (world z 0.10-0.52 m).

    Any bounce counts, but only the first entry the robot can actually reach from
    its start (distance <= v_eff * (t - t_react)): the earliest entries happen
    mid-court, ~8 m away and ~1 s away, so they are not catchable.  The ball never
    touches the robot before that instant, so the record is valid for a replay of
    the same (seeded) serve.
    """
    env = CatchEnv(num_envs=1, seed=seed, serve_speed=sp, serve_clearance=cl, radius=(radius, radius))
    for step in range(steps):
        pos = env.data.dof_pos.copy(); pos[0, 0] = -2.0; pos[0, 1] = -8.0
        env.data.set_dof_pos(pos, env.model); env.model.forward_kinematic(env.data)
        obs, rew, done, info = env.step(np.zeros((1, 3)))
        b = env.ball_pose()
        if b[0, 2] <= z_enter:
            d = float(np.hypot(b[0, 0] + 11.885, b[0, 1]))
            if d <= v_eff * max(float(env.t[0]) - t_react, 0.0):
                return b[0, 0], b[0, 1], float(env.t[0])
        if done[0]:
            return None
    return None


def teacher_velocity_feedback(seed, sp, cl, radius, lead=0.6, kv=0.8, kp=5.0, steps=240):
    """Teacher with an inner velocity loop: the mecanum lateral slip leaves a
    ~0.24 m steady-state error with pure position feedback, so the commanded
    velocity is corrected by the measured velocity error."""
    tgt = record(seed, sp, cl, radius)
    if tgt is None:
        return None
    tx, ty, tc = tgt
    env = CatchEnv(num_envs=1, seed=seed, serve_speed=sp, serve_clearance=cl, radius=(radius, radius))
    got = 0
    for step in range(steps):
        p, q = env.robot_pose(); yaw = env.yaw_of(q)
        vw = env.model.get_link_velocities(env.data)[:, 0, :2]
        err = (np.array([tx, ty]) - p[0, :2]).astype(float)
        g = 6.0 if env.t[0] < tc - lead else 9.0
        v_ref = np.clip(g * err, -4.0, 4.0)
        v_cmd = np.clip(v_ref + kv * (v_ref - vw[0]), -4.5, 4.5)
        c, s_ = np.cos(yaw[0]), np.sin(yaw[0])
        fwd = v_cmd[0] * s_ - v_cmd[1] * c
        left = v_cmd[0] * c + v_cmd[1] * s_
        obs, rew, done, info = env.step(np.array([[fwd, left, 0.0]]))
        if done[0]:
            got = int(info["caught"][0]); break
    return got


def teacher_rate(seed0, n, sp, cl, radius, lead=0.10, gain=6.0, damp=3.0, steps=240):
    """Replay the same seeds, drive to the recorded point, count catches."""
    caught = 0
    for k in range(n):
        seed = seed0 + k
        tgt = record(seed, sp, cl, radius)
        if tgt is None:
            continue
        tx, ty, tc = tgt
        env = CatchEnv(num_envs=1, seed=seed, serve_speed=sp, serve_clearance=cl, radius=(radius, radius))
        got = 0
        for step in range(steps):
            p, q = env.robot_pose(); yaw = env.yaw_of(q)
            vw = env.model.get_link_velocities(env.data)[:, 0, :2]
            err = np.array([[tx, ty]]) - p[:, :2]
            dist = np.linalg.norm(err, axis=1)
            arrive = tc - lead
            urgency = np.clip(1.5 * (arrive - env.t[0]) / max(arrive, 0.2), 0.4, 1.6)
            v_des = np.clip(gain * urgency * err - damp * vw, -4.5, 4.5)
            c, s_ = np.cos(yaw), np.sin(yaw)
            fwd = np.sum(v_des * np.stack([s_, -c], axis=1), axis=1)
            left = np.sum(v_des * np.stack([c, s_], axis=1), axis=1)
            obs, rew, done, info = env.step(np.stack([fwd, left, np.zeros(1)], axis=1))
            if done[0]:
                got = int(info["caught"][0]); break
        caught += got
    return caught, n
