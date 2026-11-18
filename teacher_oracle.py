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

def record(seed, sp, cl, radius, steps=240):
    """Run one episode with the robot parked away and record the true post-bounce
    crossing of CATCH_Z (the ball never touches the robot before that, so the
    record is valid for a replay of the same serve)."""
    env = CatchEnv(num_envs=1, seed=seed, serve_speed=sp, serve_clearance=cl, radius=(radius, radius))
    prev = env.prev_bz.copy(); bounced = np.zeros(1, bool)
    for step in range(steps):
        pos = env.data.dof_pos.copy(); pos[0, 0] = -2.0; pos[0, 1] = -8.0
        env.data.set_dof_pos(pos, env.model); env.model.forward_kinematic(env.data)
        obs, rew, done, info = env.step(np.zeros((1, 3)))
        b = env.ball_pose(); v = env.ball_vel()
        bounced |= b[:, 2] < BALL_R + 0.02
        cross = bounced & (prev > CATCH_Z) & (b[:, 2] <= CATCH_Z) & (v[:, 2] < 0)
        if cross[0]:
            return b[0, 0], b[0, 1], float(env.t[0])
        bounced &= ~done; prev = b[:, 2].copy()
    return None

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
