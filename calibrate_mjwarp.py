"""Fit the GPU (MuJoCo) ball to the reference physics.

Bounce: a tennis ball dropped from 2.54 m must rebound 1.35 m (ITF standard, and
what the MotrixSim model does).  Drag: the horizontal speed of a 15 m/s launch must
decay to ~0.857 of it after 0.5 s (our measured DRAG_K = 0.0223 1/m).

    PYTHONNOUSERSITE=1 <ms python> calibrate_mjwarp.py
"""
from __future__ import annotations

import numpy as np

from ball_physics import BALL_R
from catch_env_gpu import CatchEnvGPU


def measure(density=None, solref=None, viscosity=None, drop=2.54):
    """Returns (rebound height after a drop, horizontal speed ratio after 0.5 s)."""
    env = CatchEnvGPU(num_envs=1, seed=0, density=density, viscosity=viscosity, solref=solref)
    import mujoco_warp as mjwarp

    def put(pos, vel):
        env._host[:] = env.d.qpos.numpy()
        env._hostv[:] = env.d.qvel.numpy()
        env._host[0, env.ball_q:env.ball_q + 3] = pos
        env._host[0, env.ball_q + 3:env.ball_q + 7] = (1.0, 0.0, 0.0, 0.0)
        env._hostv[0, env.ball_v:env.ball_v + 6] = 0.0
        env._hostv[0, env.ball_v:env.ball_v + 3] = vel
        env._push()

    # --- drop test
    put((-3.0, 0.0, BALL_R + drop), (0.0, 0.0, 0.0))
    zs = []
    n = int(3.0 / env.dt)
    for i in range(n):
        mjwarp.step(env.m, env.d)
        if i % 10 == 0:
            zs.append(float(env.d.xpos.numpy()[0, env.ball_i, 2]))
    zs = np.array(zs)
    below = np.where(zs < BALL_R + 0.02)[0]
    rebound = float(zs[below[0]:].max() - BALL_R) if len(below) else 0.0

    # --- drag test: horizontal launch, speed after 0.5 s
    put((12.0, 0.0, 3.0), (-15.0, 0.0, 0.0))
    v0 = 15.0
    for _ in range(int(0.5 / env.dt)):
        mjwarp.step(env.m, env.d)
    v1 = float(np.linalg.norm(env.d.qvel.numpy()[0, env.ball_v:env.ball_v + 2]))
    return rebound, v1 / v0


if __name__ == "__main__":
    import sys
    tc = sys.argv[1] if len(sys.argv) > 1 else "0.01"
    print(f"bounce sweep (no fluid), timeconst {tc}:")
    for damp in (0.05, 0.1, 0.2, 0.3, 0.5):
        reb, _ = measure(solref=f"{tc} {damp}")
        print(f"  solref {tc} {damp:<5} -> rebound {reb:5.2f} m   (target 1.35)", flush=True)
    if len(sys.argv) > 2:
        print("drag sweep:")
        for rho in (0.9, 1.0):
            _, ratio = measure(density=rho, solref="0.01 0.2")
            print(f"  density {rho:<4} -> v(0.5s)/v0 = {ratio:.3f}   (target 0.857)", flush=True)
