"""Live viewport demo: watch the policy chase and catch serves in real time.

Opens the MotrixSim renderer and plays episodes back to back at ~60 fps (the sim
advances 20 ms per frame, the control step), so the ball's flight and the robot's
motion are smooth.  Prints each episode's outcome and a running catch rate.

    .venv/bin/python demo_live.py                       # 16-20 m/s, full serve box
    .venv/bin/python demo_live.py --speed 12 16 --radius 1.5
    .venv/bin/python demo_live.py --episodes 10         # stop after 10 serves
    .venv/bin/python demo_live.py --policy policies/catch_bc.pt
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

import motrixsim.render as mr
from catch_env import CatchEnv
from eval_catch import load_policy, obs_mode_of
from train_catch import VMAX

VMAX_NP = np.asarray(VMAX, float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="policies/catch_bc_big.pt")
    ap.add_argument("--episodes", type=int, default=0, help="0 = run forever")
    ap.add_argument("--speed", type=float, nargs=2, default=(16.0, 20.0))
    ap.add_argument("--clearance", type=float, nargs=2, default=(0.45, 0.8))
    ap.add_argument("--radius", type=float, default=3.0)
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--follow", action="store_true", default=True,
                    help="camera follows the robot")
    ap.add_argument("--wide", action="store_true", help="fixed wide camera")
    args = ap.parse_args()

    net = load_policy(args.policy)
    mode = obs_mode_of(args.policy)
    env = CatchEnv(num_envs=1, seed=int(time.time()) % 10000, serve_speed=args.speed,
                   serve_clearance=args.clearance, radius=(args.radius, args.radius),
                   obs_mode=mode)

    caught = 0
    played = 0
    frame_dt = 1.0 / args.fps
    with mr.RenderApp(headless=False, fps=args.fps) as app:
        app.launch(env.model, batch=1)
        cam = app.system_camera
        cam.set_view(lookat=(-11.5, 0.0, 0.8), distance=9.0, elevation=-18, azimuth=-140)
        app.sync(env.data)
        while played < args.episodes or args.episodes == 0:
            t0 = time.perf_counter()
            with torch.no_grad():
                a, _, _ = net.act(torch.from_numpy(env.obs()), deterministic=True)
            _obs, _rew, done, info = env.step(a * VMAX_NP)
            if args.follow:
                p = env.robot_pose()[0][0]
                cam.set_view(lookat=(float(p[0]), float(p[1]), 0.8), distance=6.5,
                             elevation=-20, azimuth=-140)
            app.sync(env.data)
            if done[0]:
                played += 1
                got = int(info["caught"][0])
                caught += got
                print(f"serve {played}: {'CAUGHT' if got else 'missed'}   "
                      f"{caught}/{played} = {100 * caught / played:.0f}%", flush=True)
            time.sleep(max(0.0, frame_dt - (time.perf_counter() - t0)))
    print(f"done: {caught}/{played} = {100 * caught / max(played, 1):.0f}%")


if __name__ == "__main__":
    main()
