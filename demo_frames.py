"""Capture a catch as a frame sequence (headless) so the motion can be inspected.

Runs one episode with the shipped policy in the reference environment and saves a
frame every few control steps, using MotrixSim's offscreen render targets.  The
live window version of the same loop is demo_live.py.

    .venv/bin/python demo_frames.py --every 20 --out frames
"""

from __future__ import annotations

import argparse
import pathlib
import time

import numpy as np
import torch

import motrixsim.render as mr
from catch_env import CatchEnv
from eval_catch import load_policy, obs_mode_of
from train_catch import VMAX

VMAX_NP = np.asarray(VMAX, float)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="policies/catch_bc_big.pt")
    ap.add_argument("--speed", type=float, nargs=2, default=(14.0, 18.0))
    ap.add_argument("--radius", type=float, default=2.0)
    ap.add_argument("--every", type=int, default=20, help="control steps between frames")
    ap.add_argument("--out", default="frames")
    ap.add_argument("--size", type=int, nargs=2, default=(720, 405))
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(exist_ok=True)
    net = load_policy(args.policy)
    mode = obs_mode_of(args.policy)
    env = CatchEnv(
        num_envs=1,
        seed=int(time.time()) % 10000,
        serve_speed=args.speed,
        serve_clearance=(0.6, 1.0),
        radius=(args.radius, args.radius),
        obs_mode=mode,
    )

    with mr.RenderApp(headless=True) as app:
        app.launch(env.model, batch=1)
        env.model.cameras.set_system_render_target("image", args.size[0], args.size[1])
        cam = app.system_camera
        cam.set_view(lookat=(-11.5, 0.0, 0.9), distance=6.0, elevation=-16, azimuth=-135)
        app.sync(env.data)

        saved = 0
        for step in range(240):
            with torch.no_grad():
                action, _, _ = net.act(torch.from_numpy(env.obs()), deterministic=True)
            _obs, _rew, done, info = env.step(action * VMAX_NP)
            if step % args.every == 0:
                task = cam.capture()
                for _ in range(200):
                    app.sync(env.data)
                    img = task.take_image()
                    if img is not None:
                        name = out / f"frame_{step:03d}.png"
                        img.save_to_disk(str(name))
                        print(f"saved {name} t={step * 0.02:.2f}s", flush=True)
                        saved += 1
                        break
                    time.sleep(0.02)
            else:
                app.sync(env.data)
            if done[0]:
                print(f"episode done at t={step * 0.02:.2f}s caught={int(info['caught'][0])}")
                break
    print(f"{saved} frames in {out}")


if __name__ == "__main__":
    main()
