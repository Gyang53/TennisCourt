"""Watch a MotrixLab catch checkpoint drive and chase serves in the viewport.

The env runs on MotrixSim (CPU) and the policy on CUDA when available; the camera
follows the robot so the serve, the chase and the ball's flight stay in frame.

    python view_catch.py --ckpt <run>/checkpoints/agent_300.pt
    python view_catch.py --ckpt policies/catch_ppo.pt --ours --tier 3
    python view_catch.py --ckpt ... --episodes 5 --wide      # fixed camera, 5 serves

Tiers match eval_three_tiers: 1 = 12-16 m/s r=0.5, 2 = 12-16 m/s r=1.5,
3 = 16-20 m/s full range.  Prints each serve's outcome and the running rate.
"""

from __future__ import annotations

import argparse
import pathlib
import time

import numpy as np
import torch

import motrixsim.render as mr
from eval_three_tiers import TIERS, load_policy
from motrix_env_core import registry
from motrix_envs import register

register()


def pick_device(name: str = "") -> torch.device:
    return torch.device(name) if name else torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, type=pathlib.Path)
    parser.add_argument("--ours", action="store_true", help="ActorCritic checkpoint instead of SKRL")
    parser.add_argument("--tier", type=int, default=3, choices=(1, 2, 3))
    parser.add_argument("--episodes", type=int, default=0, help="0 runs until the window closes")
    parser.add_argument("--fps", type=int, default=60, help="frame rate; the sim advances 20 ms per frame")
    parser.add_argument("--device", default="", help="cuda / cpu; default picks CUDA when present")
    parser.add_argument("--wide", action="store_true", help="fixed wide camera instead of following")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    label, speed, radius = TIERS[args.tier - 1]
    net, mean, var = load_policy(args.ckpt, args.ours)
    dev = pick_device(args.device)
    net = net.to(dev)
    env = registry.make(
        "tennis_catch",
        num_envs=1,
        seed=args.seed,
        env_cfg_override={"serve_speed_range": speed, "serve_radius": radius},
    )
    env.init_state()
    print(f"policy {args.ckpt.name} on {dev} | physics MotrixSim (CPU) | tier {args.tier}: {label}", flush=True)

    caught = played = 0
    frame_dt = 1.0 / args.fps
    with mr.RenderApp(headless=False) as app:
        app.launch(env.sim._model, batch=1)
        cam = app.system_camera
        cam.set_view(lookat=(-11.5, 0.0, 0.8), distance=9.0, elevation=-18, azimuth=-140)
        app.sync(env.sim._data)
        while played < args.episodes or args.episodes == 0:
            t0 = time.perf_counter()
            with torch.no_grad():
                obs = torch.from_numpy(env.state.obs.policy.numpy()).to(dev)
                x = ((obs - mean.to(dev)) / torch.sqrt(var.to(dev) + 1e-8)).clamp(-5, 5)
                action = net(x).clamp(-1.0, 1.0).cpu().numpy()
            state = env.step(torch.from_numpy(action))
            finished = state.done.numpy()
            if finished.any():
                played += 1
                got = int(state.metrics["catch_rate"].numpy()[finished][0] > 0)
                caught += got
                print(f"serve {played}: {'CAUGHT' if got else 'missed'}   {caught}/{played} = {100 * caught / played:.0f}%", flush=True)
            if not args.wide:
                # the base position is the first three dof positions of the scene
                p = np.array(env.sim._data.dof_pos[0, :3], dtype=np.float64)
                cam.set_view(lookat=(float(p[0]), float(p[1]), 0.8), distance=6.5, elevation=-20, azimuth=-140)
            app.sync(env.sim._data)
            time.sleep(max(0.0, frame_dt - (time.perf_counter() - t0)))
    print(f"done: {caught}/{played} = {100 * caught / max(played, 1):.0f}%")


if __name__ == "__main__":
    main()
