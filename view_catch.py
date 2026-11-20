"""Watch a catch checkpoint drive and chase serves in the viewport.

Live keys (the renderer reports them through ``app.input``):

    -   /   =      serve band 1 m/s slower / faster (takes effect on the next serve)
    r              serve again immediately
    w              toggle the follow camera / the fixed wide camera

The env runs on MotrixSim (CPU) and the policy on CUDA when available.  Each serve
prints its outcome plus the mean commanded yaw rate, which is what to watch when a
policy rotates more than the interception needs.

    python view_catch.py --fastsac \
        --ckpt <run>/checkpoints/model_0005750.pt --run-config <run>/task_config.yaml
    python view_catch.py --fastsac ... --clamp-yaw      # zero the yaw command
    python view_catch.py --ckpt policies/catch_ppo.pt --ours --tier 3
"""

from __future__ import annotations

import argparse
import pathlib
import time

import numpy as np
import torch

import motrixsim.render as mr
from eval_fastsac import load_fastsac
from eval_three_tiers import TIERS, load_policy
from motrix_env_core import registry
from motrix_envs import register

register()


def pick_device(name: str = "") -> torch.device:
    return torch.device(name) if name else torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, type=pathlib.Path)
    parser.add_argument("--fastsac", action="store_true", help="motrix.fastsac checkpoint (needs --run-config)")
    parser.add_argument("--run-config", type=pathlib.Path, default=None, help="the run's task_config.yaml")
    parser.add_argument("--ours", action="store_true", help="ActorCritic checkpoint instead of SKRL")
    parser.add_argument("--tier", type=int, default=3, choices=(1, 2, 3))
    parser.add_argument("--speed", type=float, nargs=2, default=None,
                        help="serve band override, e.g. 16 20; defaults to the tier's band")
    parser.add_argument("--clamp-yaw", action="store_true", help="zero the yaw command the policy emits")
    parser.add_argument("--episodes", type=int, default=0, help="0 runs until the window closes")
    parser.add_argument("--fps", type=int, default=60, help="frame rate; the sim advances 20 ms per frame")
    parser.add_argument("--device", default="", help="cuda / cpu; default picks CUDA when present")
    parser.add_argument("--wide", action="store_true", help="fixed wide camera instead of following")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    dev = pick_device(args.device)
    if args.fastsac:
        if args.run_config is None:
            parser.error("--fastsac needs --run-config (the run's task_config.yaml)")
        actor, normalizer = load_fastsac(args.ckpt, args.run_config)
        actor, normalizer = actor.to(dev), normalizer.to(dev)

        def act(obs_np: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                obs = torch.from_numpy(obs_np).float().to(dev)
                action, _, _ = actor(normalizer(obs, update=False))
            return action.cpu().numpy()

        kind = "fastsac"
    else:
        net, mean, var = load_policy(args.ckpt, args.ours)
        net = net.to(dev)

        def act(obs_np: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                obs = torch.from_numpy(obs_np).float().to(dev)
                x = ((obs - mean.to(dev)) / torch.sqrt(var.to(dev) + 1e-8)).clamp(-5, 5)
                action = net(x).clamp(-1.0, 1.0)
            return action.cpu().numpy()

        kind = "policy"

    _, tier_speed, radius = TIERS[args.tier - 1]
    speed = list(args.speed if args.speed is not None else tier_speed)
    env = registry.make(
        "tennis_catch",
        num_envs=1,
        seed=args.seed,
        env_cfg_override={"serve_speed_range": tuple(speed), "serve_radius": radius},
    )
    env.init_state()
    print(
        f"{kind} {args.ckpt.name} on {dev} | physics MotrixSim (CPU) | tier {args.tier} "
        f"speed {speed[0]:.0f}-{speed[1]:.0f} m/s r{radius} | yaw {'CLAMPED' if args.clamp_yaw else 'free'}",
        flush=True,
    )

    caught = played = 0
    yaw_abs_sum = 0.0
    steps_in_episode = 0
    frame_dt = 1.0 / args.fps
    wide = args.wide
    with mr.RenderApp(headless=False) as app:
        app.launch(env.sim._model, batch=1)
        cam = app.system_camera
        cam.set_view(lookat=(-11.5, 0.0, 0.8), distance=9.0, elevation=-18, azimuth=-140)
        app.sync(env.sim._data)
        while played < args.episodes or args.episodes == 0:
            t0 = time.perf_counter()
            # live controls: the speed change lands on the next serve, because the
            # reset samples the band from the environment config
            try:
                keys = app.input
            except AttributeError:    # renderer without an input device
                keys = None
            if keys is not None:
                if keys.is_key_just_pressed("=") or keys.is_key_just_pressed("]"):
                    speed = [speed[0] + 1.0, speed[1] + 1.0]
                    env._cfg.serve_speed_range = tuple(speed)
                    print(f"serve band -> {speed[0]:.0f}-{speed[1]:.0f} m/s (next serve)", flush=True)
                elif keys.is_key_just_pressed("-") or keys.is_key_just_pressed("["):
                    speed = [max(1.0, speed[0] - 1.0), max(2.0, speed[1] - 1.0)]
                    env._cfg.serve_speed_range = tuple(speed)
                    print(f"serve band -> {speed[0]:.0f}-{speed[1]:.0f} m/s (next serve)", flush=True)
                elif keys.is_key_just_pressed("r"):
                    env.init_state()
                    print("re-serve", flush=True)
                elif keys.is_key_just_pressed("w"):
                    wide = not wide
                    print(f"camera -> {'wide' if wide else 'follow'}", flush=True)

            action = act(env.state.obs.policy.numpy())
            if args.clamp_yaw:
                action = action.copy()
                action[..., 2] = 0.0
            yaw_abs_sum += float(np.abs(action[..., 2]).mean())
            steps_in_episode += 1
            state = env.step(torch.from_numpy(action))
            finished = state.done.numpy()
            if finished.any():
                played += 1
                got = int(state.metrics["catch_rate"].numpy()[finished][0] > 0)
                caught += got
                mean_yaw = yaw_abs_sum / max(steps_in_episode, 1)
                yaw_abs_sum = 0.0
                steps_in_episode = 0
                print(
                    f"serve {played}: {'CAUGHT' if got else 'missed'}   "
                    f"{caught}/{played} = {100 * caught / played:.0f}%   band {speed[0]:.0f}-{speed[1]:.0f}   "
                    f"mean|wz| {mean_yaw:.2f}",
                    flush=True,
                )
            if not wide:
                # the base position is the first three dof positions of the scene
                p = np.array(env.sim._data.dof_pos[0, :3], dtype=np.float64)
                cam.set_view(lookat=(float(p[0]), float(p[1]), 0.8), distance=6.5, elevation=-20, azimuth=-140)
            app.sync(env.sim._data)
            time.sleep(max(0.0, frame_dt - (time.perf_counter() - t0)))
    print(f"done: {caught}/{played} = {100 * caught / max(played, 1):.0f}%")


if __name__ == "__main__":
    main()
