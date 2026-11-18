"""Evaluate a trained catch policy.

Runs episodes with the *exact* serve planner used by the ROS bridge (legal
serve, requested speed and net clearance) rather than the training sampler.

    .venv/bin/python eval_catch.py --episodes 48
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from ball_physics import plan_serve
from catch_env import OBS_DIM, OBS_MODE_FOR_DIM, CatchEnv
from train_catch import VMAX, ActorCritic, pick_device


def exact_serves(n, rng, speed, clearance):
    """(spawn, v0) pairs from the exact legal-serve planner."""
    out = []
    while len(out) < n:
        side = 1.0 if rng.random() < 0.5 else -1.0
        spawn = np.array(
            [
                11.885 + rng.uniform(0.2, 1.4),
                side * rng.uniform(0.3, 3.8),
                rng.uniform(2.0, 2.6),
            ]
        )
        v0, _, _ = plan_serve(rng, spawn, rng.uniform(*speed), rng.uniform(*clearance))
        if v0 is not None:
            out.append((spawn, v0))
    return out


def obs_mode_of(path):
    """The observation mode a checkpoint was trained with (from its input size)."""
    ckpt = torch.load(path, map_location="cpu")
    return OBS_MODE_FOR_DIM.get(int(ckpt.get("obs_dim", OBS_DIM)), "both")


def load_policy(path) -> ActorCritic:
    ckpt = torch.load(path, map_location="cpu")
    net = ActorCritic(
        obs_dim=ckpt.get("obs_dim", OBS_DIM), hidden=ckpt.get("hidden", 128)
    )
    net.load_state_dict(ckpt["model"])
    net.eval()
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="policies/catch_ppo.pt")
    ap.add_argument("--episodes", type=int, default=48)
    ap.add_argument("--speed", type=float, nargs=2, default=(16.0, 20.0))
    ap.add_argument("--clearance", type=float, nargs=2, default=(0.45, 0.8))
    ap.add_argument("--render", action="store_true", help="render one episode to PNGs")
    args = ap.parse_args()

    if args.render:
        render(args.policy)
        return
    net = load_policy(args.policy)
    rng = np.random.default_rng(7)
    serves = exact_serves(args.episodes, rng, args.speed, args.clearance)
    env = CatchEnv(num_envs=args.episodes, seed=0, obs_mode=obs_mode_of(args.policy))
    for e, (spawn, v0) in enumerate(serves):
        p = env.data.dof_pos.copy()
        v = env.data.dof_vel.copy()
        p[e, env.ball_p[:3]] = spawn
        p[e, env.ball_p[3:]] = (1.0, 0.0, 0.0, 0.0)
        v[e, env.ball_v] = np.concatenate([v0, (0.0, 0.0, 0.0)])
        env.data.set_dof_pos(p, env.model)
        env.data.set_dof_vel(v)
    env.model.forward_kinematic(env.data)
    env.t[:] = 0.0
    env.caught[:] = 0.0
    env.done[:] = False

    caught = np.zeros(args.episodes)
    for step in range(260):
        obs = env.obs()
        with torch.no_grad():
            a, _, _ = net.act(torch.from_numpy(obs), deterministic=True)
        obs, _rew, done, info = env.step(a * VMAX)
        caught = np.where(done, info["caught"], caught)
        if done.all():
            break
    n = min(args.episodes, len(caught))
    print(
        f"episodes {n}  caught {int(caught.sum())}  catch rate {100 * caught.mean():.0f}%"
    )


def render(policy_path, seed=0):
    """Run one episode in the viewer and save frames around the catch."""
    import time

    import motrixsim.render as mr

    from ball_physics import COURT_X
    from catch_env import START

    net = load_policy(policy_path)
    rng = np.random.default_rng(seed)
    spawn, v0 = exact_serves(1, rng, (16.0, 20.0), (0.45, 0.8))[0]
    env = CatchEnv(num_envs=1, seed=0, obs_mode=obs_mode_of(policy_path))
    p = env.data.dof_pos.copy()
    v = env.data.dof_vel.copy()
    p[0, env.ball_p[:3]] = spawn
    p[0, env.ball_p[3:]] = (1.0, 0.0, 0.0, 0.0)
    v[0, env.ball_v] = np.concatenate([v0, (0.0, 0.0, 0.0)])
    env.data.set_dof_pos(p, env.model)
    env.data.set_dof_vel(v)
    env.model.forward_kinematic(env.data)
    env.t[:] = 0.0
    env.caught[:] = 0.0
    env.done[:] = False
    with mr.RenderApp(headless=False, fps=60) as app:
        app.launch(env.model, batch=1)
        cam = app.system_camera
        cam.set_view(
            lookat=(START[0] + 2.0, 0.0, 0.6), distance=7.0, elevation=-14, azimuth=-150
        )
        for step in range(240):
            obs = env.obs()
            with torch.no_grad():
                a, _, _ = net.act(torch.from_numpy(obs), deterministic=True)
            env.step(a * VMAX)
            app.sync(env.data)
            if step % 8 == 0:
                task = cam.capture()
                for _ in range(20):
                    app.sync(env.data)
                    try:
                        img = task.take_image()
                    except Exception:
                        img = None
                    if img is not None:
                        img.save_to_disk(
                            f"/home/yangg/data/tennis_robot/catch_{step:03d}.png"
                        )
                        break
                    time.sleep(1 / 60)
            if env.caught[0] or env.done[0]:
                print(f"step {step}: caught={int(env.caught[0])} t={env.t[0]:.2f}s")
                break
        task = cam.capture()
        for _ in range(40):
            app.sync(env.data)
            try:
                img = task.take_image()
            except Exception:
                img = None
            if img is not None:
                img.save_to_disk("/home/yangg/data/tennis_robot/catch_final.png")
                print("saved catch_final.png")
                break
            time.sleep(1 / 60)


if __name__ == "__main__":
    main()
