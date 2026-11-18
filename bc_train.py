"""Behaviour cloning for the catch task.

PPO from scratch did not find this task (the catch window is ~0.2 m and it only
pays out once per episode).  A scripted teacher - PD tracking of the
forward-simulated catch point - catches 78% of the easy serves, so the policy is
distilled from it and can then be fine-tuned with train_catch.py --init.

    .venv/bin/python bc_train.py --episodes 400
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from catch_env import CatchEnv
from train_catch import VMAX, ActorCritic, apply_phase

VMAX_NP = np.asarray(VMAX, dtype=float)


def teacher_action(env):
    """PD on the predicted catch point, in normalised vehicle commands."""
    p, q = env.robot_pose()
    yaw = env.yaw_of(q)
    vw = env.model.get_link_velocities(env.data)[:, 0, :2]
    tgt, _ = env.intercept()
    err = tgt - p[:, :2]
    v_des = np.clip(1.5 * err - 1.1 * vw, -VMAX_NP[0], VMAX_NP[1])
    c, s = np.cos(yaw), np.sin(yaw)
    fwd = np.sum(v_des * np.stack([s, -c], axis=1), axis=1)
    left = np.sum(v_des * np.stack([c, s], axis=1), axis=1)
    raw = np.stack([fwd, left, np.zeros(len(fwd))], axis=1)
    return np.clip(raw / VMAX_NP, -1.0, 1.0)


def collect(env, episodes, phases=(0, 1, 2, 3, 4)):
    obs_buf, act_buf = [], []
    ep = 0
    phase = 0
    while ep < episodes:
        apply_phase(env, phases[min(phase, len(phases) - 1)])
        obs = env.obs()
        for _ in range(240):
            a = teacher_action(env)
            obs_buf.append(obs)
            act_buf.append(a.astype(np.float32))
            obs, _rew, done, _info = env.step(a * VMAX_NP)
            ep += int(done.sum())
        # widen the aim radius as data accumulates
        if ep > episodes * (phase + 1) / len(phases):
            phase += 1
    return np.concatenate(obs_buf), np.concatenate(act_buf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--envs", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--out", default="policies/catch_bc.pt")
    args = ap.parse_args()

    env = CatchEnv(num_envs=args.envs, seed=1)
    obs, act = collect(env, args.episodes)
    print(f"dataset: {obs.shape[0]} transitions", flush=True)

    net = ActorCritic(hidden=args.hidden)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    O, A = torch.from_numpy(obs), torch.from_numpy(act)
    n = O.shape[0]
    for epoch in range(args.epochs):
        perm = torch.randperm(n)
        total = 0.0
        for start in range(0, n, args.batch):
            idx = perm[start:start + args.batch]
            mean, _ = net(O[idx])
            loss = nn.functional.mse_loss(mean, A[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * len(idx)
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch:3d}  mse {total / n:.5f}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": net.state_dict(), "obs_dim": int(O.shape[1]),
                    "hidden": net.trunk[0].out_features}, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
