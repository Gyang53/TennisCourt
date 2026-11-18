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
from teacher_oracle import record
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


def teacher_action_for(env, tx, ty, tc, lead=0.6, kv=0.8):
    """P on the recorded catch point with an inner velocity loop (mecanum
    strafing otherwise leaves a ~0.24 m steady-state error)."""
    p, q = env.robot_pose()
    yaw = env.yaw_of(q)
    vw = env.model.get_link_velocities(env.data)[:, 0, :2]
    err = np.array([[tx, ty]]) - p[:, :2]
    g = np.where((env.t < tc - lead)[:, None], 6.0, 9.0)
    v_ref = np.clip(g * err, -4.0, 4.0)
    v_cmd = np.clip(v_ref + kv * (v_ref - vw), -4.5, 4.5)
    c, s = np.cos(yaw), np.sin(yaw)
    fwd = np.sum(v_cmd * np.stack([s, -c], axis=1), axis=1)
    left = np.sum(v_cmd * np.stack([c, s], axis=1), axis=1)
    raw = np.stack([fwd, left, np.zeros(len(fwd))], axis=1)
    return np.clip(raw / VMAX_NP, -1.0, 1.0)


def collect_batched(env, serves, obs_mode="both", steps=240):
    """Two batched passes: record each ball's first reachable entry into the cage
    window, then replay the same serves with the teacher while recording demos."""
    from catch_env import sample_serve  # noqa: F401  (kept for callers)

    sp = np.array([x[0] for x in serves])
    v0 = np.array([x[1] for x in serves])
    # set_serves computes the analytic catch target, which is both accurate
    # (0.05 m against the engine) and computable from the observation
    env.set_serves(sp, v0, park=False)
    found = np.ones(env.n, dtype=bool)
    obs_buf, act_buf = [], []
    active = np.ones(env.n, dtype=bool)
    for _ in range(steps):
        if not active.any():
            break
        a = env.teacher_action()
        obs = env.obs()
        obs_buf.append(obs[active].copy())
        act_buf.append(a[active].astype(np.float32))
        obs, _r, done, _i = env.step(a * VMAX_NP)
        active &= ~done
    print(f"collected {sum(len(x) for x in obs_buf)} transitions from {int(found.sum())} recorded serves", flush=True)
    return np.concatenate(obs_buf), np.concatenate(act_buf)


def collect_oracle(episodes, seed0, sp, cl, radius, obs_mode="both"):
    """Two-pass demos: record the first reachable catch point in a parked pass,
    then replay the same serve with the teacher and record (obs, action)."""
    obs_buf, act_buf = [], []
    kept = 0
    for k in range(episodes):
        seed = seed0 + k
        tgt = record(seed, sp, cl, radius)
        if tgt is None:
            continue
        tx, ty, tc = tgt
        env = CatchEnv(num_envs=1, seed=seed, serve_speed=sp, serve_clearance=cl,
                       radius=(radius, radius), obs_mode=obs_mode)
        for _ in range(240):
            a = teacher_action_for(env, tx, ty, tc)
            obs = env.obs()
            obs, _r, done, _i = env.step(a * VMAX_NP)
            obs_buf.append(obs[0])
            act_buf.append(a[0].astype(np.float32))
            if done[0]:
                break
        kept += 1
    print(f"collected from {kept}/{episodes} serves", flush=True)
    return np.stack(obs_buf), np.stack(act_buf)


def collect(env, episodes, phases=(0, 1, 2, 3, 4)):
    obs_buf, act_buf = [], []
    ep = 0
    phase = 0
    while ep < episodes:
        apply_phase(env, phases[min(phase, len(phases) - 1)])
        obs = env.obs()
        for _ in range(240):
            a = env.teacher_action()
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
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--oracle", action="store_true", help="two-pass oracle demos")
    ap.add_argument("--batched", action="store_true", help="batched two-pass demos")
    ap.add_argument("--rounds", type=int, default=6, help="serve sets to collect")
    ap.add_argument("--speed", type=float, nargs=2, default=(12.0, 16.0))
    ap.add_argument("--radius", type=float, default=1.5)
    ap.add_argument("--out", default="policies/catch_bc.pt")
    args = ap.parse_args()

    if args.batched:
        from catch_env import sample_serve
        rng = np.random.default_rng(args.seed)
        env = CatchEnv(num_envs=args.envs, seed=args.seed, serve_speed=args.speed,
                       serve_clearance=(0.6, 1.0), radius=(args.radius, args.radius))
        O, A = [], []
        for _ in range(args.rounds):
            serves = [sample_serve(rng, args.speed, (0.6, 1.0), (-4.9, 0.0),
                                   (args.radius, args.radius)) for _ in range(env.n)]
            o, a = collect_batched(env, serves)
            O.append(o)
            A.append(a)
        obs, act = np.concatenate(O), np.concatenate(A)
    elif args.oracle:
        obs, act = collect_oracle(args.episodes, 600, args.speed, (0.6, 1.0), args.radius)
    else:
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
