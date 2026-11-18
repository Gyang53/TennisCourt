"""DAgger: distil the catch teacher on the states the policy actually visits.

Plain behaviour cloning failed here (teacher 92/33/8%, clone 0%): the teacher's
demos are few and its actions aggressive, and the clone drifts into states the
demos never covered.  DAgger fixes that by rolling out the *student* and asking
the teacher for the action at every visited state, aggregating the dataset over
iterations.

    .venv/bin/python dagger_train.py --iters 5 --rounds 8 --envs 32
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from catch_env import CatchEnv, sample_serve
from train_catch import VMAX, ActorCritic

VMAX_NP = np.asarray(VMAX, float)


def evaluate(net, cases, n=24, steps=240, seed=21):
    """Catch rate on a held-out set of serves per distribution."""
    out = []
    for radius, sp, cl in cases:
        env = CatchEnv(num_envs=n, seed=seed, serve_speed=sp, serve_clearance=cl,
                       radius=radius, obs_mode="both")
        caught = np.zeros(n)
        for _ in range(steps):
            with torch.no_grad():
                a, _, _ = net.act(torch.from_numpy(env.obs()), deterministic=True)
            obs, _r, done, info = env.step(a * VMAX_NP)
            caught = np.where(done, info["caught"], caught)
        out.append(float(caught.mean()))
    return out


def rollout(env, net, serves, noise=0.15, steps=240):
    """Student acts, teacher labels. Returns (obs, teacher action) pairs."""
    sp = np.array([x[0] for x in serves])
    v0 = np.array([x[1] for x in serves])
    # the analytic target computed by set_serves is both accurate (0.05 m against
    # the engine) and computable from the observation, so the teacher's labels are
    # a function of the student's input - without that DAgger regresses on noise
    env.set_serves(sp, v0, park=False)
    obs_buf, act_buf = [], []
    active = np.ones(env.n, dtype=bool)
    for _ in range(steps):
        if not active.any():
            break
        obs = env.obs()
        label = env.teacher_action()                      # what the teacher would do
        with torch.no_grad():
            a_pi, _, _ = net.act(torch.from_numpy(obs))
        obs_buf.append(obs[active].copy())
        act_buf.append(label[active].astype(np.float32))
        act = np.clip(a_pi + noise * np.random.randn(*a_pi.shape), -1.0, 1.0)
        obs, _r, done, _i = env.step(act * VMAX_NP)       # the student drives
        active &= ~done
    return np.concatenate(obs_buf), np.concatenate(act_buf)


def fit(net, obs, act, epochs, batch=512, lr=3e-4):
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    O, A = torch.from_numpy(obs), torch.from_numpy(act)
    n = O.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n)
        for start in range(0, n, batch):
            idx = perm[start:start + batch]
            mean, _ = net(O[idx])
            loss = nn.functional.mse_loss(mean, A[idx])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
    return float(loss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=8, help="serve sets per iteration")
    ap.add_argument("--envs", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=30, help="per iteration")
    ap.add_argument("--radius", type=float, default=1.5, help="training serve spread")
    ap.add_argument("--speed", type=float, nargs=2, default=(12.0, 16.0))
    ap.add_argument("--init", default="policies/catch_bc_two_pass.pt")
    ap.add_argument("--out", default="policies/catch_dagger.pt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    env = CatchEnv(num_envs=args.envs, seed=args.seed, serve_speed=args.speed,
                   serve_clearance=(0.6, 1.0),
                   radius=(args.radius, args.radius), obs_mode="both")
    net = ActorCritic(obs_dim=61, hidden=256)
    if Path(args.init).exists():
        ckpt = torch.load(args.init, map_location="cpu")
        if ckpt["obs_dim"] == 61:
            net.load_state_dict(ckpt["model"])
            print(f"initialised from {args.init}", flush=True)

    cases = [((0.5, 0.35), (12.0, 16.0), (0.6, 1.0)),
             ((1.5, 1.05), (12.0, 16.0), (0.6, 1.0)),
             ((3.0, 3.0), (16.0, 20.0), (0.45, 0.8))]
    O, A = [], []
    for it in range(1, args.iters + 1):
        for _ in range(args.rounds):
            serves = [sample_serve(rng, args.speed, (0.6, 1.0), (-4.9, 0.0),
                                   (args.radius, args.radius)) for _ in range(env.n)]
            o, a = rollout(env, net, serves)
            O.append(o)
            A.append(a)
        obs, act = np.concatenate(O), np.concatenate(A)
        loss = fit(net, obs, act, args.epochs)
        rates = evaluate(net, cases)
        print(f"iter {it}: dataset {len(obs)/1e3:5.1f}k  fit {loss:.5f}  "
              f"12-16 r0.5 {100*rates[0]:.0f}%  r1.5 {100*rates[1]:.0f}%  "
              f"16-20 {100*rates[2]:.0f}%", flush=True)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": net.state_dict(), "obs_dim": 61, "hidden": 256}, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
