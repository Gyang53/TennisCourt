"""BC warm start for MotrixLab's PPO: distil the scripted teacher into its policy.

PPO from scratch never caught a ball in 12.3M environment steps (the training
curve's catch reward stayed at 0.0 throughout), which is the same wall the
hand-written PPO hit: the agent has to be under a 0.2 m window at the right
instant and only then does it get paid.  The scripted teacher does solve that, so
the policy starts from it.

The teacher's observation is the 12-dim state the MotrixLab env emits, and the
network is the shape SKRL builds (12 -> 128 -> 128 -> 3, ELU), so the weights drop
straight into a checkpoint that ``mxlab train ... resume=<path>`` loads:

    python bc_warmstart.py --template <run>/checkpoints/best_agent.pt \
        --out <run>/checkpoints/warmstart.pt --serves 1536
    mxlab train task=tennis_catch/skrl.ppo sim=mujoco_warp num_envs=4096 \
        resume=<run>/checkpoints/warmstart.pt
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import torch
import torch.nn as nn

from ball_physics import sample_serve
from bc_train import VMAX_NP, collect_batched
from catch_env import CatchEnv

OBS_DIM, ACT_DIM, WIDTH = 12, 3, 128


def build_net() -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(OBS_DIM, WIDTH),
        nn.ELU(),
        nn.Linear(WIDTH, WIDTH),
        nn.ELU(),
        nn.Linear(WIDTH, ACT_DIM),
    )


def train_bc(obs: np.ndarray, act: np.ndarray, epochs: int, batch: int, device: str):
    net = build_net().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    x = torch.from_numpy(obs).to(device)
    y = torch.from_numpy(act).to(device)
    n = len(x)
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        total = 0.0
        for start in range(0, n, batch):
            idx = perm[start:start + batch]
            loss = nn.functional.mse_loss(net(x[idx]), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * len(idx)
        if (epoch + 1) % 100 == 0:
            print(f"epoch {epoch + 1}/{epochs} mse {total / n:.5f}", flush=True)
    return net.eval()


def bc_catch_rate(net, num_envs: int) -> float:
    """Teacher-free check: does the distilled policy catch in the reference sim?"""
    device = next(net.parameters()).device
    env = CatchEnv(num_envs=num_envs, seed=11, serve_speed=(16.0, 20.0), radius=(1.5, 1.0), obs_mode="state")
    caught = done_total = 0
    for _ in range(8):
        obs = env.obs()
        for _ in range(240):
            with torch.inference_mode():
                action = net(torch.from_numpy(obs).to(device)).cpu().numpy().clip(-1.0, 1.0)
            obs, _, done, info = env.step(action * VMAX_NP)
            if done.any():
                caught += int(info["caught"][done].sum())
                done_total += int(done.sum())
    env.close()
    return caught / max(done_total, 1)


def export(net, obs: np.ndarray, template: pathlib.Path, out: pathlib.Path) -> None:
    """Write the BC weights into a copy of an SKRL checkpoint."""
    ck = torch.load(str(template), map_location="cpu", weights_only=False)
    state = ck["policy"]
    modules = [net[0], net[2], net[4]]
    for k, module in ((0, modules[0]), (2, modules[1])):
        state[f"net.{k}.weight"] = module.weight.detach().cpu().clone()
        state[f"net.{k}.bias"] = module.bias.detach().cpu().clone()
    state["mean_layer.weight"] = modules[2].weight.detach().cpu().clone()
    state["mean_layer.bias"] = modules[2].bias.detach().cpu().clone()
    # the normalizer must match the data the policy was fitted on: fresh training
    # runs start from the checkpoint's old statistics otherwise
    normalizer = ck["observation_preprocessor"]
    normalizer["running_mean"] = torch.tensor(obs.mean(0), dtype=torch.float64)
    normalizer["running_variance"] = torch.tensor(obs.var(0), dtype=torch.float64)
    normalizer["current_count"] = torch.tensor(len(obs), dtype=torch.float64)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, str(out))
    print(f"wrote {out} (template {template.name}, policy + normalizer replaced)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", required=True, type=pathlib.Path, help="checkpoint to patch")
    parser.add_argument("--out", required=True, type=pathlib.Path)
    parser.add_argument("--serves", type=int, default=1536)
    parser.add_argument("--envs", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch", type=int, default=1024)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)
    env = CatchEnv(num_envs=args.envs, seed=0, serve_speed=(16.0, 20.0), radius=(1.5, 1.0), obs_mode="state")
    obs_buf, act_buf = [], []
    batches = max(1, args.serves // args.envs)
    for i in range(batches):
        serves = [sample_serve(rng, (16.0, 20.0), (0.6, 1.0), (-4.9, 0.0), (1.5, 1.0)) for _ in range(args.envs)]
        obs, act = collect_batched(env, serves, steps=240)
        obs_buf.append(obs)
        act_buf.append(act)
    env.close()
    obs = np.concatenate(obs_buf).astype(np.float32)
    act = np.concatenate(act_buf).astype(np.float32)
    print(f"dataset: {obs.shape[0]} transitions, obs {obs.shape[1:]} act {act.shape[1:]}")

    net = train_bc(obs, act, args.epochs, args.batch, device)
    rate = bc_catch_rate(net, num_envs=128)
    print(f"BC policy catch rate in the reference sim: {rate:.0%}")
    export(net, obs, args.template, args.out)


if __name__ == "__main__":
    main()
