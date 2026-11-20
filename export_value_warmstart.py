"""Fit the SKRL critic to the BC policy's discounted returns and export a warm start.

PPO fine-tuning from a 96% BC policy collapses when the critic starts random: the
first GAE advantages have the wrong sign and drag the actor down (measured: 72% after
one fine-tune, 0% by the end).  This script rolls the BC policy out in the training
environment, computes returns-to-go, fits the value net to them, and writes the
value state next to the policy state of an existing warm-start checkpoint.

    source ~/data/MotrixLab/.venv/bin/activate
    python export_value_warmstart.py --policy policies/catch_ppo.pt \
        --warmstart <run>/checkpoints/warmstart_best.pt \
        --out <run>/checkpoints/warmstart_value.pt
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import torch
from torch import nn

from motrix_env_core import registry
from motrix_envs import register

GAMMA = 0.99


def load_bc_policy(path: pathlib.Path):
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    model = ck["model"]
    obs_dim, hidden = int(ck["obs_dim"]), int(ck["hidden"])
    net = nn.Sequential(
        nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 3)
    )
    for key, idx in (("trunk.0", 0), ("trunk.2", 2), ("actor", 4)):
        net[idx].weight.data.copy_(model[f"{key}.weight"])
        net[idx].bias.data.copy_(model[f"{key}.bias"])
    return net.eval(), obs_dim


def rollout(policy, obs_dim: int, num_envs: int, steps: int, seed: int,
            serve_speed=(16.0, 20.0), serve_radius: float = 1.5) -> tuple[np.ndarray, np.ndarray]:
    # the critic must be fitted on the distribution the run trains on, otherwise a
    # sparse-success band leaves it predicting the mean and PPO collapses
    env = registry.make(
        "tennis_catch",
        num_envs=num_envs,
        seed=seed,
        env_cfg_override={"serve_speed_range": tuple(serve_speed), "serve_radius": serve_radius},
    )
    env.init_state()
    obs_buf, rew_buf, done_buf = [], [], []
    for _ in range(steps):
        obs = env.state.obs.policy.numpy()
        with torch.inference_mode():
            action = policy(torch.from_numpy(obs)).clamp(-1.0, 1.0).numpy()
        state = env.step(torch.from_numpy(action))
        obs_buf.append(obs)
        rew_buf.append(state.reward.numpy())
        done_buf.append(state.done.numpy())
    obs = np.concatenate(obs_buf)                 # (steps * envs, obs_dim)
    rew = np.stack(rew_buf).astype(np.float64)    # (steps, envs)
    done = np.stack(done_buf)
    returns = np.zeros_like(rew)
    running = np.zeros(rew.shape[1])
    for t in range(rew.shape[0] - 1, -1, -1):
        running = rew[t] + GAMMA * running * (1.0 - done[t])
        returns[t] = running
    return obs, returns.reshape(-1)


def fit_value(obs: np.ndarray, targets: np.ndarray, obs_dim: int, device: str, iters: int = 30000) -> nn.Module:
    X = torch.from_numpy(obs).float().to(device)
    y = torch.from_numpy(targets).float().unsqueeze(-1).to(device)
    net = nn.Sequential(nn.Linear(obs_dim, 256), nn.Tanh(), nn.Linear(256, 256), nn.Tanh(), nn.Linear(256, 1)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for i in range(iters):
        idx = torch.randint(0, X.shape[0], (4096,), device=device)
        loss = nn.functional.mse_loss(net(X[idx]), y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.inference_mode():
        mse = float(nn.functional.mse_loss(net(X), y))
    print(f"value fit: mse {mse:.4f} (target std {float(y.std()):.3f}, range {float(y.min()):.2f}..{float(y.max()):.2f})")
    return net.cpu()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True, type=pathlib.Path)
    parser.add_argument("--warmstart", required=True, type=pathlib.Path, help="existing checkpoint with the policy state")
    parser.add_argument("--out", required=True, type=pathlib.Path)
    parser.add_argument("--envs", type=int, default=512)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--serve-speed", type=float, nargs=2, default=(16.0, 20.0))
    parser.add_argument("--serve-radius", type=float, default=1.5)
    parser.add_argument("--value-iters", type=int, default=30000)
    args = parser.parse_args()

    register()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy, obs_dim = load_bc_policy(args.policy)
    obs, targets = rollout(policy, obs_dim, args.envs, args.steps, args.seed,
                           tuple(args.serve_speed), args.serve_radius)
    print(f"rollout: {obs.shape[0]} samples, return mean {targets.mean():.3f} std {targets.std():.3f}")
    value = fit_value(obs, targets, obs_dim, device, args.value_iters)

    checkpoint = torch.load(str(args.warmstart), map_location="cpu", weights_only=False)
    checkpoint["value"] = {
        "net.0.weight": value[0].weight.data.clone(),
        "net.0.bias": value[0].bias.data.clone(),
        "net.2.weight": value[2].weight.data.clone(),
        "net.2.bias": value[2].bias.data.clone(),
        "value_layer.weight": value[4].weight.data.clone(),
        "value_layer.bias": value[4].bias.data.clone(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, str(args.out))
    print(f"wrote {args.out} (keys: {sorted(checkpoint)})")


if __name__ == "__main__":
    main()
