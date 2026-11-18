"""Three-tier evaluation in the training environment itself.

Runs an SKRL checkpoint (or one of our ActorCritic policies) through the MotrixLab
catch environment on the MotrixSim backend, at the three serve tiers the shipped
policy is quoted on.  Using the same environment as training keeps the number
honest — the earlier evaluator drove the reference environment and got the action
scaling and the 61-dim observation wrong.

    source ~/data/MotrixLab/.venv/bin/activate
    python eval_three_tiers.py --ckpt <run>/checkpoints/best_agent.pt
    python eval_three_tiers.py --ckpt policies/catch_bc_big.pt --ours
"""

from __future__ import annotations

import argparse
import pathlib

import torch
from torch import nn

from motrix_env_core import registry
from motrix_envs import register

TIERS = [
    ("12-16 m/s, radius 0.5", (12.0, 16.0), 0.5),
    ("12-16 m/s, radius 1.5", (12.0, 16.0), 1.5),
    ("16-20 m/s, full range", (16.0, 20.0), 4.9),
]


def load_policy(path: pathlib.Path, ours: bool):
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    if ours:
        model = ck["model"]
        obs_dim, hidden = int(ck["obs_dim"]), int(ck["hidden"])
        net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 3)
        )
        pairs = (("trunk.0", 0), ("trunk.2", 2), ("actor", 4))
        for key, idx in pairs:
            net[idx].weight.data.copy_(model[f"{key}.weight"])
            net[idx].bias.data.copy_(model[f"{key}.bias"])
        net.eval()
        return net, torch.zeros(obs_dim), torch.ones(obs_dim)
    state = ck["policy"]
    obs_dim, hidden = int(state["net.0.weight"].shape[1]), int(state["net.0.weight"].shape[0])
    net = nn.Sequential(
        nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 3)
    )
    net[0].weight.data.copy_(state["net.0.weight"])
    net[0].bias.data.copy_(state["net.0.bias"])
    net[2].weight.data.copy_(state["net.2.weight"])
    net[2].bias.data.copy_(state["net.2.bias"])
    net[4].weight.data.copy_(state["mean_layer.weight"])
    net[4].bias.data.copy_(state["mean_layer.bias"])
    net.eval()
    pre = ck["observation_preprocessor"]
    return net, pre["running_mean"].float(), pre["running_variance"].float()


def run(net, mean, var, speed, radius, num_envs: int, rounds: int) -> float:
    env = registry.make(
        "tennis_catch",
        num_envs=num_envs,
        env_cfg_override={"serve_speed_range": speed, "serve_radius": radius},
    )
    env.init_state()
    caught = done = 0
    for _ in range(rounds):
        for _ in range(240):
            with torch.inference_mode():
                x = ((torch.from_numpy(env.state.obs.policy.numpy()) - mean) / torch.sqrt(var + 1e-8)).clamp(-5, 5)
                action = net(x).clamp(-1.0, 1.0).numpy()
            state = env.step(torch.from_numpy(action))
            d = state.done.numpy()
            if d.any():
                caught += int((state.metrics["catch_rate"].numpy()[d] > 0).sum())
                done += int(d.sum())
    return caught / max(done, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, type=pathlib.Path)
    parser.add_argument("--ours", action="store_true", help="ActorCritic checkpoint instead of SKRL")
    parser.add_argument("--envs", type=int, default=256)
    parser.add_argument("--rounds", type=int, default=2)
    args = parser.parse_args()

    register()
    net, mean, var = load_policy(args.ckpt, args.ours)
    print(f"policy {args.ckpt.name}")
    for label, speed, radius in TIERS:
        rate = run(net, mean, var, speed, radius, args.envs, args.rounds)
        print(f"{label:<28}{rate:>8.0%}")


if __name__ == "__main__":
    main()
