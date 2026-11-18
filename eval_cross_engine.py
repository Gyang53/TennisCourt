"""Evaluate an SKRL ``tennis_catch`` checkpoint on the MotrixSim reference engine.

The policy is trained in MotrixLab (optionally on the GPU physics backend); this
script runs it in the hand-written MotrixSim environment that the ROS2 bridge and
the shipped BC policy are validated against, so the number it prints is
cross-engine, not a replay of the training loop.

    source ~/data/MotrixLab/.venv/bin/activate
    python eval_cross_engine.py --ckpt runs/.../checkpoints/best_agent.pt

The policy forward mirrors ``motrix_rl_skrl.torch.train.ppo``: MLP(ELU) ->
``mean_layer``, with SKRL's running standard scaler on the observations.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from ball_physics import VMAX_FORE, VMAX_LAT, WMAX
from catch_env import CatchEnv

OBS, ACT = 12, 3
# CatchEnv.step takes the raw vehicle command in m/s and rad/s; the policy emits
# the normalised +/-1 action the MotrixLab env expects, so scale here
ACTION_SCALE = np.array([VMAX_FORE, VMAX_LAT, WMAX], dtype=np.float32)
NORMALIZER_EPS, NORMALIZER_CLIP = 1e-8, 5.0


def build_policy(ckpt: pathlib.Path):
    """Load the checkpoint's policy into the same MLP the trainer builds."""
    weights = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    state, normalizer = weights["policy"], weights["observation_preprocessor"]
    obs_size = state["net.0.weight"].shape[1]
    act_size = state["mean_layer.bias"].shape[0]

    net = nn.Sequential(
        nn.Linear(obs_size, state["net.0.weight"].shape[0]),
        nn.ELU(),
        nn.Linear(state["net.0.weight"].shape[0], state["net.2.weight"].shape[0]),
        nn.ELU(),
        nn.Linear(state["net.2.weight"].shape[0], act_size),
    )
    net[0].weight.data.copy_(state["net.0.weight"])
    net[0].bias.data.copy_(state["net.0.bias"])
    net[2].weight.data.copy_(state["net.2.weight"])
    net[2].bias.data.copy_(state["net.2.bias"])
    net[4].weight.data.copy_(state["mean_layer.weight"])
    net[4].bias.data.copy_(state["mean_layer.bias"])
    net.eval()

    mean = normalizer["running_mean"].float()
    std = torch.sqrt(normalizer["running_variance"].float() + NORMALIZER_EPS)

    def act(obs: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            scaled = ((torch.from_numpy(np.asarray(obs, np.float32)) - mean) / std).clamp(
                -NORMALIZER_CLIP, NORMALIZER_CLIP
            )
            return net(scaled).clamp(-1.0, 1.0).numpy()

    return act, (obs_size, act_size)


def run(policy, num_envs: int, radius: float, speed, rounds: int) -> float:
    env = CatchEnv(
        num_envs=num_envs,
        seed=0,
        serve_speed=speed,
        serve_clearance=(0.6, 1.0),
        radius=(radius, radius),
        obs_mode="state",
    )
    caught, episodes_done = 0, 0
    for _ in range(rounds):
        obs = env.obs()
        for _ in range(240):
            obs, _, done, info = env.step(policy(obs) * ACTION_SCALE)
            if done.any():
                caught += int(info["caught"][done].sum())
                episodes_done += int(done.sum())
        if episodes_done >= num_envs * 3:
            break
    env.close()
    return caught / max(episodes_done, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, type=pathlib.Path)
    parser.add_argument("--envs", type=int, default=512)
    # one round is 240 control steps per world: enough for a rate at 512 worlds,
    # and the reference engine costs ~3.4k env-steps/s, so rounds are the budget
    parser.add_argument("--rounds", type=int, default=2)
    args = parser.parse_args()

    policy, shapes = build_policy(args.ckpt)
    print(f"policy {args.ckpt} | obs/act {shapes}")
    print(f"{'serve':<28}{'catch rate':>12}")
    cases = [
        ("12-16 m/s, radius 0.5", 0.5, (12.0, 16.0)),
        ("12-16 m/s, radius 1.5", 1.5, (12.0, 16.0)),
        ("16-20 m/s, full range", 4.9, (16.0, 20.0)),
    ]
    for label, radius, speed in cases:
        rate = run(policy, args.envs, radius, speed, args.rounds)
        print(f"{label:<28}{rate:>11.0%}")


if __name__ == "__main__":
    main()
