"""Score one checkpoint on a single serve tier, for checkpoint sweeps.

The three-tier evaluation is the reporting protocol; this is the fast companion
used to pick a checkpoint out of a run (one tier instead of three, and a smaller
batch).  Same observation and action scaling as ``eval_three_tiers.py``.

    python eval_tier.py --ckpt <run>/checkpoints/agent_300.pt --tier 3
    python eval_tier.py --ckpt policies/catch_ppo.pt --ours --tier 3
"""

from __future__ import annotations

import argparse
import pathlib

import torch

from eval_three_tiers import TIERS, load_policy
from motrix_env_core import registry
from motrix_envs import register

register()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, type=pathlib.Path)
    parser.add_argument("--ours", action="store_true", help="ActorCritic checkpoint instead of SKRL")
    parser.add_argument("--tier", type=int, default=3, choices=(1, 2, 3))
    parser.add_argument("--envs", type=int, default=128)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    label, speed, radius = TIERS[args.tier - 1]
    net, mean, var = load_policy(args.ckpt, args.ours)
    env = registry.make(
        "tennis_catch",
        num_envs=args.envs,
        seed=args.seed,
        env_cfg_override={"serve_speed_range": speed, "serve_radius": radius},
    )
    env.init_state()
    caught = done = 0
    for _ in range(240 * args.rounds):
        with torch.no_grad():
            x = ((torch.from_numpy(env.state.obs.policy.numpy()) - mean) / torch.sqrt(var + 1e-8)).clamp(-5, 5)
            action = net(x).clamp(-1.0, 1.0).numpy()
        state = env.step(torch.from_numpy(action))
        finished = state.done.numpy()
        if finished.any():
            caught += int((state.metrics["catch_rate"].numpy()[finished] > 0).sum())
            done += int(finished.sum())
    print(f"{args.ckpt.name:20s} tier{args.tier} {100 * caught / max(done, 1):5.1f}%  ({caught}/{done})  [{label}]")


if __name__ == "__main__":
    main()
