"""Convert an SKRL checkpoint back into our ActorCritic policy format.

The training stack lives in MotrixLab (SKRL naming); the demos, the ROS2 bridge
and ``eval_catch.py`` load ``policies/*.pt`` in our own layout.  This is the bridge
for promoting a fine-tuned policy:

    python import_skrl_to_policy.py --ckpt <run>/checkpoints/best_agent.pt \
        --out policies/catch_ppo.pt
"""

from __future__ import annotations

import argparse
import pathlib

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, type=pathlib.Path)
    parser.add_argument("--out", required=True, type=pathlib.Path)
    args = parser.parse_args()

    ck = torch.load(str(args.ckpt), map_location="cpu", weights_only=False)
    state = ck["policy"]
    obs_dim = int(state["net.0.weight"].shape[1])
    hidden = int(state["net.0.weight"].shape[0])
    act_dim = int(state["mean_layer.weight"].shape[0])
    if act_dim != 3 or int(state["net.2.weight"].shape[0]) != hidden:
        raise SystemExit(f"unexpected policy shape: obs {obs_dim}, hidden {hidden}, act {act_dim}")

    # the critic is trained separately in SKRL; keep it only if the checkpoint
    # carries a matching head (the shipped tooling ignores it)
    critic = {"critic.weight": torch.zeros(1, hidden), "critic.bias": torch.zeros(1)}
    for key in ("value_layer.weight", "value_layer.bias"):
        if key in state:
            target = key.replace("value_layer", "critic")
            critic[target] = state[key].clone()

    out = {
        "obs_dim": obs_dim,
        "hidden": hidden,
        "model": {
            "trunk.0.weight": state["net.0.weight"].clone(),
            "trunk.0.bias": state["net.0.bias"].clone(),
            "trunk.2.weight": state["net.2.weight"].clone(),
            "trunk.2.bias": state["net.2.bias"].clone(),
            "actor.weight": state["mean_layer.weight"].clone(),
            "actor.bias": state["mean_layer.bias"].clone(),
            **critic,
            "log_std": state["log_std_parameter"].clone(),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, str(args.out))
    print(f"wrote {args.out}: obs {obs_dim}, hidden {hidden}, std {float(out['model']['log_std'].exp()[0]):.2f}")


if __name__ == "__main__":
    main()
