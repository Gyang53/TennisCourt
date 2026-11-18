"""Convert one of our ActorCritic checkpoints into an SKRL warm-start checkpoint.

The shipped catch policies (``policies/catch_bc_big.pt``) are Tanh MLPs on the raw
61-dim trajectory observation.  MotrixLab's SKRL trainer normalises observations
with a running standard scaler, so the exported checkpoint carries an
**identity** scaler (mean 0, variance 1) that the trainer then freezes: the
policy keeps seeing the raw observation it was fitted on.

    python export_skrl_warmstart.py --ckpt policies/catch_bc_big.pt \
        --out <run>/checkpoints/warmstart_best.pt [--std 0.1]

Match the network in the task config: ``hiddens: [256, 256]`` and
``hidden_activation: [tanh]`` when training from this checkpoint.
"""

from __future__ import annotations

import argparse
import math
import pathlib

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, type=pathlib.Path)
    parser.add_argument("--out", required=True, type=pathlib.Path)
    parser.add_argument("--std", type=float, default=0.1, help="exploration band for fine-tuning")
    args = parser.parse_args()

    src = torch.load(str(args.ckpt), map_location="cpu", weights_only=False)
    model = src["model"]
    obs_dim = int(src["obs_dim"])
    hidden = int(src["hidden"])

    state = {
        "net.0.weight": model["trunk.0.weight"].clone(),
        "net.0.bias": model["trunk.0.bias"].clone(),
        "net.2.weight": model["trunk.2.weight"].clone(),
        "net.2.bias": model["trunk.2.bias"].clone(),
        "mean_layer.weight": model["actor.weight"].clone(),
        "mean_layer.bias": model["actor.bias"].clone(),
        # the candidate policies were trained at std ~0.37; the fine-tune band is
        # set here because this is what the sampler actually uses
        "log_std_parameter": torch.full((model["actor.weight"].shape[0],), math.log(args.std)),
    }
    checkpoint = {
        "policy": state,
        # identity: the policy was fitted on raw observations
        "observation_preprocessor": {
            "running_mean": torch.zeros(obs_dim, dtype=torch.float64),
            "running_variance": torch.ones(obs_dim, dtype=torch.float64),
            "current_count": torch.tensor(1.0, dtype=torch.float64),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, str(args.out))
    print(
        f"wrote {args.out}: obs {obs_dim}, hidden {hidden}, std {args.std}, "
        f"identity normalizer (mean 0, var 1)"
    )


if __name__ == "__main__":
    main()
