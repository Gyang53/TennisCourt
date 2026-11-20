"""Score a motrix.fastsac checkpoint on one serve tier.

``mxlab play`` cannot load the fastsac provider (its schema file is missing from
``algo_base``), so this builds the actor and the observation normalizer from the
checkpoint exactly as the trainer does and runs them through the same environment
and tier definitions as ``eval_three_tiers.py``.

    python eval_fastsac.py --ckpt <run>/checkpoints/model_0005750.pt \
        --run-config <run>/task_config.yaml --tier 3
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np

import torch
import yaml

from motrix_env_core import registry
from motrix_envs import register
from motrix_envs.basic.tennis_catch.state_io import dof_layout
from eval_three_tiers import TIERS

from motrix_rl_builtin.fastsac.buffer import EmpiricalNormalization
from motrix_rl_builtin.fastsac.networks import Actor

register()


def load_fastsac(ckpt: pathlib.Path, run_config: pathlib.Path):
    """Build the actor and observation normalizer from a fastsac checkpoint."""
    agent_cfg = yaml.safe_load(pathlib.Path(run_config).read_text())["algo"]["agent"]
    state = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    actor_state = state["actor"]
    obs_dim = int(actor_state["net.0.weight"].shape[1])
    actor = Actor(
        n_obs=obs_dim,
        n_act=3,
        hidden_dim=int(agent_cfg["actor_hidden_dim"]),
        log_std_max=float(agent_cfg["log_std_max"]),
        log_std_min=float(agent_cfg["log_std_min"]),
        use_tanh=bool(agent_cfg["use_tanh"]),
        use_layer_norm=bool(agent_cfg["use_layer_norm"]),
    )
    actor.load_state_dict(actor_state, strict=True)
    actor.eval()
    normalizer = EmpiricalNormalization(shape=obs_dim, device="cpu")
    normalizer.load_state_dict(state["obs_normalizer"], strict=True)
    normalizer.eval()
    return actor, normalizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, type=pathlib.Path)
    parser.add_argument("--run-config", required=True, type=pathlib.Path, help="the run's task_config.yaml")
    parser.add_argument("--tier", type=int, default=3, choices=(1, 2, 3))
    parser.add_argument("--serve-speed", type=float, nargs=2, default=None,
                        help="override the tier band, e.g. the deployment band 16 20")
    parser.add_argument("--serve-radius", type=float, default=None,
                        help="override the tier radius, e.g. the deployment band 1.5")
    parser.add_argument("--envs", type=int, default=128)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    actor, normalizer = load_fastsac(args.ckpt, args.run_config)

    label, speed, radius = TIERS[args.tier - 1]
    if args.serve_speed is not None:
        speed = tuple(args.serve_speed)
    if args.serve_radius is not None:
        radius = args.serve_radius
    if args.serve_speed is not None or args.serve_radius is not None:
        label = f"{speed[0]:.0f}-{speed[1]:.0f} m/s, radius {radius:.1f} (custom)"
    env = registry.make(
        "tennis_catch",
        num_envs=args.envs,
        seed=args.seed,
        env_cfg_override={"serve_speed_range": speed, "serve_radius": radius},
    )
    env.init_state()
    layout = dof_layout(env.sim)
    base = layout["base_pos"]
    pose0 = np.array(env.sim._data.dof_pos[:, base], dtype=np.float64)
    yaw0 = float(np.median(np.arctan2(2 * (pose0[:, 6] * pose0[:, 5] + pose0[:, 3] * pose0[:, 4]),
                                      1 - 2 * (pose0[:, 4] ** 2 + pose0[:, 5] ** 2))))
    caught = done = 0
    yaw_offs: list[float] = []
    yaw_cmds: list[float] = []
    mis_all: list[float] = []
    corrections = correctable = 0
    prev_mis = None
    for _ in range(240 * args.rounds):
        with torch.no_grad():
            obs = torch.from_numpy(env.state.obs.policy.numpy()).float()
            action, _, _ = actor(normalizer(obs, update=False))
        yaw_cmds.append(float(action[..., 2].abs().mean()))
        ddx, ddy, _ = env._base_frame_ball()
        bearing = float(np.median(np.arctan2(np.asarray(ddy), np.asarray(ddx))))
        mis = (bearing + np.pi / 2.0 + np.pi) % (2.0 * np.pi) - np.pi
        mis_all.append(abs(mis))
        if prev_mis is not None:
            correctable += 1
            corrections += int(abs(mis) < abs(prev_mis))
        prev_mis = mis
        state = env.step(action)
        pose = np.array(env.sim._data.dof_pos[:, base], dtype=np.float64)
        yaw = np.arctan2(2 * (pose[:, 6] * pose[:, 5] + pose[:, 3] * pose[:, 4]),
                         1 - 2 * (pose[:, 4] ** 2 + pose[:, 5] ** 2))
        yaw_offs.append(float(np.median(np.abs(np.degrees((yaw - yaw0 + np.pi) % (2 * np.pi) - np.pi)))))
        finished = state.done.numpy()
        if finished.any():
            caught += int((state.metrics["catch_rate"].numpy()[finished] > 0).sum())
            done += int(finished.sum())
    over = 100.0 * float(np.mean(np.array(yaw_offs) > 40.0)) if yaw_offs else 0.0
    print(f"aim: mean|misalign| {np.degrees(np.mean(mis_all)):5.1f} deg, "
          f"<30deg {100.0 * np.mean(np.array(mis_all) < np.radians(30)):4.1f}%, "
          f"turning toward aim {100.0 * corrections / max(correctable, 1):4.1f}% of steps")
    print(f"yaw: mean|off| {np.mean(yaw_offs):5.1f} deg, max {np.max(yaw_offs):5.1f} deg, "
          f"steps>40deg {over:4.1f}%, mean|wz| {np.mean(yaw_cmds):.2f}")
    print(
        f"{args.ckpt.name:22s} tier{args.tier} {100 * caught / max(done, 1):5.1f}%  ({caught}/{done})  [{label}]"
    )


if __name__ == "__main__":
    main()
