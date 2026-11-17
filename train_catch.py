"""PPO training for the catch task on one batched MotrixSim scene.

All environments live in a single batched SceneData, so the physics runs in
parallel in the Rust backend (see catch_env.py).

Actions are normalised (-1..1) vehicle-frame commands: forward (camera axis),
left, yaw rate.  A curriculum raises the serve speed and lowers the net
clearance as training progresses, ending at the real serve distribution.

    .venv/bin/python train_catch.py --envs 48 --iters 400
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from catch_env import CatchEnv, OBS_DIM

VMAX = np.array([4.0, 4.5, 3.0])
HIDDEN = 128

# serve speed, net clearance, aim radius (how far from the robot the serve bounces).
# The teacher advances a phase only once the catch rate proves the policy can
# handle the current one - a time-based schedule moved on before it had learned.
CURRICULUM = [
    ((8.0, 11.0), (0.5, 0.9), (0.5, 0.3)),
    ((11.0, 14.0), (0.5, 0.9), (1.0, 0.6)),
    ((13.0, 16.0), (0.6, 1.0), (1.8, 1.2)),
    ((15.0, 18.0), (0.5, 0.9), (2.5, 2.0)),
    ((16.0, 20.0), (0.45, 0.8), (3.0, 3.0)),
]
PROMOTE_RATE = 0.25          # rolling catch rate that unlocks the next phase


class ActorCritic(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, act_dim=3, hidden=HIDDEN):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.actor = nn.Linear(hidden, act_dim)
        self.critic = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.full((act_dim,), -1.0))
        for m in (self.trunk[0], self.trunk[2], self.actor, self.critic):
            nn.init.orthogonal_(m.weight, np.sqrt(2))
            nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor.weight, 0.01)

    def forward(self, x):
        h = self.trunk(x)
        return self.actor(h), self.critic(h).squeeze(-1)

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        mean, value = self.forward(obs)
        std = self.log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        action = mean if deterministic else dist.sample()
        raw = action.clamp(-1.0, 1.0)
        logp = dist.log_prob(action).sum(-1)
        return raw.numpy(), value.numpy(), logp.numpy()


def apply_phase(env, phase: int):
    speed, clearance, radius = CURRICULUM[min(phase, len(CURRICULUM) - 1)]
    env.serve_speed = speed
    env.serve_clearance = clearance
    env.radius = radius


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env = CatchEnv(num_envs=args.envs, seed=args.seed)
    net = ActorCritic()
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)

    obs = env.obs()
    ep_return = np.zeros(args.envs)
    finished_returns, catch_flags = [], []
    t0 = time.perf_counter()
    stats = {"iters": 0, "steps": 0}

    phase = 0
    apply_phase(env, phase)
    for it in range(1, args.iters + 1):
        buf_obs, buf_act, buf_logp, buf_val, buf_rew, buf_done = [], [], [], [], [], []
        for _ in range(args.rollout):
            obs_t = obs                      # the observation the action is sampled from
            a, v, logp = net.act(torch.from_numpy(obs_t))
            obs, rew, done, info = env.step(a * VMAX)
            buf_obs.append(obs_t)            # (was appending the post-step obs: off by one)
            buf_act.append(a)
            buf_logp.append(logp)
            buf_val.append(v)
            buf_rew.append(rew)
            buf_done.append(done.astype(np.float32))
            ep_return += rew
            for e in np.where(done)[0]:
                finished_returns.append(float(ep_return[e]))
                catch_flags.append(float(info["caught"][e]))
                ep_return[e] = 0.0
        stats["steps"] += args.envs * args.rollout

        o = torch.from_numpy(np.stack(buf_obs))
        a = torch.from_numpy(np.stack(buf_act))
        lp = torch.from_numpy(np.stack(buf_logp))
        val = torch.from_numpy(np.stack(buf_val))
        rew = torch.from_numpy(np.stack(buf_rew))
        don = torch.from_numpy(np.stack(buf_done))

        # GAE
        with torch.no_grad():
            _, last_v = net(torch.from_numpy(obs))
            adv = torch.zeros_like(rew)
            gae = torch.zeros(args.envs)
            for t in reversed(range(args.rollout)):
                nxt = last_v if t == args.rollout - 1 else val[t + 1]
                mask = 1.0 - don[t]
                delta = rew[t] + args.gamma * nxt * mask - val[t]
                gae = delta + args.gamma * args.lam * mask * gae
                adv[t] = gae
            ret = adv + val
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        flat = lambda x: x.reshape(-1, *x.shape[2:])
        O, A, LP, ADV, RET = flat(o), flat(a), flat(lp), flat(adv), flat(ret)
        n = O.shape[0]
        idx = np.arange(n)
        for _ in range(args.epochs):
            np.random.shuffle(idx)
            for start in range(0, n, args.minibatch):
                b = idx[start:start + args.minibatch]
                mean, value = net(O[b])
                dist = torch.distributions.Normal(mean, net.log_std.exp())
                new_lp = dist.log_prob(A[b]).sum(-1)
                ratio = (new_lp - LP[b]).exp()
                s1 = ratio * ADV[b]
                s2 = ratio.clamp(1 - args.clip, 1 + args.clip) * ADV[b]
                actor_loss = -torch.min(s1, s2).mean()
                value_loss = (value - RET[b]).pow(2).mean()
                entropy = dist.entropy().sum(-1).mean()
                loss = actor_loss + args.vf_coef * value_loss - args.ent_coef * entropy
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()

        if it % args.log_every == 0 or it == 1:
            window = max(1, min(len(catch_flags), 200))
            rate = 100.0 * np.mean(catch_flags[-window:]) if catch_flags else float("nan")
            ret_mean = np.mean(finished_returns[-window:]) if finished_returns else float("nan")
            sp = time.perf_counter() - t0
            if rate >= 100.0 * PROMOTE_RATE and phase < len(CURRICULUM) - 1 and len(catch_flags) > 150:
                phase += 1
                apply_phase(env, phase)
                print(f"  -> phase {phase} unlocked (catch {rate:.0f}%)", flush=True)
            print(f"iter {it:4d}  steps {stats['steps']/1e6:5.2f}M  "
                  f"catch {rate:5.1f}%  return {ret_mean:6.2f}  "
                  f"phase {phase} serve {env.serve_speed[0]:.0f}-{env.serve_speed[1]:.0f} m/s  "
                  f"{stats['steps']/sp/1e3:6.1f}k steps/s  {sp/60:5.1f} min", flush=True)
            if it % args.save_every == 0:
                Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                torch.save({"model": net.state_dict(), "obs_dim": OBS_DIM}, args.out)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": net.state_dict(), "obs_dim": OBS_DIM}, args.out)
    print(f"saved {args.out}")
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=48)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--rollout", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--ent-coef", type=float, default=0.005)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--out", default="policies/catch_ppo.pt")
    train(ap.parse_args())


if __name__ == "__main__":
    main()
