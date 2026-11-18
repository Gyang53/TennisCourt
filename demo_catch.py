"""Show what the catch policy actually does.

1. runs N episodes headless and prints the catch rate
2. records trajectories and writes a top-down SVG (court, ball path, robot path)
3. renders a few frames around the catch of the showcased episodes

    .venv/bin/python demo_catch.py --episodes 64 --radius 1.5 --speed 12 16
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from catch_env import CatchEnv, OBS_MODE_FOR_DIM
from eval_catch import load_policy
from train_catch import VMAX

COURT_X, COURT_Y = 11.885, 5.485
SERVICE_X, SINGLES_Y = 6.4, 4.115


def run_episodes(net, episodes, radius, speed, clear, record=8, steps=240, obs_mode="both"):
    env = CatchEnv(num_envs=episodes, seed=21, serve_speed=speed, serve_clearance=clear,
                   radius=radius, aim=(-4.9, 0.0), obs_mode=obs_mode)
    caught = np.zeros(episodes)
    ball_path = np.zeros((episodes, steps, 3), np.float32)
    robot_path = np.zeros((episodes, steps, 2), np.float32)
    catch_step = np.full(episodes, -1)
    end_step = np.full(episodes, steps - 1)
    for step in range(steps):
        # record *before* stepping: env.step() resets finished envs internally
        ball_path[:, step] = env.ball_pose()
        robot_path[:, step] = env.robot_pose()[0][:, :2]
        obs = env.obs()
        with torch.no_grad():
            a, _, _ = net.act(torch.from_numpy(obs), deterministic=True)
        obs, _rew, done, info = env.step(a * VMAX)
        newly = done & (info["caught"] > 0) & (catch_step < 0)
        catch_step[newly] = step
        end_step[done] = step
        caught = np.where(done, info["caught"], caught)
        if done.all():
            break
    return caught, ball_path, robot_path, end_step, env


def _png(path, img):
    """Minimal RGB PNG writer (no image libraries available here)."""
    import struct
    import zlib

    h, w, _ = img.shape
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))

    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 6))
           + chunk(b"IEND", b""))
    with open(path, "wb") as f:
        f.write(png)


def write_png(path, ball, robot, caught, idx, ends=None):
    """Top-down court plot: ball path (blue), robot path (orange), catch (green)."""
    W, H = 1100, 760
    img = np.zeros((H, W, 3), np.uint8)
    img[:, :] = (18, 24, 38)
    s = min((W - 160) / (2 * COURT_X + 6), (H - 140) / (2 * COURT_Y + 6))
    cx, cy = W / 2, H / 2

    def px(x, y):
        return int(cx + x * s), int(cy - y * s)

    def line(a, b, color, thick=2):
        n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1]))) + 1
        for t in np.linspace(0, 1, n):
            x = int(a[0] + (b[0] - a[0]) * t)
            y = int(a[1] + (b[1] - a[1]) * t)
            img[max(0, y - thick // 2):y + thick // 2 + 1,
                max(0, x - thick // 2):x + thick // 2 + 1] = color

    def poly(points, color, thick=2):
        pts = [px(*q) for q in points]
        for a, b in zip(pts, pts[1:]):
            line(a, b, color, thick)

    def rect(x0, y0, x1, y1, color, thick=2):
        line(px(x0, y0), px(x1, y0), color, thick)
        line(px(x1, y0), px(x1, y1), color, thick)
        line(px(x1, y1), px(x0, y1), color, thick)
        line(px(x0, y1), px(x0, y0), color, thick)

    rect(-COURT_X - 2.6, -COURT_Y - 2.6, COURT_X + 2.6, COURT_Y + 2.6, (40, 70, 50), 1)
    rect(-COURT_X, -COURT_Y, COURT_X, COURT_Y, (90, 120, 170), 2)
    for sy_ in (SINGLES_Y, -SINGLES_Y):
        line(px(-COURT_X, sy_), px(COURT_X, sy_), (120, 130, 150), 1)
    for sgn in (-1, 1):
        line(px(sgn * SERVICE_X, SINGLES_Y), px(sgn * SERVICE_X, -SINGLES_Y), (120, 130, 150), 1)
    line(px(0, COURT_Y + 1.4), px(0, -COURT_Y - 1.4), (235, 235, 245), 3)

    ends = None if ends is None else ends
    for k in idx:
        end = len(ball[k]) if ends is None else max(3, min(int(ends[k]) + 1, len(ball[k])))
        poly(ball[k, :end, :2:], (70, 140, 255), 2)
        poly(robot[k, :end], (255, 150, 60), 2)
        if caught[k]:
            c = px(*ball[k, end - 1, :2])
            for dx in range(-7, 8):
                for dy in range(-7, 8):
                    if 40 < dx * dx + dy * dy < 60:
                        img[max(0, c[1] + dy), max(0, c[0] + dx)] = (60, 220, 130)
    _png(path, img)


def write_svg(path, ball, robot, caught, idx, ends=None):
    """Top-down SVG: court, ball path coloured by height, robot path, catch point."""
    W, H = 900, 620
    sx = (W - 120) / (2 * COURT_X + 6)
    sy = (H - 120) / (2 * COURT_Y + 6)
    s = min(sx, sy)
    cx, cy = W / 2, H / 2

    def px(x, y):
        return cx + x * s, cy - y * s

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
           f'width="{W}" height="{H}" font-family="sans-serif" font-size="13" fill="#3a4150">',
           f'<rect width="{W}" height="{H}" fill="none"/>']
    # surround
    x0, y0 = px(-COURT_X - 2.8, COURT_Y + 2.8)
    x1, y1 = px(COURT_X + 2.8, -COURT_Y - 2.8)
    out.append(f'<rect x="{x0}" y="{y0}" width="{x1-x0}" height="{y1-y0}" '
               f'fill="#f2f4f7" stroke="#c9ced6"/>')
    x0, y0 = px(-COURT_X, COURT_Y)
    x1, y1 = px(COURT_X, -COURT_Y)
    out.append(f'<rect x="{x0}" y="{y0}" width="{x1-x0}" height="{y1-y0}" '
               f'fill="none" stroke="#2f80ed" stroke-width="2"/>')
    for hx in (SERVICE_X,):
        for sgn in (-1, 1):
            a = px(sgn * hx, SINGLES_Y)
            b = px(sgn * hx, -SINGLES_Y)
            out.append(f'<line x1="{a[0]}" y1="{a[1]}" x2="{b[0]}" y2="{b[1]}" '
                       f'stroke="#7a8291" stroke-dasharray="6 5"/>')
    for sy_ in (SINGLES_Y, -SINGLES_Y):
        a = px(-COURT_X, sy_)
        b = px(COURT_X, sy_)
        out.append(f'<line x1="{a[0]}" y1="{a[1]}" x2="{b[0]}" y2="{b[1]}" '
                   f'stroke="#c9ced6"/>')
    a, b = px(0, COURT_Y + 1.5), px(0, -COURT_Y - 1.5)
    out.append(f'<line x1="{a[0]}" y1="{a[1]}" x2="{b[0]}" y2="{b[1]}" '
               f'stroke="#3a4150" stroke-width="3"/>')
    for k in idx:
        end = len(ball[k]) if ends is None else max(3, min(int(ends[k]) + 1, len(ball[k])))
        ball_xy = ball[k, :end, :2]
        rob = robot[k, :end]
        pts = " ".join(f"{px(*p)[0]:.1f},{px(*p)[1]:.1f}" for p in ball_xy[::3] if p[0] > -60)
        out.append(f'<polyline points="{pts}" fill="none" stroke="#2f80ed" stroke-width="2" opacity="0.85"/>')
        rpts = " ".join(f"{px(*p)[0]:.1f},{px(*p)[1]:.1f}" for p in rob[::3])
        out.append(f'<polyline points="{rpts}" fill="none" stroke="#eb5757" stroke-width="2" opacity="0.9"/>')
        if caught[k]:
            p_end = ball_xy[-1]
            out.append(f'<circle cx="{px(*p_end)[0]:.1f}" cy="{px(*p_end)[1]:.1f}" r="7" '
                       f'fill="none" stroke="#27ae60" stroke-width="3"/>')
    out.append(f'<text x="{W-330}" y="{H-40}" fill="#7a8291">'
               f'ball path (--), robot path (—); overall catch rate '
               f'{100*np.mean(caught > 0):.0f}%</text>')
    out.append("</svg>")
    open(path, "w").write("\n".join(out))


def render_frames(net, radius, speed, clear, seed, frames=3):
    import motrixsim.render as mr

    env = CatchEnv(num_envs=1, seed=seed, serve_speed=speed, serve_clearance=clear,
                   radius=radius, aim=(-4.9, 0.0))
    # first pass: where does the catch happen?
    obs = env.obs()
    catch_step = None
    for step in range(240):
        with torch.no_grad():
            a, _, _ = net.act(torch.from_numpy(obs), deterministic=True)
        obs, _rew, done, info = env.step(a * VMAX)
        if done[0]:
            catch_step = step if info["caught"][0] > 0 else None
            break
    if catch_step is None:
        print("no catch in this episode; skipping render")
        return []
    targets = {max(0, catch_step - 45), max(0, catch_step - 12), catch_step}
    env = CatchEnv(num_envs=1, seed=seed, serve_speed=speed, serve_clearance=clear,
                   radius=radius, aim=(-4.9, 0.0))
    obs = env.obs()
    saved = []
    with mr.RenderApp(headless=False, fps=60) as app:
        app.launch(env.model, batch=1)
        cam = app.system_camera
        cam.set_view(lookat=(-11.2, 0.0, 0.55), distance=5.2, elevation=-14, azimuth=-135)
        app.sync(env.data)
        for step in range(catch_step + 1):
            with torch.no_grad():
                a, _, _ = net.act(torch.from_numpy(obs), deterministic=True)
            obs, _rew, done, info = env.step(a * VMAX)
            if step in targets:
                app.sync(env.data)
                task = cam.capture()
                for _ in range(40):
                    app.sync(env.data)
                    try:
                        img = task.take_image()
                    except Exception:
                        img = None
                    if img is not None:
                        name = f"/home/yangg/data/tennis_robot/rl_catch_{len(saved)}.png"
                        img.save_to_disk(name)
                        saved.append(name)
                        break
                    time.sleep(1 / 60)
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="policies/catch_bc.pt")
    ap.add_argument("--episodes", type=int, default=64)
    ap.add_argument("--radius", type=float, default=1.5)
    ap.add_argument("--speed", type=float, nargs=2, default=(12.0, 16.0))
    ap.add_argument("--clearance", type=float, nargs=2, default=(0.6, 1.0))
    ap.add_argument("--render-seeds", type=int, nargs="*", default=[3, 7])
    args = ap.parse_args()

    net = load_policy(args.policy)
    caught, ball, robot, _catch_step, _env = run_episodes(
        net, args.episodes, (args.radius, args.radius * 0.7), args.speed, args.clearance)
    print(f"episodes {args.episodes}  caught {int((caught>0).sum())}  "
          f"rate {100*np.mean(caught>0):.0f}%  serve {args.speed[0]:.0f}-{args.speed[1]:.0f} m/s "
          f"radius {args.radius}")
    good = [int(k) for k in np.argsort(-caught)[:6]]
    write_svg("/home/yangg/data/tennis_robot/rl_catch_paths.svg", ball, robot, caught,
              [k for k in good if caught[k] > 0][:4])
    print("wrote rl_catch_paths.svg")
    for seed in args.render_seeds:
        saved = render_frames(net, (args.radius, args.radius * 0.7), args.speed,
                              args.clearance, seed)
        print("frames:", saved)


if __name__ == "__main__":
    main()
