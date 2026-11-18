"""Simulation physics shared by the ROS bridge, the training env and the tools.

Ball flight, bounce, the legal-serve solver, and the measured wheel -> body
calibration.  The MJCF has no fluid model and its soft contact cannot produce a
repeatable restitution at 0.5 ms, so those live here; the ROS bridge, the
training environment and the tools all import them from this module.
"""
from __future__ import annotations

import math

import numpy as np

BALL_R = 0.0335                  # m, ball radius (MJCF)
COURT_X, SINGLES_Y = 11.885, 4.115
SERVICE_X = 6.4                  # service line, |x|
NET_TOP = 1.07                   # net collision box top (flat in the MJCF)
GRAV = np.array([0.0, 0.0, -9.81])
GRAVITY = 9.81                  # scalar form, used by the serve sampler
START = (-11.885, 0.0)          # robot home: baseline centre
# aerodynamic drag of a tennis ball: 0.5 * rho * Cd * A / m  [1/m]
# rho 1.2 kg/m3, Cd 0.6, A = pi r^2, m 0.057 kg.  The MJCF has no fluid model,
# so the bridge applies this to the ball every step (and the serve solver uses it)
DRAG_K = 0.5 * 1.2 * 0.6 * math.pi * BALL_R ** 2 / 0.057


# tennis ball on a hard court: COR 0.73 at the ITF rebound-test impact (7 m/s),
# ~0.55 at a 100 km/h serve.  Modelled as COR = 0.80 - 0.010 * v_normal,
# clamped; the engine's own contact is critically damped and does not bounce.
COR_A, COR_B, COR_MIN, COR_MAX = 0.83, 0.010, 0.45, 0.78
TANGENT_LOSS = 0.90              # horizontal speed kept through the bounce


def bounce_cor(v_normal: float) -> float:
    return float(np.clip(COR_A - COR_B * abs(v_normal), COR_MIN, COR_MAX))


# ------------------------------------------------------------- drive model
WHEELS = ("FL", "FR", "RL", "RR")

# wheel speed -> body velocity, measured on the model (ros2_bridge notes)
G_FORE = 0.09958          # base_link x [m/s] per rad/s
G_LAT = 0.09712           # base_link y [m/s] per rad/s
G_YAW = 0.18613           # base_link z [rad/s] per rad/s
VMAX_FORE, VMAX_LAT, WMAX = 4.0, 4.5, 3.0
WHEEL_KP, TORQUE_LIMIT = 10.0, 12.0
BODY_KP, YAW_KP = 0.30, 0.30


def fly_step(p, v, h):
    """One explicit step of the drag-affected flight."""
    speed = float(np.linalg.norm(v))
    return p + v * h, v + (GRAV - DRAG_K * speed * v) * h


def flight(p0, v0, dt=0.0005, max_t=3.0):
    """Fly ballistically (drag included).  Returns (landing xy, net clearance).

    Landing is the first floor touch; clearance is the ball surface height over
    the net box at x = 0 (None if the flight never crosses the net).
    """
    p, v = np.array(p0, float), np.array(v0, float)
    clear = None
    t = 0.0
    while t < max_t:
        prev = p
        p, v = fly_step(p, v, dt)
        t += dt
        if clear is None and prev[0] > 0.0 >= p[0] and p[0] != prev[0]:
            u = (0.0 - prev[0]) / (p[0] - prev[0])
            clear = float(prev[2] + u * (p[2] - prev[2])) - NET_TOP - BALL_R
        if p[2] <= BALL_R and v[2] < 0.0:
            return p[:2].copy(), clear
    return None, clear


def _launch(speed, azimuth, elevation):
    """Launch velocity of magnitude `speed`; azimuth is measured from the -X
    axis (the direction the serve travels) toward +Y."""
    return np.array([-speed * math.cos(elevation) * math.cos(azimuth),
                     speed * math.cos(elevation) * math.sin(azimuth),
                     speed * math.sin(elevation)])


def catch_window(spawn, v0, z_target=0.30, dt=0.002, max_t=4.0):
    """Where and when the ball first descends through z_target after its first
    bounce - the point the robot has to be under.  Used as the teacher's oracle."""
    p = np.array(spawn, float)
    v = np.array(v0, float)
    bounced = False
    prev_z = float(p[2])
    t = 0.0
    while t < max_t:
        p, v = fly_step(p, v, dt)
        t += dt
        low = p[2] <= BALL_R
        if low:
            vn = -v[2]
            v[2] = vn * bounce_cor(vn)
            v[0] *= TANGENT_LOSS
            v[1] *= TANGENT_LOSS
            p[2] = BALL_R
            bounced = True
        if bounced and prev_z > z_target >= p[2] and v[2] < 0.0:
            return np.array([p[0], p[1]]), t
        prev_z = float(p[2])
    return None, None


def clearance_at(spawn, speed, elevation_deg, azimuth=0.0, dt=0.001):
    """(landing xy, net clearance) of a serve launched at that elevation/azimuth."""
    return flight(spawn, _launch(speed, azimuth, math.radians(elevation_deg)), dt=dt)


def plan_serve(rng, spawn, speed, clearance, spread=0.0):
    """A legal serve at the requested speed and net clearance.

    The elevation is bisected so the aimed flight clears the net by the
    requested amount while the bounce stays inside the diagonal receiving
    service box; the azimuth comes from the straight (drag has no lateral
    component) horizontal path to a random legal bounce point.
    Returns (v0, target, clearance) or (None, reason, None).
    """
    want = clearance + (rng.uniform(0.0, spread) if spread else 0.0)
    side = 1.0 if spawn[1] >= 0.0 else -1.0
    y_land = -side * rng.uniform(0.5, 3.6)
    azimuth = 0.0
    elev = 15.0
    land = clear = None
    for _ in range(3):                       # aim, then refine the aim with the depth it gives
        # clearance grows with elevation, so bisect on it alone and check the
        # resulting depth afterwards (a steeper arc lands deeper)
        lo, hi = 1.0, 45.0
        for _ in range(16):
            mid = 0.5 * (lo + hi)
            _, clear_m = clearance_at(spawn, speed, mid, azimuth)
            if clear_m is not None and clear_m >= want:
                hi = mid
            else:
                lo = mid
        elev = hi
        land, clear = clearance_at(spawn, speed, elev, azimuth, dt=0.0005)
        if land is None or clear is None:
            return None, f"speed {speed:.1f} m/s never clears the net", None
        if not (-SERVICE_X + 0.4 < land[0] < -0.8):
            return None, (f"speed {speed:.1f} m/s with {clearance:.2f} m net clearance bounces at "
                          f"x={land[0]:.2f}, outside the service box"), None
        azimuth = math.atan2(y_land - spawn[1], spawn[0] - land[0])
    v0 = _launch(speed, azimuth, math.radians(elev))
    land, clear = flight(spawn, v0)
    if land is None or clear is None:
        return None, "no flight solution", None
    if abs(land[1]) >= SINGLES_Y:
        return None, f"aimed bounce y={land[1]:.2f} is outside the singles line", None
    target = np.array([land[0], land[1], BALL_R])
    return v0, target, clear


# ------------------------------------------------------------- serve sampling
def clears_net(spawn, v0, need=0.15, dt=0.002, max_t=2.5):
    """Drag-aware clearance (ball surface over the net box); rejects net clips."""
    p = np.array(spawn, float)
    v = np.array(v0, float)
    t = 0.0
    while t < max_t:
        prev = p
        p, v = fly_step(p, v, dt)
        t += dt
        if prev[0] > 0.0 >= p[0] and p[0] != prev[0]:
            u = (0.0 - prev[0]) / (p[0] - prev[0])
            return bool(prev[2] + u * (p[2] - prev[2]) - 1.07 - BALL_R >= need)
    return False


def first_catchable(spawn, v0, z_enter=0.52, dt=0.002, v_eff=3.6, t_react=0.2, max_t=4.0):
    """First moment the ball is inside the cage window (z <= z_enter) at a point
    the robot can reach from START - the teacher's target.

    Any bounce counts.  Computed from the serve state, so it is a prediction; it
    only has to be good enough for the teacher to arrive (the policy re-plans).
    """
    py, vy_ = np.array(spawn, float), np.array(v0, float)
    t = 0.0
    while t < max_t:
        speed = float(np.linalg.norm(vy_))
        vy_ = vy_ + (-DRAG_K * speed * vy_ + np.array([0.0, 0.0, -9.81])) * dt
        py = py + vy_ * dt
        t += dt
        if py[2] <= BALL_R:
            vn = -vy_[2]
            vy_[2] = vn * bounce_cor(vn)
            vy_[0] *= TANGENT_LOSS
            vy_[1] *= TANGENT_LOSS
            py[2] = BALL_R
        if py[2] <= z_enter:
            d = float(np.hypot(py[0] - START[0], py[1] - START[1]))
            if d <= v_eff * max(t - t_react, 0.0):
                return py[:2].copy(), t
    return None, None


def sample_serve(rng, speed_range, clearance_range, aim=(-4.9, 0.0), radius=(1.5, 1.0)):
    """Fast approximate serve for training resets.

    The first bounce is sampled around `aim` within `radius`, which controls how
    far from the robot the ball comes down (the post-bounce flight lands ~7 m
    past the bounce).  The serve is solved drag-free and rejected unless the
    drag-free arc clears the net generously, so the real (drag affected) ball
    still passes over it.  Microseconds; the exact planner in ros2_bridge is
    used for the real serves.
    """
    v0 = np.array([-10.0, 0.0, 2.0])
    spawn = np.array([COURT_X + 1.0, 0.0, 2.2])
    for _ in range(60):
        bx = float(np.clip(aim[0] + rng.uniform(-radius[0], radius[0]), -6.2, -1.2))
        by = float(np.clip(aim[1] + rng.uniform(-radius[1], radius[1]), -3.6, 3.6))
        side = -1.0 if by >= 0.0 else 1.0          # diagonal: spawn on the other side
        spawn = np.array([COURT_X + rng.uniform(0.2, 1.4),
                          side * rng.uniform(0.3, 3.8),
                          rng.uniform(2.0, 2.6)])
        target = np.array([bx, by, BALL_R])
        speed = rng.uniform(*speed_range)
        dx, dy = target[0] - spawn[0], target[1] - spawn[1]
        L = math.hypot(dx, dy)
        dz = target[2] - spawn[2]
        disc = speed ** 4 - GRAVITY * (GRAVITY * L * L + 2.0 * dz * speed * speed)
        if disc < 0.0:
            continue
        tan_t = (speed * speed - math.sqrt(disc)) / (GRAVITY * L)   # flat arc
        elev = math.atan(tan_t)
        v0 = np.array([speed * math.cos(elev) * dx / L,
                       speed * math.cos(elev) * dy / L,
                       speed * math.sin(elev)])
        if spawn[0] * v0[0] >= 0.0:
            continue
        # net clearance integrated with the real drag: a serve that clips the net
        # dies there, its engine trajectory leaves the analytic model, and that
        # would poison both the teacher target and the labels
        if not clears_net(spawn, v0, rng.uniform(*clearance_range)):
            continue
        return spawn, v0
    return spawn, v0

