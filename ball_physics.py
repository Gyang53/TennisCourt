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
