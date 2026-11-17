"""Generate the mecanum chassis MJCF straight from the reference URDF.

Source of truth: /home/yangg/data/mecanum_tutorial/src/robot_description/urdf/mecanum_robot.urdf
  * every joint axis is "0 0 1" in its child frame
  * roller long axis == its local Z, mesh carries rpy "0 -pi/2 0"
  * ring radius 0.0397 m, roller 0.0629 x diameter 0.0206 m, wheel radius 0.05 m
The scene is authored at SCALE x (the RallyCourt USD instance is 2x the URDF), so all
translations and mesh assets are scaled together.

Only the deck + catch net are synthetic; the whole chassis/wheels/rollers come from
the URDF meshes and joint origins.
"""
import math
import os
import sys
import xml.etree.ElementTree as ET

URDF = "/home/yangg/data/mecanum_tutorial/src/robot_description/urdf/mecanum_robot.urdf"
MESH_ROOT = "/home/yangg/data/mecanum_tutorial/src/robot_description"
HERE = os.path.dirname(os.path.abspath(__file__))
MESH_OUT = os.path.join(HERE, "meshes")

SCALE = 2.0                      # URDF -> RallyCourt robot size
G = 9.81

# deck + net (synthetic, in scaled metres)
CAGE_S = 0.8                     # top cage scaled to 0.8 (as edited in Blender)
DECK_HALF = (0.275, 0.275, 0.01 * CAGE_S)   # deck plan size unchanged, thickness x0.8
DECK_Z = 0.107                   # deck plate sits on the URDF base plate
DECK_TOP = DECK_Z + DECK_HALF[2]
NET_HALF = 0.26 * CAGE_S         # cage half width (plan)
NET_WALL_H = 0.45 * CAGE_S       # wall height
LIP_H = 0.20 * CAGE_S            # front lip height
WALL_T = 0.03 * CAGE_S           # wall half thickness
POST_R = 0.01 * CAGE_S           # post radius
POST_BASE_Z = 0.015              # chassis top: posts come down onto the mecanum chassis
POST_TOP_Z = DECK_TOP + NET_WALL_H

# mono camera pair migrated from RallyCourt.usd (LeftCam / RightCam):
# 0.27 m either side of the robot's centre line, 0.21 m above the base origin,
# FOV 72.34 x 57.47 deg (lens 390 mm on a 570.24 mm sensor), clip start 0.01 m
CAM_LATERAL = 0.20               # +-X in the base frame (0.27 in the USD, kept clear of the cage wall)
CAM_FORWARD = -0.19              # on the deck at the cage opening side (-Y = facing the net)
CAM_HEIGHT = 0.21
CAM_FOVY = 57.47                 # vertical FOV in degrees (horizontal 72.34)
CAM_XYAXES = "-1 0 0 0 0 1"      # camera -Z along base -Y (toward the court net), +Y up
MASS_DECK = 6.0
ROLLER_MASS = 0.1        # per roller
WHEEL_HUB_MASS = 0.3     # hub, so hub + 9 rollers = 1.2 kg per wheel
BALL_R = 0.0335          # tennis ball
BALL_MASS = 0.057
COURT_HALF = (11.885, 5.485)   # 23.77 x 10.97 m
NET_X = 2 * COURT_HALF[0]      # net line, 11.885 m from the robot's baseline
NET_HALF_Y = 6.4
NET_H = 1.07
ROBOT_START = (-COURT_HALF[0], 0.0)   # centre mark of its own baseline = robot frame origin
ROBOT_YAW = math.pi / 2       # facing +X (the net); wheels roll along Y (baseline)

# (offset along the roller axis, radius) sampled from the URDF roller mesh at
# x/half_len = -0.9, -0.45, 0, +0.45, +0.9, scaled to the scene (x2)
ROLLER_SPHERES = ((-0.0566, 0.0077), (-0.0283, 0.0186), (0.0, 0.0206),
                  (0.0283, 0.0186), (0.0566, 0.0077))


def parse_urdf():
    tree = ET.parse(URDF)
    robot = tree.getroot()
    links = {}
    for link in robot.findall("link"):
        name = link.get("name")
        data = {"name": name}
        vis = link.find("visual")
        if vis is not None:
            geo = vis.find("geometry")
            mesh = None if geo is None else geo.find("mesh")
            if mesh is not None:
                o = vis.find("origin")
                data["mesh"] = mesh.get("filename").replace("package://robot_description/", "")
                data["vis_rpy"] = [float(x) for x in (o.get("rpy").split() if o is not None and o.get("rpy") else [0, 0, 0])]
                data["vis_xyz"] = [float(x) for x in (o.get("xyz").split() if o is not None and o.get("xyz") else [0, 0, 0])]
        ine = link.find("inertial")
        if ine is not None:
            data["mass"] = float(ine.find("mass").get("value"))
            i = ine.find("inertia")
            data["inertia"] = [float(i.get(k)) for k in ("ixx", "iyy", "izz")]
        links[name] = data
    joints = {}
    for j in robot.findall("joint"):
        o = j.find("origin")
        xyz = [float(x) for x in (o.get("xyz").split() if o is not None and o.get("xyz") else [0, 0, 0])]
        rpy = [float(x) for x in (o.get("rpy").split() if o is not None and o.get("rpy") else [0, 0, 0])]
        ax = j.find("axis")
        axis = [float(x) for x in ax.get("xyz").split()] if ax is not None else [0, 0, 1]
        joints[j.get("name")] = {"parent": j.find("parent").get("link"), "child": j.find("child").get("link"),
                                 "xyz": xyz, "rpy": rpy, "axis": axis,
                                 "type": j.get("type")}
    return links, joints


def rpy_to_quat(r, p, y):
    cr, sr = math.cos(r / 2), math.sin(r / 2)
    cp, sp = math.cos(p / 2), math.sin(p / 2)
    cy, sy = math.cos(y / 2), math.sin(y / 2)
    return [cy * cp * cr + sy * sp * sr,
            cy * cp * sr - sy * sp * cr,
            cy * sp * cr + sy * cp * sr,
            sy * cp * cr - cy * sp * sr]


def fmt(v):
    return " ".join(f"{x:.6g}" for x in v)


def main(out_path, court=None):
    links, joints = parse_urdf()
    if court is None:
        court = "rally_court" in os.path.basename(out_path)
    robot_start = ROBOT_START if court else (0.0, 0.0)
    os.makedirs(MESH_OUT, exist_ok=True)
    mesh_files = {}          # unique asset name -> relative path
    for d in links.values():
        if "mesh" not in d:
            continue
        src = os.path.join(MESH_ROOT, d["mesh"])
        if not os.path.exists(src):
            raise SystemExit(f"missing mesh {src}")
        stem = os.path.splitext(os.path.basename(src))[0]
        # Forge/Blender mesh import takes OBJ/STL/GLB (not DAE); the URDF ships the same
        # geometry as collision STL, converted once to meshes/urdf_<stem>.obj.
        dst = os.path.join(MESH_OUT, f"urdf_{stem}.obj")
        if not os.path.exists(dst):
            raise SystemExit(f"missing converted mesh {dst} (convert the URDF STL first)")
        mesh_files[stem] = os.path.relpath(dst, HERE)
    # roller collider geometry comes from the URDF barrel profile (see ROLLER_SPHERES)

    L = []
    a = L.append
    a('<mujoco model="rallycourt_mecanum_robot">')
    a('  <compiler angle="radian" autolimits="true"/>')
    a('  <option timestep="0.002" iterations="120" tolerance="1e-8" gravity="0 0 -9.81"/>')
    a('  <asset><texture type="skybox" builtin="gradient" rgb1="0.42 0.55 0.72"'
      ' rgb2="0.86 0.88 0.92" width="512" height="512"/></asset>')
    a('  <default>')
    a('    <geom friction="1.0 0.01 0.001" solref="0.004 1"/>')
    a('  </default>')
    a('  <asset>')
    for name, path in mesh_files.items():
        # roller_col is already authored at the scaled size
        scale = '' if name == 'roller_col' else f' scale="{SCALE} {SCALE} {SCALE}"'
        a(f'    <mesh name="{name}" file="{path}"{scale}/>')
    if court:
        a('    <mesh name="net_grid" file="meshes/net_grid.obj"/>')
        a('    <mesh name="net_band" file="meshes/net_band.obj"/>')
    a('  </asset>')
    a('  <worldbody>')
    a('    <light name="sun" pos="0 0 6" dir="0.45 0.35 -0.82" directional="true"'
      ' diffuse="0.9 0.9 0.9" ambient="0.3 0.3 0.32"/>')
    if court:
        # standard court frame: net on x = 0, baselines at x = +-11.885, +Y = robot's left
        # court and surround are visuals only; the green sits 1 mm lower so the two
        # coplanar tops do not z-fight
        a(f'    <geom name="court" type="box" size="{COURT_HALF[0]} {COURT_HALF[1]} 0.02"'
          ' pos="0 0 -0.02" rgba="0.16 0.32 0.52 1" contype="0" conaffinity="0"/>')
        a(f'    <geom name="ground" type="box" size="{COURT_HALF[0] + 3} {COURT_HALF[1] + 3} 0.02"'
          ' pos="0 0 -0.021" rgba="0.12 0.30 0.16 1" contype="0" conaffinity="0"/>')
        # one flat collider under the whole play area: a seam at the court edge (exactly
        # where the baseline is drawn) would kick the wheels as they cross it
        a(f'    <geom name="ground_col" type="box" size="{COURT_HALF[0] + 3} {COURT_HALF[1] + 3} 0.02"'
          ' pos="0 0 -0.02" mass="0" rgba="0 0 0 0" condim="3" friction="1.2 0.01 0.001"'
          ' contype="1" conaffinity="5"/>')
        a('    <geom name="net_grid" type="mesh" mesh="net_grid" mass="0"'
          ' contype="0" conaffinity="0" rgba="0.12 0.12 0.13 1"/>')
        a('    <geom name="net_band" type="mesh" mesh="net_band" mass="0"'
          ' contype="0" conaffinity="0" rgba="0.97 0.97 0.97 1"/>')
        a(f'    <geom name="net_col" type="box" size="0.006 {NET_HALF_Y} {NET_H / 2}"'
          f' pos="0 0 {NET_H / 2}" mass="0" contype="32" conaffinity="2" group="3"/>')
        for sy in (-1, 1):
            a(f'    <geom name="net_post_{"p" if sy > 0 else "n"}" type="capsule"'
              f' size="0.02 {NET_H / 2}" pos="0 {sy * NET_HALF_Y:.3f} {NET_H / 2}"'
              ' rgba="0.15 0.15 0.16 1" contype="32" conaffinity="2"/>')
        # court lines (visual only). ITF Rule 1: lines 5 cm wide, the centre mark
        # 10 cm x 5 cm drawn inside the court, perpendicular to and touching the
        # baseline. MJCF box sizes are half-extents, so the line half-width is LW/2.
        LW, lz = 0.05, 0.001
        lw = LW / 2
        cx, cy = COURT_HALF
        singles_y = 4.115
        service_x = 6.40
        for name, size, pos in (
            ("line_sideline_d_pos", (cx, lw, 0.001), (0.0, cy, lz)),
            ("line_sideline_d_neg", (cx, lw, 0.001), (0.0, -cy, lz)),
            ("line_sideline_s_pos", (cx, lw, 0.001), (0.0, singles_y, lz)),
            ("line_sideline_s_neg", (cx, lw, 0.001), (0.0, -singles_y, lz)),
            ("line_baseline_pos", (lw, cy, 0.001), (cx, 0.0, lz)),
            ("line_baseline_neg", (lw, cy, 0.001), (-cx, 0.0, lz)),
            # ITF Rule 1: the service lines are drawn between the singles sidelines
            ("line_service_pos", (lw, singles_y, 0.001), (service_x, 0.0, lz)),
            ("line_service_neg", (lw, singles_y, 0.001), (-service_x, 0.0, lz)),
            ("line_centre", (service_x, lw, 0.001), (0.0, 0.0, lz)),
            ("line_mark_pos", (0.05, 0.025, 0.001), (cx - lw - 0.05, 0.0, lz)),
            ("line_mark_neg", (0.05, 0.025, 0.001), (-cx + lw + 0.05, 0.0, lz)),
        ):
            a(f'    <geom name="{name}" type="box" size="{fmt(size)}" pos="{fmt(pos)}"'
              ' rgba="0.95 0.95 0.95 1" contype="0" conaffinity="0"/>')

    else:
        a('    <geom name="floor" type="plane" size="0 0 0.05" rgba="0.18 0.35 0.22 1" condim="3"'
          ' friction="1.2 0.01 0.001" contype="1" conaffinity="5"/>')

    # base height: tread radius minus the wheel joint's local z
    wheel_joint = joints["FL_wheel_joint"]
    wheel_r = 0.05 * SCALE
    base_z = wheel_r - wheel_joint["xyz"][2] * SCALE

    yaw = ROBOT_YAW if court else 0.0
    q_body = rpy_to_quat(0.0, 0.0, yaw)
    a(f'    <body name="base" pos="{robot_start[0]:.3f} {robot_start[1]:.3f} {base_z:.4f}"'
      f' quat="{fmt(q_body)}">')
    a('      <freejoint name="base_free"/>')
    a('      <site name="imu" pos="0 0 0" size="0.01" rgba="1 0.4 0 1" group="4"/>')
    b = links["base_link"]
    m_base = b["mass"] * SCALE ** 3
    bw, bh, bd = 0.30 * SCALE, 0.28 * SCALE, 0.30            # plate + deck envelope
    ix = m_base * (bh ** 2 + bd ** 2) / 12.0
    iy = m_base * (bw ** 2 + bd ** 2) / 12.0
    iz = m_base * (bw ** 2 + bh ** 2) / 12.0
    a(f'      <inertial pos="0 0 {0.06 * SCALE:.3f}" mass="{m_base:.4f}"'
      f' diaginertia="{ix:.6g} {iy:.6g} {iz:.6g}"/>')
    q = rpy_to_quat(*b.get("vis_rpy", [0, 0, 0]))
    a(f'      <geom name="base_vis" type="mesh" mesh="base" quat="{fmt(q)}" mass="0"'
      ' contype="0" conaffinity="0" rgba="0.78 0.79 0.82 1"/>')
    a(f'      <geom name="base_col" type="box" size="{0.15*SCALE:.4f} {0.14*SCALE:.4f} {0.03*SCALE:.4f}"'
      f' pos="0 0 {-0.023*SCALE:.4f}" mass="0" rgba="0.7 0.7 0.72 1"'
      ' contype="8" conaffinity="1" group="3"/>')
    # deck + net cage (synthetic)
    a(f'      <geom name="deck" type="box" size="{fmt(DECK_HALF)}" pos="0 0 {DECK_Z:.3f}"'
      ' rgba="0.85 0.85 0.88 1" contype="16" conaffinity="1"/>')
    top = DECK_TOP + NET_WALL_H / 2
    # wheels roll along X, so the cage opening faces -Y: opening perpendicular to
    # the mecanum travel direction (robot drives sideways toward the court net)
    for name, pos, size in (
        ("net_wall_py", (0.0, NET_HALF, top), (NET_HALF, WALL_T, NET_WALL_H / 2)),
        ("net_wall_px", (NET_HALF, 0.0, top), (WALL_T, NET_HALF, NET_WALL_H / 2)),
        ("net_wall_nx", (-NET_HALF, 0.0, top), (WALL_T, NET_HALF, NET_WALL_H / 2)),
    ):
        a(f'      <geom name="{name}" type="box" size="{fmt(size)}" pos="{fmt(pos)}"'
          ' rgba="0.42 0.45 0.50 0.55" contype="16" conaffinity="1"/>')
    a(f'      <geom name="net_lip" type="box" size="{NET_HALF:.3f} {WALL_T:.3f} {LIP_H/2:.3f}"'
      f' pos="0 {-NET_HALF:.3f} {DECK_TOP + LIP_H/2:.3f}" rgba="0.42 0.45 0.50 0.55"'
      ' contype="16" conaffinity="1"/>')
    for name, cx in (("LeftCam", CAM_LATERAL), ("RightCam", -CAM_LATERAL)):
        a(f'      <camera name="{name}" pos="{cx:.3f} {CAM_FORWARD:.3f} {CAM_HEIGHT:.3f}"'
          f' xyaxes="{CAM_XYAXES}" fovy="{CAM_FOVY}"/>')
    for sx in (-1, 1):
        for sy in (-1, 1):
            a(f'      <geom name="post_{"p" if sx>0 else "n"}x{"p" if sy>0 else "n"}y" type="capsule"'
              f' size="{POST_R:.3f} {(POST_TOP_Z - POST_BASE_Z) / 2:.3f}"'
              f' pos="{sx*(NET_HALF-POST_R):.3f} {sy*(NET_HALF-POST_R):.3f}'
              f' {(POST_TOP_Z + POST_BASE_Z) / 2:.3f}" rgba="0.9 0.9 0.92 1"'
              ' contype="16" conaffinity="1"/>')

    roller_links = [n for n in links if "roller" in n]
    for prefix in ("FL", "FR", "RL", "RR"):
        wj = joints[f"{prefix}_wheel_joint"]
        wl = links[f"{prefix}_wheel_link"]
        pos = [v * SCALE for v in wj["xyz"]]
        q = rpy_to_quat(*wj["rpy"])
        a(f'      <body name="{prefix}_wheel" pos="{fmt(pos)}" quat="{fmt(q)}">')
        a(f'        <joint name="{prefix}_wheel_joint" type="hinge" axis="{fmt(wj["axis"])}"'
          ' damping="0.02" armature="0.005" frictionloss="0.005"/>')
        m_w = WHEEL_HUB_MASS
        w_r, w_l = 0.05 * SCALE, 0.0553 * SCALE
        w_ax = 0.5 * m_w * w_r ** 2
        w_tr = m_w * (3 * w_r ** 2 + w_l ** 2) / 12.0
        a(f'        <inertial pos="0 0 0" mass="{m_w:.4f}"'
          f' diaginertia="{w_tr:.6g} {w_tr:.6g} {w_ax:.6g}"/>')
        vq = rpy_to_quat(*wl.get("vis_rpy", [0, 0, 0]))
        a(f'        <geom name="{prefix}_hub_vis" type="mesh" mesh="mecanum_wheel" quat="{fmt(vq)}"'
          ' mass="0" contype="0" conaffinity="0" rgba="0.16 0.16 0.18 1"/>')
        for rl_name in [n for n in roller_links if n.startswith("r") and f"_{prefix}_" in n]:
            rj = joints[f"{rl_name.replace('_link', '_joint')}"]
            rl = links[rl_name]
            rq = rpy_to_quat(*rj["rpy"])
            a(f'        <body name="{rl_name}" pos="{fmt([v * SCALE for v in rj["xyz"]])}" quat="{fmt(rq)}">')
            a(f'          <joint name="{rl_name.replace("_link", "_joint")}" type="hinge"'
              ' axis="0 0 1" damping="5e-5" armature="1e-5" frictionloss="1e-6"/>')
            mq = rpy_to_quat(*rl.get("vis_rpy", [0, 0, 0]))
            m_r = ROLLER_MASS
            r_r, r_l = 0.0103 * SCALE, 0.0629 * SCALE
            r_ax = 0.5 * m_r * r_r ** 2
            r_tr = m_r * (3 * r_r ** 2 + r_l ** 2) / 12.0
            a(f'          <inertial pos="0 0 0" mass="{m_r:.5f}"'
              f' diaginertia="{r_tr:.6g} {r_tr:.6g} {r_ax:.6g}"/>')
            a(f'          <geom name="{rl_name}_vis" type="mesh" mesh="roller" quat="{fmt(mq)}"'
              ' mass="0" contype="0" conaffinity="0" rgba="0.55 0.57 0.62 1"/>')
            # collider: 5 spheres along the roller axis, radii taken from the URDF
            # barrel profile so they sit just inside the true tread (a capsule cannot:
            # its end caps bulge past the 45-deg tilted tread circle).
            for k, (t, rad) in enumerate(ROLLER_SPHERES):
                a(f'          <geom name="{rl_name}_col{k}" type="sphere" size="{rad:.5f}"'
                  f' pos="0 0 {t:.5f}" mass="0" condim="3" friction="1.4 0.01 0.001"'
                  ' contype="4" conaffinity="2" group="3"/>')
            a('        </body>')
        a('      </body>')
    a('    </body>')
    if court:
        ib = 0.4 * BALL_MASS * BALL_R ** 2
        a('    <body name="ball" pos="-5.000 0 0.600">')
        a('      <freejoint name="ball_free"/>')
        a(f'      <inertial pos="0 0 0" mass="{BALL_MASS}"'
          f' diaginertia="{ib:.6g} {ib:.6g} {ib:.6g}"/>')
        a(f'      <geom name="ball" type="sphere" size="{BALL_R}" mass="0"'
          ' friction="0.8 0.01 0.001" solref="0.003 1"'
          ' contype="2" conaffinity="29" rgba="0.85 0.82 0.25 1"/>')
        a('    </body>')
    a('  </worldbody>')
    a('  <actuator>')
    for p in ("FL", "FR", "RL", "RR"):
        a(f'    <motor name="{p}_drive" joint="{p}_wheel_joint" gear="1"'
          ' ctrlrange="-12 12" forcerange="-12 12"/>')
    a('  </actuator>')
    a('  <sensor>')
    for p in ("FL", "FR", "RL", "RR"):
        a(f'    <jointvel joint="{p}_wheel_joint" name="{p}_wheel_vel"/>')
    a('    <framepos objtype="body" objname="base" name="base_pos"/>')
    a('    <framequat objtype="body" objname="base" name="base_quat"/>')
    a('    <framelinvel objtype="body" objname="base" name="base_linvel"/>')
    a('    <accelerometer site="imu" name="imu_acc"/>')
    a('    <gyro site="imu" name="imu_gyro"/>')
    a('    <frameangvel objtype="site" objname="imu" name="imu_angvel_world"/>')
    a('  </sensor>')
    a('</mujoco>')
    xml = "\n".join(L) + "\n"
    with open(out_path, "w") as f:
        f.write(xml)
    print(f"wrote {out_path}: {len(roller_links)} rollers, scale {SCALE}, court={court}")


if __name__ == "__main__":
    outs = sys.argv[1:] or [os.path.join(HERE, "tennis_robot.xml")]
    for out in outs:
        main(out)
