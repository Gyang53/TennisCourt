"""Drive probe: does a GPU physics engine actually move this mecanum base?

The mecanum model produces zero traction in mujoco-warp 4.0.1 across every roller
collision variant tried, while CPU MuJoCo and MotrixSim both drive it - so this
probe is the acceptance test for any candidate GPU engine: command the wheels
forward for two seconds and compare the base displacement with the reference
(5.27 m on MotrixSim, 6.4 m on CPU MuJoCo).

    python drive_probe.py --engine mjwarp --model <scene.xml> --worlds 16
    python drive_probe.py --engine mujoco --model <scene.xml>
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np


def load_scene(path: pathlib.Path):
    import mujoco

    xml_dir = path.parent
    assets = {
        str(p.relative_to(xml_dir)): p.read_bytes()
        for p in xml_dir.rglob("*")
        if p.is_file() and p.suffix in {".obj", ".stl", ".png", ".jpg"}
    }
    return mujoco.MjModel.from_xml_string(path.read_text(), assets=assets)


def wheel_dofs(mjm) -> list[int]:
    import mujoco

    names = [mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(mjm.njnt)]
    return [int(mjm.jnt_dofadr[names.index(f"{p}_wheel_joint")]) for p in ("FL", "FR", "RL", "RR")]


def run_mujoco(mjm, seconds: float = 2.0) -> np.ndarray:
    import mujoco

    data = mujoco.MjData(mjm)
    for _ in range(int(0.2 / mjm.opt.timestep)):     # settle
        mujoco.mj_step(mjm, data)
    start = data.qpos[:2].copy()
    for _ in range(int(seconds / mjm.opt.timestep)):
        data.ctrl[:] = [-41.2, 41.2, 41.2, -41.2]     # full forward (velocity servos)
        mujoco.mj_step(mjm, data)
    return data.qpos[:2] - start


def _qpos(data) -> np.ndarray:
    try:
        import warp as wp

        return np.array(wp.to_torch(data.qpos).cpu().numpy())
    except Exception:
        return np.array(data.qpos.numpy())


def run_mjwarp(mjm, worlds: int, seconds: float = 2.0) -> np.ndarray:
    import mujoco
    import mujoco_warp as mjwarp

    model = mjwarp.put_model(mjm)
    data = mjwarp.put_data(
        mjm,
        mujoco.MjData(mjm),
        nworld=worlds,
        nconmax=128,
        njmax=384,
        naconmax=96 * max(worlds, 8),
    )
    ctrl = np.zeros((worlds, mjm.nu), np.float32)
    ctrl[:] = [-41.2, 41.2, 41.2, -41.2]
    for _ in range(int(0.2 / mjm.opt.timestep)):          # settle
        data.ctrl.assign(ctrl)
        mjwarp.step(model, data)
    start = _qpos(data)[0, :2].copy()
    for _ in range(int(seconds / mjm.opt.timestep)):
        data.ctrl.assign(ctrl)
        mjwarp.step(model, data)
    return _qpos(data)[0, :2] - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=pathlib.Path)
    parser.add_argument("--engine", choices=("mujoco", "mjwarp"), default="mjwarp")
    parser.add_argument("--worlds", type=int, default=16)
    parser.add_argument("--seconds", type=float, default=2.0)
    args = parser.parse_args()

    mjm = load_scene(args.model)
    print(f"scene {args.model.name}: {mjm.nbody} bodies, {mjm.ngeom} geoms, nv {mjm.nv}")
    wheels = wheel_dofs(mjm)
    print(f"wheel dofs {wheels}")
    if args.engine == "mujoco":
        delta = run_mujoco(mjm, args.seconds)
        print(f"RESULT mujoco  base {np.round(delta, 2).tolist()} m")
        return
    delta = run_mjwarp(mjm, args.worlds, args.seconds)
    print(f"RESULT mjwarp  base {np.round(delta, 2).tolist()} m")


if __name__ == "__main__":
    main()
