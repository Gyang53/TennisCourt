# UniLab integration notes

Status as of this session. The catch policy itself is unaffected — see the main
README; this file records what was tried for GPU training.

## What is installed

* `~/data/dev/UniLab` — upstream clone (unilabsim/UniLab, `v`-agnostic main).
* Conda env **`ms`** (`/home/yangg/data/miniforge3/envs/ms`) now holds UniLab's
  locked set via
  `UV_PROJECT_ENVIRONMENT=/home/yangg/data/miniforge3/envs/ms uv sync --extra mjwarp`:
  `unilab 1.5.11`, `mujoco-warp 3.11.0`, `warp-lang 1.16.0`, `mujoco 4.0.1`.
* CLI works (call it by full path, `eval`/`train` are shell builtins):

  ```bash
  /home/yangg/data/miniforge3/envs/ms/bin/train --help
  /home/yangg/data/miniforge3/envs/ms/bin/eval  --help
  ```

  Backends registered: `mujoco, mjwarp, genesis, newton, motrix, superdex, drake,
  isaacsim, isaacgym`. The CPU ones need their own extras
  (`unilab[mujoco]`, `unilab[motrix]`), i.e. `uv sync --extra mujoco --extra motrix`.

## Hard blocker: mjwarp wheels vs released MuJoCo

GPU physics needs `mujoco_warp.put_model`, which fails on a missing C struct field
for **every published wheel/mujoco combination** tried:

| mujoco-warp | mujoco | error |
|---|---|---|
| 4.0.1 | 4.0.1 | `MjModel.has_flex_snh` missing |
| 3.10.0.1 | 3.10.0 | `MjModel.nbranch` missing |
| 3.11.0 | 3.11.0 | `MjModel.nbranch` missing |
| 3.11.0 | 4.0.1 | `MjModel.nflexevpair` missing |

Each error is a *different* field, so the wheels are built against MuJoCo `main`,
newer than any published `mujoco` wheel (`mujoco-core` on PyPI only ships 4.0.0 /
4.0.1). Warp itself is fine: it enumerates `cuda:0`, RTX 5060, sm_120.

Two ways forward, both outside pip-only setup:

1. **Docker**: use UniLab's `Dockerfile`, which pins the source-built
   mujoco/mjwarp pair (their CI builds both from one revision).
2. **Source build**: clone `google-deepmind/mujoco` and `mujoco_warp` at the same
   revision, `pip install .` for both into `ms` (needs CMake + CUDA toolchain),
   then `eval --sim mjwarp`.

Also note the environment trap found on the way: a stale
`~/.local/lib/python3.12/site-packages/mujoco*` shadowed the conda env's MuJoCo
(user site beats conda site). It was removed; keep it that way or run with
`PYTHONNOUSERSITE=1`.

## Porting the catch task (planned, not started)

UniLab tasks are manager terms plus a Hydra config per sim:

```
src/unilab/tasks/catch/mecanum/manager_terms.py   # observation / reward /
                                                  # termination / action terms
src/unilab/conf/ppo/task/mecanum_catch/motrix.yaml
src/unilab/conf/ppo/task/mecanum_catch/mjwarp.yaml
assets/mecanum/{rally_court.xml, tennis_robot.xml, meshes/}
```

Physics simplifications UniLab's backends allow, and that the current bridge hacks
should be replaced by:

* **air drag**: MuJoCo `<option density="1.2" viscosity="1.8e-5">` (real fluid
  model) instead of the per-step velocity update in `ball_physics.py`;
* **bounce**: MuJoCo contact restitution (`solref` damping) is proper physics, so
  the explicit impulse in `ros2_bridge._bounce` can go;
* the wheel velocity servo and the catch predicate become action/termination terms.

That also fixes the label-consistency problem we hit: with real engine physics the
analytic trajectory model and the engine agree, so a scripted teacher's target is a
function of the observation and BC/DAgger can fit it.
