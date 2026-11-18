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

## SOLVED: mjwarp runs on the GPU with PyPI versions

The earlier "ABI mismatch" was **my API misuse**, not upstream: in current mjwarp
`mjwarp.put_model(mjm)` returns a *wrapped model object*, and `step` must be given
that object - passing MuJoCo's `MjModel` made it look for C fields (`nbranch`,
`has_flex_snh`, ...) that the versions in use legitimately lack.

Working combination, straight from PyPI (no nightly index needed):

```bash
uv pip install --python /home/yangg/data/miniforge3/envs/ms/bin/python \
    "mujoco==4.0.1" "mujoco-warp==4.0.1"
```

```python
mjm = mujoco.MjModel.from_xml_path("rally_court.xml")
m = mjwarp.put_model(mjm)          # <- wrapped model
d = mjwarp.put_data(mjm, mujoco.MjData(mjm), nworld=8192)
mjwarp.step(m, d)                  # <- pass the wrapped model
```

Measured on the RTX 5060 with `rally_court.xml` (42 bodies, 251 geoms, 36 rollers),
500 Hz physics:

| worlds | throughput |
|---|---|
| 1024 | 0.22M env-steps/s |
| 4096 | 0.57M env-steps/s |
| **8192** | **0.67M env-steps/s** (sustained) |

GPU load during the run: **78 W, 95% utilisation** (contrast: the CPU pipeline's
25 W idle-ish card). For reference MotrixSim on the CPU does 79k env-steps/s at
batch 32, so GPU physics is ~8.5x faster at 8192 worlds.

## GPU catch environment (`catch_env_gpu.py`)

Ported the catch environment onto mjwarp: batched worlds, the same observation
layout / action space / reward / catch predicate / teacher as the MotrixSim env, and
physics that are *modelled* rather than hacked - air drag from MuJoCo's fluid model
(`<option density viscosity>`) and the bounce from contact restitution
(`solref`), instead of the bridge-side velocity update and impulse.

Two things needed fixing to get it running:

* the 36 mecanum rollers exceed MuJoCo's default 64 constraints per world - the
  limits are per-world arguments of `mjwarp.put_data` (not MJCF `<size>`):
  `nconmax=256, njmax=1024, naconmax=256*nworld`;
* `put_model` returns a *wrapped* model; `step` must receive that object.

Measured: 512 worlds x 240 control steps in ~53 s (93k physics-steps/s), no
overflow warnings. Raw physics throughput of the same model without the env layer
is 0.67M env-steps/s at 8192 worlds, so the per-control-step numpy<->warp transfers
are the current bottleneck.

**Not calibrated yet**: MuJoCo's contact stiffness/damping gives a much deader ball
than the reference - drop 2.54 m rebounds 0.13 m where the ITF-standard reference
(and the MotrixSim model) gives 1.35 m, and the fluid model's drag is unverified
against the measured serve flight. Until `solref`/`density`/`viscosity` are fitted
to those two measurements the scripted teacher cannot score there (2/512), which in
turn means demo collection on the GPU is not yet usable.

## (Historical) the nightly-index detour

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

### Source build (tried, also blocked)

Built everything from source on the same day (both repos at main, 2026-11-17):

* MuJoCo **C library**: builds fine (cmake 3.28 + gcc, `libmujoco.so.4.0.2` in
  `~/data/dev/mujoco_install`), 41 s on 24 cores;
* `mujoco_warp`: installs fine from source;
* MuJoCo **Python bindings**: the git checkout cannot build them (upstream's own
  CMakeLists comment says the git path is stale/unsupported); the **sdist** path
  works, but `make_sdist.sh` pins `pip==26.2` which does not exist - remove that
  line and it produces `mujoco_core-4.0.2.tar.gz` in seconds, then
  `MUJOCO_PATH=<install> MUJOCO_PLUGIN_PATH=<build>/lib/mujoco_plugin pip install`
  it. `mujoco/_version.py` has to be written and the shell's
  `_EXPECTED_VERSION` bumped to match, after which `import mujoco` works.

The last step still fails, and it is upstream: `mujoco_warp` main asks for
`MjModel.nbranch`, which today's MuJoCo main (`mjmodel.h`) does not define, and the
bindings' committed enum codegen is older than `mjtype.h` (`mjtCtrlInput` missing
from `_enums`). So **mjwarp main runs ahead of mujoco main** and the two cannot be
paired from HEAD; a full source build needs the exact MuJoCo commit mjwarp's CI
pins.

### The exact pairing (found) and the network wall

`mujoco_warp`'s own `uv.lock` pins `mujoco-core==4.0.2.dev<build>` from
**`https://py.mujoco.org/`** - their nightly index, built from MuJoCo `main` the
same day. That is the only source of a matched pair; PyPI's `mujoco-core` stops at
4.0.1. Exact artifacts (build `4.0.2.dev1024142490`):

```
https://py.mujoco.org/mujoco-core/mujoco_core-4.0.2.dev1024142490-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl   (9.76 MB)
https://py.mujoco.org/mujoco/mujoco-4.0.2.dev1024142490-py3-none-any.whl                                                       (2 MB)
```

Install them with `--no-deps` (the shell hard-depends on `mujoco-studio==<same
dev version>`, which the nightly index does not serve):

```bash
uv pip install --python /home/yangg/data/miniforge3/envs/ms/bin/python --no-deps <core.whl> <shell.whl>
```

`mujoco_warp` 4.0.1 is already built from source in `ms`, so after those two wheels
the GPU path should light up. **It is blocked only by download speed on this
machine**: py.mujoco.org delivers ~2 KB/s here (0 B/s for long stretches) while
PyPI gives ~25 KB/s, so the 9.76 MB wheel does not complete. A resumable downloader
is running in `/tmp/mjnightly/` (state file `state` appears when finished); fetch
the two URLs on a faster link and the install is one command.

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
