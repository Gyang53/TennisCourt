# TennisCourt — mecanum ball-collector robot on a tournament tennis court

MJCF scenes for MotrixSim, generated from the reference mecanum URDF
(`mecanum_tutorial`) plus a standard tennis court built procedurally.

## Scenes

| File | Contents |
|---|---|
| `rally_court.xml` | Standard court (net on x = 0, baselines at x = ±11.885) + robot + tennis ball |
| `tennis_robot.xml` | The robot alone on an infinite plane (starts at its own origin) |

## Robot frame

* origin = centre mark of the robot's own baseline (in `rally_court.xml`: `(-11.885, 0)`)
* `+X` = facing the court net
* `+Y` = along the baseline, to the robot's left
* `+Z` = up

The wheels roll along `±Y`, so moving toward the net is a mecanum **lateral** move and
the catch-cage opening (facing `+X`) is perpendicular to the travel direction.

## Model

* 41 links / 40 joints: 4 wheel hubs (hinge) + 36 rollers (free-spinning hinge each)
* masses: roller 0.1 kg, hub 0.3 kg → 1.2 kg per wheel assembly, 20.86 kg total
* wheel radius 0.10 m, roller ring 0.0794 m, roller 0.0629 × ⌀0.0206 m (URDF geometry, ×2 scale)
* actuators: 4 torque motors, ±12 N·m
* collisions: chassis box, cage (walls/lip/deck/posts), roller sphere chains fitted inside
  the barrel profile, court slab and net box

## ROS2 control and sensors

`ros2_bridge.py` runs the sim and the ROS2 interface in one process, following
`isaacsim/isaac/RallyCourt/rally_ctrl/MECANUM_SENSORS.md`:

```bash
source /opt/ros/jazzy/setup.bash
.venv/bin/python ros2_bridge.py             # viewer window
.venv/bin/python ros2_bridge.py --headless  # no window
```

The robot starts at the court origin (baseline centre, world `-11.885, 0`), camera facing
the net, and holds still until a `cmd_vel` arrives.

| topic | dir | type | meaning |
|---|---|---|---|
| `/mecanum/cmd_vel` | sub | `geometry_msgs/Twist` | vehicle frame: `linear.x` forward (camera axis) m/s, `linear.y` left m/s, `angular.z` CCW rad/s; **0.5 s deadman** |
| `/mecanum/wheel_cmd` | sub | `std_msgs/Float64MultiArray` | raw wheel speed targets `[FL, FR, RL, RR]` rad/s, bypasses `cmd_vel` while fresh (0.5 s). Reserved for direct wheel control / RL |
| `/mecanum/state` | pub | `std_msgs/Float64MultiArray` | `[x, y, theta, vx, vy]`, 120 Hz, same numbers as `/odom` but flat |
| `/imu/data_raw` | pub | `sensor_msgs/Imu` | 6-axis (acc + gyro) at 120 Hz, `frame_id=imu_link`, ROS specific-force convention (rest `z = -9.81`), no orientation (`orientation_covariance[0] = -1`) |
| `/odom` | pub | `nav_msgs/Odometry` | 120 Hz, `odom -> base_link`; only `x`, `y`, `theta` and `vx`, `vy` are filled |
| `/tf` | pub | `tf2_msgs/TFMessage` | `odom -> base_link`, `base_link -> imu_link` |
| `/joint_states` | pub | `sensor_msgs/JointState` | the four wheel joints |
| `/clock` | pub | `rosgraph_msgs/Clock` | simulation time |
| `/rally/serve` | srv | `std_srvs/Trigger` | spawn one legal serve from the opposite half: random spawn behind the far baseline, diagonal into the receiving service box, clears the net, first bounce inside the singles lines |

```bash
ros2 topic echo /imu/data_raw --field linear_acceleration
ros2 topic pub -r 20 /mecanum/cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.4}}"
ros2 topic pub --once /mecanum/cmd_vel geometry_msgs/msg/Twist "{}"   # stop now
ros2 topic echo /mecanum/state                                     # x, y, theta, vx, vy
ros2 topic pub -r 20 /mecanum/wheel_cmd std_msgs/msg/Float64MultiArray "{data: [5.0, 5.0, 5.0, 5.0]}"
```

Frames: `odom` is the startup pose — origin at the robot, +X along the camera axis (toward
the net), +Y left, +Z up — so `theta = 0` at power-on. `base_link` is the **vehicle frame**
(+X forward = camera axis, +Y left, +Z up); the MJCF base body itself is rotated 90 deg from
it (+X left, +Y rearward), which the bridge folds into the published poses, the odometry and
the IMU axes, so `vx` is forward speed and `ax` is forward acceleration. `imu_link` is
coincident with `base_link`.

Wheel targets are wheel angular speed in rad/s, not tread speed: 5 rad/s on all four wheels
drives the body right at ~0.50 m/s (measured gains: body x 0.09958 m/s, y 0.09712 m/s,
yaw 0.18613 rad/s per wheel rad/s).

```bash
ros2 service call /rally/serve std_srvs/srv/Trigger
```

The serve solves a drag-affected launch (rho 1.2, Cd 0.6, m 0.057 kg) whose first bounce
lands in the diagonal receiving service box, inside the singles lines. Two node parameters
shape it (pass floats, `18`, not `18.0`, is rejected by `ros2 param set`):

```bash
ros2 param set /mecanum_drive serve_speed 18.0       # m/s at launch (default 20.0)
ros2 param set /mecanum_drive serve_clearance 0.5    # m over the net box (default 0.45)
ros2 service call /rally/serve std_srvs/srv/Trigger
# success=True, message='spawn (...) -> target (...) speed 18.0 m/s clearance 0.71 m'
```

Speed and clearance are coupled by the geometry: with the toss 2.0-2.6 m high, a 20 m/s serve
must be lofted 6-9 deg to stay inside the service box, so a big clearance forces a steeper,
slower arc. The elevation is bisected on the *aimed* flight, so the requested clearance is
hit to within 0.01 m (20 m/s: 0.45 -> 0.44, 0.80 -> 0.79); a clearance that would push the
bounce past the service line (e.g. 1.6 m at 20 m/s) is rejected with `success=False` instead
of spawning something illegal.

Ball physics the engine does not do for us, all in `ros2_bridge.py`:

* **drag** - quadratic, rho 1.2 / Cd 0.6 / m 0.057 kg, applied every step and inside the
  serve solver so the two agree.
* **bounce** - the ball/floor contact pair in the MJCF is *critically damped on purpose*
  (`solref="0.002 1.0"`): the engine's soft contact lasts ~3 steps at 0.5 ms, so the
  restitution it produced swung between 0.1 and 0.6 depending on the impact phase. The
  bounce is now an explicit impulse, `COR = clip(0.83 - 0.010*v_normal, 0.45, 0.78)`,
  horizontal speed kept at 90 %, and the ball leaves with rolling spin. Checked against the
  ITF rebound test: a 2.54 m drop rebounds 1.35 m (standard 1.35-1.47 m), serve bounces
  apex 1.1-1.5 m.
* **why 0.5 ms** - at 2 ms a 20 m/s ball moves 50 mm per step, more than its 33.5 mm radius,
  and tunnels through the court.

Not implemented: the `/mecanum/goal` point controller — this bridge only takes velocity
commands. The wheel feedforward gains were measured on this model
(0.09958 m/s, 0.09712 m/s, 0.18613 rad/s per wheel rad/s) and are trimmed against the
measured body velocity.

## Catch policy (reinforcement learning)

`train_catch.py` trains a PPO policy to bring the ball into the robot's cage,
using **one batched MotrixSim scene** (`SceneData(model, batch=[N])`), so all
environments step in parallel inside the Rust backend (~40 env-sim-seconds per
wall second at 32-48 envs, measured).

* action: vehicle-frame `(forward, left, yaw rate)`, normalised to `[-1, 1]` and
  mapped through the same wheel IK + torque servo as the ROS bridge
* observation, three selectable modes (`CatchEnv(obs_mode=...)`; the checkpoint's
  input size picks it automatically in `eval_catch.py`):
  - `state` (12): robot offset/velocity/heading + current ball position/velocity
  - `traj` (55): robot state + **the ball's path since the serve**, resampled to 16
    points + elapsed time
  - `both` (61, default): state and trajectory together
* reward: potential shaping on the cage-to-ball distance, `+1` when the ball
  ends up inside the cage, episodes end on catch / ball out of the play area /
  timeout
* curriculum: serve speed, net clearance and the *aim radius* (how far from the
  robot the serve bounces) ramp up over training, ending at the real
  distribution (16-20 m/s, 0.45-0.8 m clearance)

```bash
.venv/bin/python train_catch.py --envs 48 --iters 400     # writes policies/catch_ppo.pt
.venv/bin/python eval_catch.py --episodes 48             # exact legal serves
```

`catch_env.py` holds the vectorised environment; `eval_catch.py` grades a policy
with the same exact serve planner the ROS bridge uses.

### Demo

Watch it live (opens the renderer, ~60 fps, serves back to back, prints each
outcome):

```bash
.venv/bin/python demo_live.py                      # 16-20 m/s, full serve box
.venv/bin/python demo_live.py --speed 12 16 --radius 1.5
.venv/bin/python demo_live.py --episodes 10        # stop after 10 serves
.venv/bin/python demo_live.py --wide               # fixed camera instead of chase cam
```

Headless statistics and a top-down path plot:

```bash
# 32 serves at 16-20 m/s over the full legal box: prints the catch rate and
# writes rl_catch_paths.png (top-down ball/robot paths, catch points)
.venv/bin/python demo_catch.py --episodes 32

# other distributions / the earlier policy
.venv/bin/python demo_catch.py --episodes 32 --speed 12 16 --radius 1.5 \
    --clearance 0.6 1.0 --policy policies/catch_bc.pt

# no rendering (fast)
.venv/bin/python demo_catch.py --episodes 64 --render-seeds
```

`demo_catch.py` only writes statistics and the plot unless `--render-seeds` is
given (`-1` searches for a catching episode first). Both scripts pick the policy's
observation mode from the checkpoint, so `catch_bc.pt` (12-dim) and
`catch_bc_A3.pt` (61-dim) run with the same command.

Feeding the trajectory explicitly (the ball's coordinates from the serve, one
control step at a time, no future information) was measured against the plain
state observation, same seeds and serve distributions:

| policy | 12-16 m/s r0.5 | 12-16 m/s r1.5 | 16-20 m/s r3.0 |
|---|---|---|---|
| `state` 12 dims (128 wide) | 59% | 34% | 0% |
| `traj` 55 dims (128 wide) | 9% | 0% | 0% |
| `both` 61 dims (256 wide) | **62%** | **38%** | 0% |

The trajectory alone is *harder* to learn (the network has to recover the
ballistics from 16 samples; BC fit MSE 0.0099 vs 0.0033 with state included),
and no input representation lifts the 16-20 m/s case, because the teacher that
generates the demonstrations is the ceiling there.

PPO from scratch did not find this task (the catch window is ~0.2 m wide and only
pays out once per episode), so the policy is first distilled from a scripted
teacher and then fine-tuned:

```bash
.venv/bin/python bc_train.py --episodes 400        # policy -> policies/catch_bc.pt
.venv/bin/python train_catch.py --init policies/catch_bc.pt --iters 220 --rollout 128
```

The teacher is PD tracking of the catch point, forward-simulated with the same
drag and bounce model (0.1-0.15 m prediction error); it catches 78% of the easy
serves and 28% at 12-16 m/s, and the cloned policy reproduces 50% / 38% before
any RL fine-tuning.

Measured status (48 envs, `policies/catch_bc.pt`):

| serve | aim radius | catch rate |
|---|---|---|
| 12-16 m/s | (0.5, 0.3) | 50% |
| 12-16 m/s | (1.5, 1.0) | 38% |
| 12-16 m/s | (3.0, 3.0) | 16% |
| 16-20 m/s, exact planner | anywhere legal | ~0% |

The catch window is about ±0.2 m while the ball crosses it at 8 m/s, so this is
a precision task: the scripted teacher itself only manages 28% at 12-16 m/s and
nothing at 16-20 m/s, and no cloned policy can beat its teacher. To push the
full-range number up, in order of impact:

1. shrink the catch-point prediction error below 0.1 m (5 ms forward
   integration is at 0.11-0.14 m; a short-horizon re-plan would close it);
2. widen the physical catchment (a funnel/backboard over the cage mouth raises
   the ceiling faster than any controller work);
### Teacher (done) and its distillation (not working yet)

The scripted teacher was aiming at the wrong event: it insisted on the ball's
descent *after* its first bounce, which only 4-9 of 10 serves ever produce inside
the playable area. With the catch defined as "the first entry into the cage
window (world z 0.10-0.52 m), any bounce, that the robot can reach from its
start" and an inner velocity loop (pure position feedback leaves a 0.24 m
mecanum strafe error at the arrival), the teacher catches

| serve | teacher | teacher before |
|---|---|---|
| 12-16 m/s r0.5 | 92% (22/24) | 0-33% |
| 12-16 m/s r1.5 | 33% (8/24) | 0-33% |
| 16-20 m/s r3.0 | 8% (2/24) | 0% |

Run on the whole batch it is fast: `CatchEnv.set_serves` + `record_catch_points`
(park the robots, record each ball's first reachable entry) then a replay pass
with `teacher_action`. 24 serves in ~3 s.

Distilling it does **not** work, and the reason is structural rather than a
tuning problem: the teacher's label depends on the *recorded engine-truth* catch
point, which is not a function of anything the policy can observe (the trajectory
it sees is produced by the same physics, but the target is the outcome of the
engine's soft-contact bounce). Supervising on it is therefore regression on noise
- BC fit MSE 0.0127 and DAgger 0.44, i.e. close to the label variance - and both
clones catch 0%.

Two ways out, both explicit trade-offs:

1. **self-consistent teacher**: label with the *analytic* target
   (`first_catchable`, computable from the trajectory the policy sees). That teacher
   catches ~31% (its analytic bounce model is 0.1-0.2 m from the engine), and a
   clone can fit it - a ~30% policy that is deployable;
2. **privileged information**: train with the recorded target (or the full future
   trajectory) as an input. Higher ceiling, but the policy then needs that input at
   deployment, i.e. a predictor good enough to supply it.

Solved (plan A): make the target computable from the observation, then the
labels are a function of the input and distillation works.

1. `sample_serve` rejects serves that clip the net (drag-aware check) - a clipped
   serve dies at the net and its engine trajectory leaves the analytic model;
2. with that, the analytic flight matches the engine to **0.05 m** over 2.6 s, so
   `set_serves` can compute the catch target itself (first reachable entry into the
   cage window, any bounce) - no engine recording needed;
3. the teacher wants *smooth* gains: kp 1.5/2.5 without the velocity loop, 96/96/71%
   (the aggressive kp 9 + velocity loop got 12% on fast serves);
4. `set_serves` must initialise the trajectory buffer exactly like `reset()`
   (`traj[:, 0] = spawn`) - otherwise the policy trains on (0,0,0) as the first
   trajectory point and fails on the correct observation at evaluation.

The drive now accelerates along its own spec (fore 3.5, lat 2.5, decel 5.0 m/s^2)
instead of jumping to the commanded speed - that removed the startup hop
(chassis peak 0.1854 -> 0.1584 m) and the policy was retrained on it:

| policy | 12-16 r0.5 | 12-16 r1.5 | 16-20 r3.0 |
|---|---|---|---|
| teacher (scripted, the demos' ceiling) | 100% | 96% | 50% |
| **`policies/catch_bc_big.pt`** (800 serves, 100 epochs, GPU) | **92%** | **100%** | **54%** |
| `catch_bc_spec.pt` (400 serves, 60 epochs) | 81% | 71% | 42% |
| `catch_bc_A3.pt` (instant commands, hops at startup) | 62% | 52% | 50% |
| `catch_bc.pt` (12-dim state obs, shipped before) | 59% | 34% | 0% |

`catch_bc_big.pt` is the policy to use (the demos default to it): 92/100/54% with
physically correct acceleration and no startup hop, and 12/20 = 60% in the live
viewport on the fastest distribution.

Two things do *not* work yet, both measured:

* **PPO fine-tuning** from this policy collapses it (0-2% within 60 iterations)
  even with a critic-only warmup, lr 1e-4 for the actor / 1e-3 for the critic and
  log_std starting at -2.5. The hand-rolled PPO is not trustworthy here; use a
  tested implementation (MotrixLab's SKRL/RSL-RL, or UniLab) rather than more
  tuning of this one.
* **GPU physics**. Training does use the GPU for the policy update (60k samples x
  100 epochs: 7.6 s on the 5060, 1.7x over CPU), which is why the card draws little
  power: the bottleneck is the CPU physics (MotrixSim's backend is Rust/CPU).
  `mujoco-warp` installs and Warp enumerates `cuda:0` (sm_120), but every published
  wheel is built against a newer MuJoCo than the released bindings here - tried
  4.0.1/4.0.1 -> `MjModel.has_flex_snh` missing, 3.10.0.1/3.10.0 -> `nbranch`
  missing. Getting GPU physics therefore means building mujoco + mujoco_warp from
  one source revision (CMake + CUDA), or letting a framework pin the pair for you:
  `uv run --extra mjwarp ... --sim mjwarp` in UniLab is exactly that, and its task
  contract can host this environment.

Training runs on the GPU when one is present (`--device cuda|cpu`, auto-detected;
the RTX 5060 is used at capability 12.0). Note the honest split: the policy update
is 1.7x faster on the GPU (60k samples x 10 epochs: 1.37 s -> 0.80 s), but the
bottleneck is the CPU physics - MotrixSim's backend is Rust/CPU, so a GPU only
helps the optimiser. For GPU physics you would need a CUDA backend (mjwarp,
Genesis, Isaac through UniLab).

### Startup hop

Commanding the full 4.5 m/s lateral speed from rest makes the mecanum rollers slip
and pitches the chassis up 30 mm for ~0.3 s - the visible "jump" at the start of an
episode. `_drive` now ramps the commanded body velocity at 20 m/s^2 for the first
0.5 s of an episode: the hop drops to 8 mm and the catch rates are unchanged
(52 / 65 / 48% at 12-16 r0.5 / r1.5 and 16-20 r3.0, versus 62 / 52 / 50 before, the
spread being serve-sample noise). Ramping for the whole episode instead costs 40
points at the tightest distribution, because the policy needs its braking.

### What was tried on the fast serves (did not pay off yet)

Putting the catch point and its time-to-arrival into the observation (12 -> 15
dims), plus two predictor fixes: the forward integration was re-predicting the
*next* bounce instead of the current descent once the ball had bounced (the
target jumped ~5 m), and the bounce itself was sampled at 50 Hz, so the flag
never saw the contact (now tracked per physics substep).

Measured, same seeds and serve distributions:

| policy | 12-16 m/s r0.5 | 12-16 m/s r1.5 | 16-20 m/s r3.0 |
|---|---|---|---|
| shipped (12-dim obs, old predictor) | **59%** | **34%** | 0% |
| 15-dim obs + fixed predictor | 9% | 16% | 6% |

So the fast serves did come off zero (0 -> 6%), but the easy serves collapsed:
with the corrected predictor the scripted teacher itself only catches 2/24 at
r0.5 (its tracking residual is 0.3-0.4 m while the window is +-0.2 m), so the
cloned policy inherits that ceiling. The shipped policy is the 12-dim one; the
15-dim variant and the predictor fixes are in the history, not in HEAD.

3. only then PPO fine-tuning, with critic pretraining, actor lr 1e-4,
   `log_std` starting at -2.0 and the actor frozen for the first ~20 iterations
   (the naive fine-tune destroyed the cloned policy: 50% -> 1%).

## MotrixLab training (MotrixSim physics + CUDA learner)

The catch task also runs inside MotrixLab, where the physics is MotrixSim's CPU
batch and the PPO learner runs on CUDA.  Policies are quoted on the three serve
tiers of `eval_three_tiers.py`; the fastest one is the 20 m/s band
(16-20 m/s, radius 4.9).

| tool | purpose |
| --- | --- |
| `eval_three_tiers.py --ckpt ... [--ours] [--task ...] [--model xml] [--seed N]` | three-tier reporting protocol |
| `eval_tier.py --ckpt ... --tier 3` | single-tier companion for checkpoint sweeps |
| `view_catch.py --ckpt ... --tier 3` | live viewport: MotrixSim physics, policy on CUDA |

Measured on 256 worlds x 2 rounds, seed 0:

| policy | 12-16 m/s r0.5 | 12-16 m/s r1.5 | 16-20 m/s r4.9 |
| --- | --- | --- | --- |
| `policies/catch_ppo.pt` (reference) | 72% | 52% | 43% |
| 20 m/s fine-tune (best ladder checkpoint) | 51% | 37% | 52% |

What the experiments established:

* more compute does not help: a 10M env-step run on the 20 m/s band collapsed to 0%;
* the collapse is driven by the update size: `learning_rate=5e-4` loses a 39%
  policy within 100 control steps, `1e-5` holds it and gains a few points;
* the workable recipe is a ladder: `learning_rate=3e-5`, `rollouts=128`, short
  stages (~1500 control steps) and checkpoint selection **by evaluation** —
  training reward peaks earlier than catch rate;
* `algo.trainer.env_overrides` sets the training serve band without touching the
  registered task, e.g. `serve_speed_range=[16,20]` and `serve_radius=4.9`.

Starting a stage from a previous checkpoint:

```bash
source ~/data/MotrixLab/.venv/bin/activate
python -m mxlab train +task/tennis_catch=skrl.ppo num_envs=1024 \
  algo.trainer.timesteps=1500 algo.trainer.init_policy=<checkpoint>.pt \
  algo.agent.learning_rate=0.00003 algo.agent.rollouts=128 checkpoint.interval=25 \
  '+algo.trainer.env_overrides.serve_speed_range=[16,20]' \
  '+algo.trainer.env_overrides.serve_radius=4.9'
```

The PPO ladder saturates near 50% on that band, so the same task was then trained
with MotrixLab's off-policy provider:

### FastSAC on the 20 m/s band

`motrix.fastsac` (replay + distributional critic + target smoothing, UTD 8) on
`tennis_catch_wide` — the registered task with the wide serve radius, so training
and evaluation share one distribution. Two settings matter: `gamma` must cover the
4 s episode (0.99, not the 0.97 default) and the update ratio can stay at 8 because
the FastSAC profiler shows the collector blocked on CPU physics while the learner
waits.

512 worlds x 6000 iterations, ~18 minutes, from scratch:

| checkpoint | tier 3 (16-20 m/s, r4.9) |
| --- | --- |
| 250 | 0.0% |
| 1000 | 10.5% |
| 3000 | 88.1% |
| `model_0005750.pt` (full protocol, 256x2) | **90.4%** (889/983) |

For reference, on the same protocol: `catch_ppo.pt` 43%, best PPO ladder
checkpoint 52%.  The FastSAC policy specialises to its training band — the same
checkpoint scores 82.0% on the deployment band (16-20 m/s, r=1.5), 27.5% on tier 2
and 21.8% on tier 1, so quote one policy per band or train on a mixture.

```bash
# train (the recipe lives in configs/task/tennis_catch_wide/motrix.fastsac.yaml)
python -m mxlab train +task/tennis_catch_wide=motrix.fastsac
# evaluate a checkpoint: mxlab play cannot load the fastsac provider
python eval_fastsac.py --ckpt <run>/checkpoints/model_0005750.pt \
  --run-config <run>/task_config.yaml --tier 3 --envs 256 --rounds 2
```

### Yawing: allowed, but bounded

The policy aims the paddle with body yaw, and the free-yaw FastSAC policy spends
most of its time beyond 40 degrees from the spawn heading.  `tennis_catch_yaw`
adds a heading threshold to the reward: inside `yaw_limit` (40 degrees) nothing is
charged, beyond it every radian costs `yaw_penalty` (0.002) per control step, with
the spawn heading read from the state instead of assumed.  Retrained for the same
~19 minutes (512 worlds x 6000 iterations):

| metric (tier 3, 256x2) | free yaw | 40 deg threshold |
| --- | --- | --- |
| mean heading offset | 37.8 deg | **23.1 deg** |
| max heading offset | 56.2 deg | **37.5 deg** |
| steps beyond 40 deg | 56.5% | **0.0%** |
| mean commanded yaw rate | 0.28 | 0.23 |
| catch rate | 90.4% | **91.5%** |

Rotating only when the intercept needs it costs nothing in catch rate here, and the
`yaw_penalty` knob defaults to 0 so the other tasks are unchanged.

### Aiming the cage opening

The cage is open on the body -y side (three 0.36 m walls and a 0.08 m lip), so the
opening has to face the ball, and a free-yaw policy leaves it 26 degrees off on
average.  `facing_scale` pays `cos(angle between the opening and the ball)` every
control step; enabling it at 0.001, tightening the heading threshold to 35 degrees
and retraining once:

| metric (tier 3, 256x2) | yaw threshold only | + aiming term |
| --- | --- | --- |
| mean opening misalignment | 26 deg | **2.5 deg** |
| time within 30 deg | - | **100%** |
| mean / max heading offset | 23.1 / 37.5 deg | **22.2 / 33.5 deg** |
| mean commanded yaw rate | 0.23 | 0.20 |
| catch rate | 91.5% | **93.8%** |

Facing the ball costs no yaw budget: the policy rotates a little less and catches
more.  `facing_scale` defaults to 0, so the other tasks keep their reward.

## Cameras

Two mono cameras migrated from the source `RallyCourt.usd` (LeftCam / RightCam):

* mounted on the cage deck at the opening side, `±0.20 m` either side of the centre line,
  `0.21 m` above the base origin, looking toward the court net
* FOV `57.47°` vertical / `72.34°` horizontal (390 mm lens on a 570.24 mm sensor in the USD)
* render through them with `RenderApp.get_camera(0|1)`

## Court net

Tournament spec: posts 1.07 m, centre strap 0.914 m (0.156 m sag), white top band 0.06 m,
5 cm mesh grid with 8 mm threads, built as meshes by `gen_net.py`.

## Regenerating

```bash
python gen_net.py                            # meshes/net_grid.obj, meshes/net_band.obj
python gen_mjcf.py tennis_robot.xml rally_court.xml
```

`gen_mjcf.py` reads the reference URDF, so it needs
`mecanum_tutorial/src/robot_description/urdf/mecanum_robot.urdf` and the converted
`meshes/urdf_*.obj` (committed here).

## Mesh sources

* `meshes/urdf_*.obj` — converted from the URDF collision STL
  (`mecanum_tutorial/src/robot_description/meshes/STL`)
* `meshes/net_*.obj` — generated by `gen_net.py`
