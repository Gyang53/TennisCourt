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
