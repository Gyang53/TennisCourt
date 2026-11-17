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

The serve solves a drag-affected launch (rho 1.2, Cd 0.6, m 0.057 kg) that passes through a
random legal bounce point in `tf` seconds, arcing it higher until it clears the net box by
10 cm; measured landing error 3-4 cm over ten serves. The bridge applies the same quadratic
drag to the ball every step (the MJCF has no fluid model). The ball bounces off the court
with COR ~0.6 (contact pair `ball`/`ground_col`, `solref="0.0015 0.3"`), which is why the
model runs at `timestep="0.0005"`: at 2 ms a 20 m/s ball moves 50 mm per step, more than its
33.5 mm radius, and tunnels through the court.

Not implemented: the `/mecanum/goal` point controller — this bridge only takes velocity
commands. The wheel feedforward gains were measured on this model
(0.09958 m/s, 0.09712 m/s, 0.18613 rad/s per wheel rad/s) and are trimmed against the
measured body velocity.

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
