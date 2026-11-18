# 交接状态（compact 后续跑从这里开始）

日期：2026-11-18

## 仓库 / 远程

* 代码：`/home/yangg/data/tennis_robot`（git），远程 `ssh://ssh.github.com:443/Gyang53/TennisCourt.git`，分支 `main`
* 最新提交：`2955243`（GPU 物理标定）。全部改动已推送

## 环境

| 用途 | 怎么用 |
|---|---|
| MotrixSim（CPU 物理 / ROS2） | `tennis_robot/.venv`（motrixsim 0.10.1 + pyyaml；torch 用系统 `/usr/bin/python3.12` 的 2.11+cu128，经 `PYTHONPATH=.venv/lib/python3.12/site-packages` 组合）|
| MuJoCo Warp（GPU 物理） | conda `ms`：`/home/yangg/data/miniforge3/envs/ms/bin/python`，`mujoco 4.0.1` + `mujoco-warp 4.0.1`（PyPI 即可，**不需要 nightly**）|
| UniLab | 已装进 `ms`（`unilab 1.5.11`，`ms/bin/{train,eval,demo}`，8 后端）——**用户已改主意，训练优先用 MotrixLab** |
| **MotrixLab** | `/home/yangg/data/MotrixLab`，`mxlab 0.4.0`，venv：`source /home/yangg/data/MotrixLab/.venv/bin/activate` |

## 已交付且可用（CPU 路径，全部验证过）

* 模型生成 `gen_mjcf.py` → `rally_court.xml` / `tennis_robot.xml`（42 body、251 geom、36 辊子、dt=0.0005）
* ROS2 桥接 `ros2_bridge.py`：`/mecanum/cmd_vel`（车辆系，0.5 s deadman）、`/imu/data_raw`（6 轴，静止 -9.81）、`/odom`（x,y,theta,vx,vy）、`/joint_states`、`/clock`、`/tf`、`/mecanum/wheel_cmd`（轮速直控）、`/rally/serve`（合法发球，速度/过网高度可参数化）
* 球物理 `ball_physics.py`：阻力 `DRAG_K=0.0223`、弹跳显式冲量（ITF 2.54 m→1.35 m 已校准）、`sample_serve`/`first_catchable`/`clears_net`
* 训练：`catch_env.py`（MotrixSim 批量 env，`obs_mode`=state12/traj55/both61）、`bc_train.py`、`train_catch.py`、`eval_catch.py`、`dagger_train.py`
* **最优策略 `policies/catch_bc_big.pt`**（61 维观测，256 宽）：**12–16 r0.5 92% / r1.5 100% / 16–20 全范围 54%**；窗口实测 60%
* 演示：`demo_live.py`（60 fps 实时视口，默认该策略）、`demo_catch.py`（统计 + 俯视轨迹图）
* 运动：按规格加减速（前后 3.5、横移 2.5、减速 5.0 m/s²），**无起步起跳**

## GPU 路径现状（`catch_env_gpu.py`）

已通：
* mjwarp 正确用法：`m = mjwarp.put_model(mjm)`；`d = mjwarp.put_data(mjm, MjData, nworld=..., nconmax=256, njmax=1024, naconmax=256*nworld)`；`mjwarp.step(m, d)`（**必须传 put_model 的返回值**，之前"ABI 不匹配"全是我传错对象）
* 裸物理吞吐：8192 世界 **0.67M env-steps/s**，GPU **78 W / 95%**
* 物理标定完成：`density=0.9`+`viscosity=1.8e-5`（15 m/s 平抛 0.5 s 保留 0.862，目标 0.857）；`solref="0.01 0.1"`（2.54 m 落下回弹 **1.23 m**，目标 1.35）
* 机器人跨引擎良好：满速前进 1 s → 2.36 m（MotrixSim 2.5 m）
* 引擎真值两遍法 `record_catch_points()`：256/256 条发球记录，33.5 s

未通：
* **GPU 教师只有 27%**（68/256），MotrixSim 同教师 60–90%
* 混杂开销：经 env 层 93k physics-steps/s（裸物理 670k），numpy↔warp 每次控制步搬运是瓶颈

## MotrixLab 接入（已完成并跑通训练入口）

改动都在 `/home/yangg/data/MotrixLab`（上游仓库克隆，**未提交**，便于 `git pull`）：

| 文件 | 内容 |
|---|---|
| `motrix_envs/src/motrix_envs/basic/tennis_catch/tennis_catch.xml` + `meshes/` | 我们的 42-body 模型（抄自 `rally_court.xml`）|
| `.../tennis_catch/cfg.py` | `@registry.envcfg("tennis_catch")` + `DirectEnvCfg`（`SimCfg(dt=0.0005)`、`ctrl_dt=0.02`、`max_episode_seconds=4.0`、发球/奖励参数）|
| `.../tennis_catch/tennis_catch.py` | `@registry.env("tennis_catch")` `DirectEnv`：12 维观测、3 维动作（车体系 fwd/left/yaw）、驱动斜坡 + 麦轮 IK + 力矩伺服、接球几何判定、奖励 shaping |
| `.../tennis_catch/serve.py` | 向量化发球采样（网高间隙解析判定）|
| `basic/__init__.py` | 加一行 `tennis_catch` |
| `configs/task/tennis_catch/skrl.ppo.yaml`、`rslrl.ppo.yaml` | 任务配方（`rollouts=32`、128×2、lr 5e-4）|

训练命令（**在 MotrixLab 根目录**）：
```bash
source .venv/bin/activate
mxlab train task=tennis_catch/skrl.ppo num_envs=512 logging.interval=10
# 结果: runs/tennis_catch/skrl/torch/ppo/<timestamp>/{checkpoints,summaries}
mxlab play env=tennis_catch policy=runs/.../checkpoints/best_agent.pt
```

### GPU 物理后端（`sim=mujoco_warp`）

MotrixSim 后端是 **CPU 物理**（512 世界 × 42 body 把 24 核吃满，GPU 只在跑 128×128 的 PPO 更新 ≈ 10%）。为把物理放到 GPU，我们给 MotrixLab 加了一个后端：

| 文件 | 内容 |
|---|---|
| `.../tennis_catch/mjwarp_backend.py` | `MujocoWarpBackend`（`SimBackend` 子类，注册名 `mujoco_warp`）：`put_model/put_data(nworld, nconmax=256, njmax=1024, naconmax=256*nworld)`、`dof_pos/dof_vel` 零拷贝视图（`wp.to_torch`）、`CtrlTargetsWrite` 写入 → `wp.copy(d.ctrl)`、`step(n)` 循环 |
| `.../tennis_catch/state_io.py` | 两个后端差异集中处：dof 索引（MotrixSim 的 `floating_bases` / MuJoCo 的 `jnt_qposadr`）与 reset 写状态（MotrixSim `set_dof_pos` / warp `qpos.assign` + `forward`）|

用法：`mxlab train task=tennis_catch/skrl.ppo sim=mujoco_warp num_envs=4096`

状态：GPU 后端已跑通（`MujocoWarpBackend`，物理与策略更新都在 GPU）。实测对比（同一模型、40 子步/控制步）：

| 后端 | env-steps/s | 说明 |
|---|---|---|
| MotrixSim（CPU） | ~3.4k | 24 核吃满，GPU 仅 ~10% |
| mujoco_warp（GPU，4096 世界） | **~14.6k** | GPU ~41%，功率 ~47 W |

调优记录（都是实测，不是估算）：
* 接触预算直接决定耗时：`nconmax/naconmax` 128/96 → 10.8k；64/48 → **14.3k**（已采用）；32/24 → 17.7k 但余量太小
* 每子步有 ~4 ms **固定开销**（n=64 也是 3.9 ms），所以小批量极不划算：n=512 → 3k env-steps/s，n=4096 → 14.6k
* 求解器迭代数（120 → 10）、线搜索（`ls_iterations`）、关闭辊子碰撞，**都没有影响** —— 不是求解器瓶颈
* 提醒：早期记的 "0.67M env-steps/s" 是**单位错误**（那是世界-子步/s，除以 40 才是 env-steps/s = 16.7k）

环境：为跑 mjwarp 把 venv 的 `mujoco` 从 3.9.0 升到 4.0.1（`motrix_env_mujoco` 的精确锁被打破，但该包只服务 MuJoCo 部署/导出路径，本任务不用）。

注意：为装 mjwarp 把 venv 的 `mujoco` 从 `3.9.0`（`motrix_env_mujoco` 的精确锁）升到 `4.0.1` —— 该包只服务 MuJoCo 部署/导出路径，本任务的训练与播放不走它。

### 接入时踩的坑（MotrixLab/MotrixSim 后端）

1. `BodyPositionWrite` 等声明式写**只对第一个浮基生效**（我们的球是第二个浮基，写入静默无效）→ reset 必须走 `sim._data.set_dof_pos/set_dof_vel`（和 `ros2_bridge.py` 同一套 API）。
2 `_named_body` 按 **body 名**（"base"/"ball"）解析，不是浮基名；自由关节名不在 `joint_names` 里。
3. **dof 索引必须从模型取**：42 body 里有 36 个辊子关节夹在 wheels 和 ball 之间 → ball 在 dof_pos **47:54**、dof_vel **46:52**（wheels 在 6/16/26/36），硬编码 11/10 会把球的速度写到角速度上。
4. 需要 `meshes/` 目录一起放进包（MJCF 用相对 `meshes/...` 路径）。
5. `SkrlCfg` 没有 `num_steps_per_env`（那是 RSLRL 的字段），SKRL 用 `algo.agent.rollouts`；未知键 Hydra 直接报错。
6. MotrixLab venv 里装的是 `motrixsim_core 0.10.2.dev132873`（开发构建），与我换装 0.10.1 后**性能同量级**（3.1k vs 5.2k env-steps/s）——之前记的 "130k env-steps/s" 是错的（那是无辊子/无场地接触的简化模型）。

### 实测性能（重要，纠正历史数字）

裸 `motrixsim.step_n(d, 40)`，1024 世界：**5.2k env-steps/s = 0.21M physics substeps/s**（参考 venv 3.1k）。即这个模型（36 辊子 × 地面接触 + 252 geom）每个世界每个控制步约 0.2 ms。训练吞吐因此约 **5k env-steps/s，1M 步 ≈ 3.5 分钟**；10M 步 ≈ 35 分钟。

### GPU 训练跑通（首个完整 run）

| 项 | 值 |
|---|---|
| 命令 | `mxlab train task=tennis_catch/skrl.ppo sim=mujoco_warp num_envs=4096` |
| 预算 | `trainer.timesteps=3000`（**控制步**）× 4096 世界 = 12.3M env 步 |
| 耗时 | **31 分 54 秒**，1.6 控制步/s = 6.5k env-steps/s |
| 硬件占用 | GPU 91%、75 W、3.8 GB；**CPU 0% user** |
| 产出 | `runs/tennis_catch/skrl/torch/ppo/<ts>/checkpoints/{agent_N,best_agent,final}.pt` |

跨引擎评测：`python eval_cross_engine.py --ckpt <run>/checkpoints/best_agent.pt --envs 512`
（把 SKRL 前向按训练代码原样重建：ELU MLP → `mean_layer`，观测走 `RunningStandardScaler`，在 MotrixSim 参考环境里跑）

注意：评测用小批量（48 世界）时只有 ~243 env-steps/s —— 固定开销主导，评测也用 512+ 世界。

## 跨引擎训练的真实结论（重要）

MotrixLab 集成与 GPU 训练链路都跑通了（4096 世界、GPU 91%、CPU 0%、6.5k env-steps/s），但**训练结果不能跨引擎使用**，原因已定位到三处物理不一致：

| 检查 | MotrixSim（参考） | CPU MuJoCo | mujoco_warp |
|---|---|---|---|
| 15 m/s 平抛 0.5 s 后速度比 | 0.854（手动冲量） | — | 0.854（环境层同一份代码 ✓）|
| 2 s 满速前进位移 | 4.56 m | **9.11 m**（车身还抬起 z=0.35）| **0.0 m**（两轮锁死两轮空转）|
| BC 策略接球率 | 97%（框架 env）/ 67%（参考 env）| — | 0%（修复物理后仍 0）|

根因不是我们的接线（ctrl 已确认加到 actuator、球飞行一致），而是**麦轮辊子接触对求解器极度敏感**：同一份 MJCF 在三个引擎里给出三种牵引。辊子靠 4 球 × 9 个、`friction=1.4` 各向异性接触传力，锥模型/迭代/摩擦处理不同，结果完全不同。

已做的修复（都正确、可复用）：
* 球物理统一 —— 阻力与 `COR(v)` 弹跳改为**环境层逐子步施加**（两后端同一份数学，实测 0.854/0.854），不再依赖任何引擎内置流体/恢复系数
* 框架缺的 `init_policy` 钩子（SKRL 的 `resume` 未实现）—— BC 权重可直接灌进 PPO
* BC 热启动实测：参考 env 67%、MotrixSim 后端框架 env **97%**

### 方案 B 已实施：驱动改为速度伺服

模型改成 `<velocity joint=... kv=2 ctrlrange="±60" forcerange="±12">`，三处驱动写入同步改为"直接下发轮速"（`gen_mjcf.py`、`catch_env.py::_drive`、`ros2_bridge.py::control` 两个写入点），MotrixLab env 同样改为下发 `_wheel_ref`。轮速由引擎的速度约束保持，不再依赖"接触摩擦 -> 力矩"这条引擎相关的链路。

实测（2 s 满速前进）：

| 引擎 | 位移 | 备注 |
|---|---|---|
| MotrixSim（参考） | **5.27 m** | 无抬头（z 0.157），原力矩伺服 4.56 m |
| CPU MuJoCo | **6.4 m** | 原力矩伺服 9.11 m 且抬头到 z=0.35 |
| **mujoco_warp** | **0.0 m** | 轮子转到 ±41 rad/s 却零牵引 |

差异从 2× 收到 ~20% ✓；**但 mujoco-warp 完全无法驱动这个模型** —— 单世界（`nworld=1`）也一样，摩擦各向同性/`condim=4`/`cone`/`PGS` 等变体全部无效，而 CPU MuJoCo 同模型正常。这是 mjwarp 对 36 个各向异性辊子接触的实现限制，不是我们的接线。

策略未受影响：速度伺服下 BC 策略在框架 env（MotrixSim）里 **98%**（力矩伺服时 97%）。

**结论：本机器人的 GPU 训练被 mjwarp 的麦轮接触限制挡住**；训练仍走 MotrixSim（3.4k env-steps/s，12.3M env 步 ≈ 60 分钟），GPU 那条路要等 mjwarp 能正确处理这类接触，或者把轮子简化成"各向异性单体 + 显式麦轮力模型"（会牺牲接触真实性）。

### 让驱动跨引擎一致的另外两条路（未采用）
1. **换执行器**：轮子从"力矩 + 摩擦牵引"改为速度伺服（引擎无关），代价是失去打滑真实感，但与真机更接近（真机也用轮速闭环）；
2. **标定 MuJoCo 接触参数**（`solref/solimp/cone/impratio/摩擦`）拟合实测驱动规格（加速 3.5/2.5、减速 5.0、vmax 4.0/4.5、2 s→4.56 m）—— 工作量更大但保留物理真实性。

**当前可用结论**：策略训练与验证继续走 MotrixSim（BC 92/100/54、热启动 BC 67–97%）；GPU 侧只适合做**球/策略层面**的大规模实验，不适合作为本模型的训练环境，直到驱动一致性问题解决。

## 下一步（按顺序）

1. **GPU 诊断 27%**：在记录的真实穿越时刻量笼坐标系下的脱靶距离（CPU 侧同一套方法曾定位出"目标定义错误"和"横移滑移"）。若属跟踪问题，把辊子接触/摩擦参数按实测驱动规格（加速 3.5/2.5、减速 5.0、vmax 4.0/4.5）拟合。
2. **接入 MotrixLab 训练**（用户现在要求训练走 MotrixLab）：
   - MotrixLab 建在 MotrixSim 上，我们的 `catch_env.py` 同源，改造成它的 env 接口（obs/action/reward/termination + `configs/task/<task>/<algo>.yaml`）应比 UniLab 薄；
   - 参考 `/home/yangg/data/MotrixLab/configs/task/` 现有任务与 `mxlab train task=... play=true`；
   - `mxlab play env=...` 用于回放。
3. 若 MotrixLab 侧跑通，用它的并行环境 + SKRL/RSL-RL/FastSAC 替代手写 PPO（手写 PPO 微调在本项目两次把策略打崩，勿再用）。

## 关键坑（勿重复踩）

* `mjwarp.put_model` 返回包装对象，`step` 传它；接触上限是 `put_data` 的参数，不是 MJCF `<size>`
* MuJoCo 软接触 `solref` 时间常数太小会数值爆炸（0.002 s → 回弹 39 m），用 0.01 s
* `~/.local/lib/python3.12/site-packages` 里的 mujoco 会遮蔽 conda 环境，跑 `ms` 时加 `PYTHONNOUSERSITE=1`
* BC/DAgger 的标签必须是**观测的函数**（解析目标），否则监督=噪声（拟合 MSE 0.44 那次就是这个）
* 训练/评测的观测必须一致（`set_serves` 没初始化轨迹缓冲区那次导致"拟合完美但 0%"）
* 未提交的临时环境改动：`ms` 里装了 mujoco/mjwarp 4.0.1；`~/data/dev/{mujoco,mujoco_warp}` 源码树与 `mujoco_install` 保留可用
