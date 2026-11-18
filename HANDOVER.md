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

### 接入时踩的坑（MotrixLab/MotrixSim 后端）

1. `BodyPositionWrite` 等声明式写**只对第一个浮基生效**（我们的球是第二个浮基，写入静默无效）→ reset 必须走 `sim._data.set_dof_pos/set_dof_vel`（和 `ros2_bridge.py` 同一套 API）。
2 `_named_body` 按 **body 名**（"base"/"ball"）解析，不是浮基名；自由关节名不在 `joint_names` 里。
3. **dof 索引必须从模型取**：42 body 里有 36 个辊子关节夹在 wheels 和 ball 之间 → ball 在 dof_pos **47:54**、dof_vel **46:52**（wheels 在 6/16/26/36），硬编码 11/10 会把球的速度写到角速度上。
4. 需要 `meshes/` 目录一起放进包（MJCF 用相对 `meshes/...` 路径）。
5. `SkrlCfg` 没有 `num_steps_per_env`（那是 RSLRL 的字段），SKRL 用 `algo.agent.rollouts`；未知键 Hydra 直接报错。
6. MotrixLab venv 里装的是 `motrixsim_core 0.10.2.dev132873`（开发构建），与我换装 0.10.1 后**性能同量级**（3.1k vs 5.2k env-steps/s）——之前记的 "130k env-steps/s" 是错的（那是无辊子/无场地接触的简化模型）。

### 实测性能（重要，纠正历史数字）

裸 `motrixsim.step_n(d, 40)`，1024 世界：**5.2k env-steps/s = 0.21M physics substeps/s**（参考 venv 3.1k）。即这个模型（36 辊子 × 地面接触 + 252 geom）每个世界每个控制步约 0.2 ms。训练吞吐因此约 **5k env-steps/s，1M 步 ≈ 3.5 分钟**；10M 步 ≈ 35 分钟。

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
