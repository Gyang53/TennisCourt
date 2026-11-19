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

## "离线 98%、训练里 0%" 的四次排查（全部是训练/验证管线不一致）

| # | 不一致 | 表现 | 修法 | 修后 |
|---|---|---|---|---|
| 1 | 球阻力/弹跳是参考 env **手动施加**的，移植版依赖引擎内置 | 框架 env 里 BC 3% | 环境层逐子步 `apply_ball_impulse` | 97% |
| 2 | 两后端球物理不同（warp 流体+恢复系数 vs 冲量）| warp 侧 BC 0% | 停用引擎内置，两后端同一份代码 | 两后端阻力均 0.854 |
| 3 | SKRL 强制观测归一化，BC 网络却在**原始**观测上拟合 | 微调 100 步内 98%→0% | BC 用数据集统计量归一化后拟合，统计量写入 checkpoint | 热启动 97%（按 SKRL 方式应用）|
| 4 | checkpoint 的 `log_std` 继承自零起点 run（std 2.75）| 动作≈均匀噪声 | `bc_warmstart.export` 固定 std=0.3 | — |

关键检验命令（以 SKRL 完全相同的方式应用 checkpoint，1000 回合 97%）：

```python
x = ((obs - mean) / sqrt(var + 1e-8)).clamp(-5, 5)   # checkpoint 的 observation_preprocessor
a = net(x)                                            # 12->128->128->3 ELU
```

`bc_warmstart.py` 现在自带演示缓存（`/tmp/tennis_bc_dataset.npz`），重拟合 1 分钟内完成。
微调仍需温和超参：`lr 5e-5`、`entropy_loss_scale 0`、`rollouts 64`（`lr 5e-4` 会在数百步内把 97% 的策略打崩）。

## 最后一个关键 bug：`models.separate` 默认 false → 共享网络

诊断打点（`INITDBG`）显示 `agent.policy` 与 `agent.value` 是**同一个 `Shared` 对象（id 相同）**：MotrixLab 的 SKRL 配方默认 **policy 与 value 共享网络**，于是 `init_policy` 先灌 BC 权重、再灌旧 value 权重，**后者覆盖前者** —— 训练里跑的从来不是 BC 策略（这解释了此前所有"离线 97%、训练 0%"）。

修复三件套（都在 `/home/yangg/data/MotrixLab`）：

1. 任务配方加 `algo.models.separate: true`（policy/value 分开）与 `clip_actions: true`；
2. `init_policy` 只加载 **policy**（新 critic 从零开始更稳），并 `strict=False`（共享模型写出的 checkpoint 带 `value_layer.*`）；
3. 热启动时**冻结观测归一化**（`_FrozenScaler`），避免统计量漂移把策略输入带离拟合分布。

验证：`catch (max)` 稳定出现 1.0、`catch (mean) ≈ 0.002–0.009`（此前恒 0）。

另外量化了探索噪声的代价（同一策略、框架 env、1000 回合）：**确定性 97% / std 0.1 → 64% / std 0.3 → 30%** —— 所以热启动的 `log_std` 设为 std 0.1，不是细节。

## 观测维度决定上限：MotrixLab env 改为 61 维轨迹观测

用修正后的评测（乘 VMAX 动作口径）测得，12 维状态观测的上限：

| 档位 | 12 维热启动 BC | 12 维 + PPO 微调 | 61 维 BC 基线（交付策略）|
|---|---|---|---|
| 12–16 m/s, r0.5 | 18% | 28% | **92%** |
| 12–16 m/s, r1.5 | 65% | 68% | **100%** |
| 16–20 m/s 全范围 | 35% | 29% | **54%** |

与早期结论一致：**不含球轨迹的策略封顶在 30–65%**，微调只能小幅挪动。因此 MotrixLab env 的观测改为 61 维（`12 + 16×3 + 1`，与参考 `obs_mode="both"` 逐项同序：状态 12、路径 xy/5、路径 z/3、age/4），`bc_warmstart.py` 同步改为 `obs_mode="both"`、网络输入维度从数据推断、不再依赖 template checkpoint（直接合成 SKRL 格式）。

评测口径提醒：`CatchEnv.step` 收**原始车辆指令**（m/s、rad/s），策略输出是 ±1 归一化 —— `eval_cross_engine.py` 早先漏乘 VMAX，导致所有 0% 结论无效（已修，`4fe30e2`）。

### 自蒸馏 61 维 BC 不如交付策略

| 档位 | 自蒸馏 61 维（231k 示范、128 宽）| 交付策略（256 宽，`policies/catch_bc_big.pt`）|
|---|---|---|
| 12–16, r0.5 | 23% | **92%** |
| 12–16, r1.5 | 44% | **100%** |
| 16–20 全范围 | 27% | **54%** |

所以不再重造策略，改为**用交付策略做 PPO 热启动**：`export_skrl_warmstart.py` 把它转成 SKRL checkpoint（`trunk.*→net.*`、`actor.*→mean_layer.*`、std 0.1、**恒等归一化**），任务配方同步改为 `hiddens [256,256]` + `hidden_activation [tanh]`。

## 最终结果：MotrixLab PPO 微调交付策略

起点 = 交付策略 `policies/catch_bc_big.pt`（转成 SKRL 热启动，`std=0.1`、恒等归一化、冻结）；
`mxlab train task=tennis_catch/skrl.ppo num_envs=512 sim=默认(MotrixSim) algo.agent.learning_rate=1e-4
algo.agent.entropy_loss_scale=0.0 algo.agent.rollouts=64` → 8000 控制步 = 4.1M env 步 / 19:21。

同一评测口径（框架 env、MotrixSim 后端、192 世界 × 2 轮 ≈ 384 回合/档）：

| 档位 | 起点（交付策略）| 微调后 `policies/catch_ppo.pt` | 变化 |
|---|---|---|---|
| 12–16 m/s, r0.5 | 41% | **64%** | **+23** |
| 12–16 m/s, r1.5 | 55% | 49% | −6 |
| 16–20 m/s 全范围 | 37% | **39%** | +2 |

训练分布（16–20 m/s, r1.5）同口径：95% → **97%**。权重只移动了 0.9%（actor ΔL2 0.0071 / 范数 0.778）。

**结论**：微调对最紧的档位（r0.5）有显著提升，最难档小幅提升，中间档略降。`policies/catch_ppo.pt` 作为**候选策略**保留，默认仍用 `catch_bc_big.pt`。

### 未解的口径差异（务必注意）

`eval_three_tiers.py`（框架 env）测出的绝对数字明显低于历史口径（`eval_catch.py` + 参考 env 报告的 92/100/54）：同一份交付策略在 r0.5 档只有 41% ✗。两者在**发球采样参数或接球判定**上必然有差异，未定位。因此：
* 只在同一口径内比较（上表的 +23 / −6 / +2 有效）；
* 不要跨脚本比较绝对数字；
* `eval_cross_engine.py` 对 61 维观测仍然不可靠（已发现两处 bug，61 维下仍报 0%），优先用 `eval_three_tiers.py`。

## GPU 训练：平面基座抽象（`tennis_catch_planar`）

结论先行：**mujoco-warp 无法仿真这个麦轮机构**（4 球/辊、1 球/辊、圆柱/辊、球轮四种碰撞体都不产生牵引；`nworld=1` 亦然；摩擦各向同性/`condim=4`/`cone`/`impratio`/`PGS` 全无效，而 CPU MuJoCo 每种都能驱动；warp 里车身还坐得更高 0.167 vs 0.158）。MotrixSim **没有外部力注入 API**，所以"现象学推力"无法两引擎一致实现。

因此 GPU 训练改用**平面基座抽象**：`planar_catch.xml` 把 4 轮 + 36 辊子的接触换成 3 个正交速度伺服关节（x/y/yaw），保留实测驱动斜坡（3.5/2.5 加速、5.0 减速、4.0/4.5 m/s），观测与动作接口与完整模型完全一致（61 维 / (fwd,left,yaw)）。

| 项 | 结果 |
|---|---|
| 方向/量纲 | 2 s 前进 **5.71 m**（+x，与参考约定一致：车辆前向 = (sinθ, −cosθ)）|
| GPU 吞吐（warp，8192 世界）| **57.2k env-steps/s**（143 ms/控制步）= CPU 路径的 **~16×** |
| 同一物理在 CPU 上 | 用于验证的完整麦轮模型仍跑 MotrixSim |

训练：`mxlab train task=tennis_catch_planar/skrl.ppo sim=mujoco_warp`（8192 世界、`timesteps=2000` = 16.4M env 步，实测 **5 分 40 秒** = 48k env-steps/s），热启动用交付策略。

### 抽象必须匹配机构的"可达动力学"

未标定的平面基座（把 4.5 m/s 横移指令全额执行）导致微调把策略拉向不存在的权限：

| 评测环境 | 交付策略（起点）| GPU 微调（未标定抽象）|
|---|---|---|
| 平面基座 | 100% / 100% / 30% | 99% / 99% / **39%**（微调有效）|
| **完整麦轮模型** | 41% / 55% / 37% | **12% / 12% / 21%**（迁移失败）|

实测麦轮稳态可达比（斜坡后）：前进 **0.94**、横移 **0.93**、转向 **0.95** —— 用 `PLANAR_GAINS` 灌进平面基座后，1.2 s 时的速度是 3.76 / 2.78 m/s，与完整模型的 3.76 / 2.79 误差 <1%。重训后迁移为 **27% / 14% / 20%**，仍低于起点。

### GPU 训练的最终结论

| 策略 | 平面 env（训练域）| **完整麦轮 env（验收域）** |
|---|---|---|
| 交付策略（起点）| 100 / 100 / 30 | **41 / 55 / 37** |
| GPU 微调（未标定抽象）| 99 / 99 / 39 | 12 / 12 / 21 |
| GPU 微调（已标定抽象）| ~99 / ~99 / ~39 | 27 / 14 / 20 |
| **CPU 微调（真实物理）** | — | **64 / 49 / 39** ← 交付候选 `policies/catch_ppo.pt` |

原因：平面基座是**理想速度跟踪**（无滑移、无横移位置误差、无偏航耦合），因此在其上微调会把策略推向该理想化的癖好；而麦轮有约 0.2 m 的横移稳态误差与偏航漂移，交付策略正是靠补偿这些才达到 41–55%，一旦漂走就大幅掉档。

**定位**：GPU 环境（16× 吞吐、5 分 40 秒跑 16.4M 步）作为**提案生成器**，每次产出后用完整模型验收，只有不回归才采纳；交付走 CPU 微调。要让 GPU 训练真正产生净收益，下一步必须把麦轮的**位置误差特性**（滑移导致的位置滞后、横移方向偏航耦合）加进平面基座，并用"保持目标点"这类测试对齐两种机构。

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
