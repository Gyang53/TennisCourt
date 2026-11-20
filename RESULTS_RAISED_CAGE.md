# 网球接球机器人：抬升篮筐后的训练结果（2026-11-20）

## 1. 本次改动

篮筐底板高度从 **0.26 m 抬到 0.50 m**，并为整机增加篮筐质量。

### 几何（实测，离地高度）

| 部件 | 尺寸（全长） | 高度范围 |
| --- | --- | --- |
| 底板 deck | 0.550 × 0.550 × 0.016 m | 0.484 – **0.500 m** |
| 支撑柱 ×4（新增） | 0.028 × 0.028 × 0.240 m | 0.161 – 0.401 m |
| 四角立柱 ×4 | r 0.008、长 0.476 m | 0.400 – 0.860 m |
| 侧墙 ×3（+x/+y/−x） | 0.416 × 0.048 × 0.360 m | 0.500 – 0.860 m |
| 开口唇边（−y） | 0.416 × 0.048 × 0.160 m | 0.500 – 0.660 m |
| 内部净空 | 0.416 × 0.416 × 0.360 m | 0.500 – 0.860 m |

篮筐开口朝车身 **−y**（三面 0.36 m 墙 + −y 侧 0.16 m 低唇边），因此车身朝向会直接影响进球。

### 质量与惯量

| 项 | 旧 | 新 |
| --- | --- | --- |
| 总质量 | 16 kg | **20 kg**（篮筐折算 4 kg @ 局部 z 0.3945） |
| 质心（车体系） | 0.120 m | **0.175 m**（世界高度 ≈0.33 m） |
| 惯量 diag | 0.54 / 0.60 / 0.90 | **0.92 / 0.99 / 1.10** |

### 判定与奖励常量

- 接球窗口：车体系 **0.344 – 0.704 m**（世界 0.50 – 0.86 m）
- `CATCH_Z`（末段判定高度）：**0.55 m**
- 偏航阈值：**35°**（超出后 0.002 / 弧度 / 步）
- 开口对准奖励：`facing_scale = 0.001 × cos(开口与球夹角)`

## 2. 训练配方

| 项 | 值 |
| --- | --- |
| 算法 | MotrixLab `motrix.fastsac`（off-policy、replay、分布型 critic、UTD 8） |
| 物理 / 学习 | MotrixSim CPU（512 世界）+ CUDA（RTX 5060 Laptop） |
| 关键超参 | `gamma=0.99`（回合末端奖励需要）、`rollouts`/ring 默认 |
| 预算 | 512 世界 × 6000 迭代 ≈ 307 万环境步，约 19 分钟 |
| 任务 | `tennis_catch_yaw`（t3, 16–20 m/s r4.9）、`tennis_catch_yaw_t2`（12–16 r1.5）、`tennis_catch_yaw_t1`（12–16 r0.5） |

## 3. 三档结果（各用自己的档位，256 环境 × 2 轮）

| 档位 | 最好命中率 | 对应 checkpoint | 开口对准 | 偏航 mean/max |
| --- | --- | --- | --- | --- |
| **tier 1**（12–16 m/s, r0.5） | **92.5%** | `runs/tennis_catch_yaw_t1/motrix/torch/fastsac/26-11-20_18-38-51-933801/checkpoints/model_0005750.pt` | 5.5° | 8.7° / 12.4° |
| tier 2（12–16 m/s, r1.5） | 10.9% | `runs/tennis_catch_yaw_t2/motrix/torch/fastsac/26-11-20_18-21-02-661734/checkpoints/model_0003000.pt` | — | — |
| tier 3（16–20 m/s, r4.9） | **53.1%** | `runs/tennis_catch_yaw/motrix/torch/fastsac/26-11-20_17-52-44-999978/checkpoints/model_0001250.pt`（1500 迭代版本） | 4.8° | 23.7° / 32.0° |

tier 1 的学习曲线（64 环境抽查）：500 → 3.1%，1000 → 54.2%，1500 → 73.0%，2000 → 89.8%，3000 → 90.2%，5750 → 92.5%。

## 4. 与抬升前的对照（旧篮筐 0.26 m、16 kg）

| 阶段 | tier 3 命中率 | 说明 |
| --- | --- | --- |
| PPO（最好 checkpoint） | 52% | 加算力无效；崩溃由更新幅度驱动 |
| FastSAC 自由偏航 | **90.4%** | off-policy replay 解决优势信号问题 |
| + 35–40° 偏航阈值 | 91.5% | 偏航超出阈值的时间占比 56.5% → 0% |
| + 开口对准奖励 | **93.8%** | 开口平均偏角 18.3° → 2.5° |

## 5. 结论与已知差距

1. **抬升篮筐显著提高难度**：同样算法在旧几何上 tier 3 达 93.8%，新几何上长训练只到 5.6%；5 分钟短预算版本反而到 53.1% —— 说明 **学习率调度与总迭代数耦合**（同样 1250 迭代、不同总预算，结果差 10 倍），新几何需要重新调预算/调度或做课程。
2. tier 1 已达 **92.5%**，可直接用于该档演示与后续 sim2real。
3. tier 2/3 尚未达标（10.9% / 53.1%），下一步建议：
   - 用 tier 1 的策略做 `init` 热启动再训 t2，再以 t2 热启动 t3（课程式）；
   - 或扫描 `num_learning_iterations`（500/1000/1500）配合 checkpoint 选点；
   - 或加大篮筐开口（当前净空 0.416 m）降低快球档的精度要求。
4. **未验证**：sim2real（域随机化、真机标定）；质量模型仍是单点集中质量。
5. UniLab 已安装在 `~/data/UniLab`（MJWarp 后端，32k env-steps/s，约 8.4×），但缺 tennis-catch 任务 owner，暂未接入。

## 6. 复现命令

```bash
cd ~/data/MotrixLab && source .venv/bin/activate

# 训练（三档；每档约 19 分钟）
python -m mxlab train +task/tennis_catch_yaw=motrix.fastsac
python -m mxlab train +task/tennis_catch_yaw_t2=motrix.fastsac
python -m mxlab train +task/tennis_catch_yaw_t1=motrix.fastsac

# 评测（在 tennis_robot 下）
cd ~/data/tennis_robot
python eval_fastsac.py --ckpt <run>/checkpoints/model_0005750.pt \
  --run-config <run>/task_config.yaml --tier 1 --envs 256 --rounds 2

# 视口（-/= 实时调球速，r 重发，w 切机位）
python view_catch.py --fastsac --ckpt <run>/checkpoints/model_0005750.pt \
  --run-config <run>/task_config.yaml --tier 1
```

## 7. 文件与提交

| 仓库 | 内容 | 远程 |
| --- | --- | --- |
| `MotrixLab` | 抬升篮筐 + 支撑柱 + 质量/惯量、接球窗口、35° 偏航阈值、对准奖励、t1/t2 任务 | 仅本地（官方仓库不推送） |
| `tennis_robot` | `eval_fastsac.py`（FastSAC 评测 + 对准/偏航指标）、`eval_tier.py`、`view_catch.py`（实时调球速/偏航统计）、本报告 | origin + 10.1.1.99:asic/tenniscourt_motrix |
