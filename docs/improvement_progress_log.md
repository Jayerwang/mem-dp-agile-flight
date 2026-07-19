# MemFlight 改进进度日志

更新时间：2026-07-19

本文档用于持续记录 DiffPhysDrone / MemFlight 改进实验的阶段结果、暴露问题和下一步计划。后续每完成一次训练或统一测评，都优先更新本文档。

## 1. 当前统一测评协议

当前主要使用 CUDA original-like 测评协议，目的是在本地/云服务器可运行的条件下，对原模型和改进模型做一致对比。

```text
env_kind: original
speed_mtp: 2.5
speeds: 2, 4, 6, 7, 8, 9, 10 m/s
num_episodes: 15
batch_size: 32
主要指标: success rate, crash rate, timeout rate
```

说明：

- 该协议不是论文 Figure 8 的 Flightmare/ROS 图像渲染协议。
- 该协议适合比较本项目不同 checkpoint 在同一 CUDA 环境下的相对能力。
- 训练日志中的 `AR = success_rate * avg_speed` 只能作为训练内部指标，不能直接替代统一 eval 结果。

## 2. 当前权重与路径记录

服务器侧主要路径：

```text
项目根目录:
/root/autodl-tmp/.autodl/DiffPhysDrone-master/DiffPhysDrone-master

原版 checkpoint:
backup_checkpoints/20260413_131854/checkpoint0004.pth

v4 训练目录:
memdpflight_v4_progress_goal

v4 重要权重:
memdpflight_v4_progress_goal/checkpoint00001.pth  # v4_5000
memdpflight_v4_progress_goal/checkpoint00002.pth  # v4_10000
memdpflight_v4_progress_goal/checkpoint_best.pth  # v4_best, 训练 AR 最优
```

注意：

- 旧目录 `memdpflight_ft_phase2_safe/checkpoint_best.pth` 曾被 20 iter smoke test 覆盖，不再作为可靠的旧 `ft_best` 代表。
- 当前 v4 结果中，统一 eval 最优是 `checkpoint00001.pth`，不是 `checkpoint_best.pth`。

建议每次训练结束后记录：

```bash
ls -lh --time-style=long-iso checkpoint_best.pth checkpoint00001.pth checkpoint00002.pth
sha256sum checkpoint_best.pth checkpoint00001.pth checkpoint00002.pth
stat checkpoint_best.pth checkpoint00001.pth checkpoint00002.pth
```

## 3. 阶段记录

### 3.1 原版 baseline

权重：

```text
backup_checkpoints/20260413_131854/checkpoint0004.pth
```

最新统一 eval 结果：

| 速度 | Success | Crash | Timeout |
| ---: | ---: | ---: | ---: |
| 2 | 72.71% | 4.38% | 22.92% |
| 4 | 85.21% | 4.38% | 10.42% |
| 6 | 85.21% | 7.71% | 7.08% |
| 7 | 82.50% | 8.96% | 8.54% |
| 8 | 86.67% | 6.88% | 6.46% |
| 9 | 81.04% | 12.71% | 6.25% |
| 10 | 79.79% | 15.00% | 5.21% |

平均：

```text
success: 81.88%
crash:   8.57%
timeout: 9.55%
```

主要问题：

- 低速 2m/s timeout 较高。
- 高速 9-10m/s crash 增加明显。
- 原模型并不是在所有 CUDA original-like 速度段都稳定优秀。

### 3.2 MemFlight / Phase2 safe 微调阶段

思路：

- 在原 DiffPhysDrone 模型基础上引入显式记忆机制。
- 使用 past/current 深度特征重建，参考 MAVRL 中的历史信息利用思想。
- 通过 full-density phase2 safety fine-tuning 将记忆增强模型重新适配 original-like 障碍环境。

历史观察：

```text
phase1_done 直接测评并不一定优于原版。
经过 full-density phase2 微调后，平均成功率能超过原版。
旧 ft_best 曾达到约 82.9% 平均成功率，但当前同名文件已被后续 smoke test 覆盖。
```

暴露问题：

- checkpoint 命名容易覆盖，尤其是固定文件名 `checkpoint_best.pth`。
- 训练日志 best 指标和最终统一 eval 指标不完全一致。
- 高速 crash 和低速 timeout 仍未同时解决。

### 3.3 v4: Progress / Goal Loss 微调

代码版本：

```text
本地:
D:\UAV\DiffPhysDrone-master\DiffPhysDrone-master\memdpflight\mem_main_cuda_v4_progress_goal.py

服务器训练目录:
/root/autodl-tmp/.autodl/DiffPhysDrone-master/DiffPhysDrone-master/memdpflight_v4_progress_goal
```

核心改动：

- 不改模型结构，不改 CUDA kernel。
- 新增 `progress loss`：鼓励每个时间步持续接近目标。
- 新增 `goal loss`：惩罚 episode 末端距离目标过远。
- 新增 `force_full_density`：固定满密度障碍物，和 original-like 测评更一致。

训练命令核心参数：

```text
resume: checkpoint_phase1_done.pth
num_iters: 10000
phase0_iters: 0
phase1_iters: 0
speed_mtp: 2.5
fov_x_half_tan: 0.82
cam_angle: 20
ground_voxels: true
random_rotation: true
yaw_drift: true
coef_collide: 10.0
coef_obj_avoidance: 4.0
coef_progress: 0.05
coef_goal: 0.01
progress_margin: 0.02
recon_coef_past: 0.03
recon_coef_cur: 0.02
force_full_density: true
```

训练日志关键现象：

```text
v4_best training AR: 6.048
training SR: 1.000
training Speed: 6.05 m/s
```

这说明训练内部 AR 持续升高，且后期策略明显偏向更高速度。

#### v4 统一 eval 结果

平均结果：

| 模型 | 平均 Success | 平均 Crash | 平均 Timeout | 结论 |
| --- | ---: | ---: | ---: | --- |
| 原版 `checkpoint0004` | 81.88% | 8.57% | 9.55% | baseline |
| `v4_5000` | 83.45% | 7.62% | 8.93% | 当前最佳 |
| `v4_10000` | 82.83% | 8.18% | 8.99% | 略优于原版 |
| `v4_best` | 82.65% | 8.07% | 9.28% | 训练 AR 最优，但 eval 不是最优 |

分速度结果：

| 速度 | 原版 S/C/T | v4_5000 S/C/T | v4_10000 S/C/T | v4_best S/C/T |
| ---: | --- | --- | --- | --- |
| 2 | 72.71 / 4.38 / 22.92 | 70.83 / 4.17 / 25.00 | 69.79 / 3.75 / 26.46 | 69.38 / 3.96 / 26.67 |
| 4 | 85.21 / 4.38 / 10.42 | 87.71 / 3.33 / 8.96 | 85.42 / 3.75 / 10.83 | 85.83 / 3.33 / 10.83 |
| 6 | 85.21 / 7.71 / 7.08 | 86.88 / 6.46 / 6.67 | 87.71 / 6.67 / 5.62 | 87.08 / 6.46 / 6.46 |
| 7 | 82.50 / 8.96 / 8.54 | 86.04 / 7.50 / 6.46 | 86.25 / 8.54 / 5.21 | 85.83 / 8.96 / 5.21 |
| 8 | 86.67 / 6.88 / 6.46 | 85.83 / 8.33 / 5.83 | 85.42 / 9.58 / 5.00 | 85.62 / 8.75 / 5.62 |
| 9 | 81.04 / 12.71 / 6.25 | 84.17 / 10.62 / 5.21 | 84.17 / 10.42 / 5.42 | 83.96 / 10.42 / 5.62 |
| 10 | 79.79 / 15.00 / 5.21 | 82.71 / 12.92 / 4.38 | 81.04 / 14.58 / 4.38 | 80.83 / 14.58 / 4.58 |

S/C/T 表示：

```text
Success / Crash / Timeout
```

## 4. v4 暴露出来的问题

### 4.1 低速 2m/s timeout 恶化

现象：

```text
原版 timeout: 22.92%
v4_5000 timeout: 25.00%
v4_10000 timeout: 26.46%
v4_best timeout: 26.67%
```

判断：

- 低速失败主要是 timeout，不是 crash。
- v4 在 2m/s 下更保守，或者目标推进不足。

可能原因：

- 训练后期速度分布偏向中高速。
- `progress_margin=0.02` 是固定值，没有按目标速度或场景尺度归一化。
- `coef_speed=0.0`，没有显式强化“按指定速度飞行”。
- 训练 best 使用 AR，会偏向高速收益，不会奖励低速到达稳定性。

### 4.2 8m/s crash 上升

现象：

```text
原版 crash: 6.88%
v4_5000 crash: 8.33%
v4_10000 crash: 9.58%
v4_best crash: 8.75%
```

判断：

- v4 在 8m/s 附近减少 timeout，但付出更高 crash 代价。
- 说明 progress/goal 约束让模型更积极向前推进，但局部安全裕度不足。

可能原因：

- 8m/s 是策略从中速到高速行为切换的边界区。
- 避障损失没有直接表达未来风险，只表达当前 rollout 中的障碍距离。
- 记忆重建仍然是深度特征重建，不是风险重建。

### 4.3 `checkpoint_best.pth` 不等于 eval 最优

现象：

```text
训练 AR 最优: v4_best
统一 eval 最优: v4_5000
```

判断：

- 当前 best 保存指标偏向 `success * avg_speed`。
- 统一 eval 是分速度平均，低速和中速失败也会计入。
- 因此训练内部 best 可能更像“高速倾向权重”，不适合作为最终模型选择标准。

## 5. 当前阶段结论

v4 达到部分预期：

- 平均成功率高于原版。
- 平均 crash 低于原版。
- 平均 timeout 低于原版。
- 6-10m/s 多数速度段 timeout 下降。
- 9-10m/s 高速段整体优于原版。

v4 未完全达到预期：

- 2m/s timeout 明显恶化。
- 8m/s crash 上升。
- 训练 best 和统一 eval best 不一致。
- 后续训练到 10000 iter 不如 5000 iter，存在后期速度偏置。

当前代表模型：

```text
v4_5000 = memdpflight_v4_progress_goal/checkpoint00001.pth
```

暂不建议用：

```text
v4_best = memdpflight_v4_progress_goal/checkpoint_best.pth
```

作为最终代表模型。

## 6. 下一步计划

### 6.1 立即执行：结果留档

目的：确认权重没有混淆或覆盖。

待做：

- 记录 `v4_5000 / v4_10000 / v4_best` 的文件修改时间。
- 记录三者 sha256。
- 将最新 eval 结果目录复制或记录到文档中。

结果目录：

```text
experiments/results/round1_20260719_092530
```

### 6.2 补充测评：高速度 stress test

目的：判断 v4 是否真的提高高速能力边界。

建议速度：

```text
11, 12 m/s
```

如果 12m/s 仍然成功率较高，可继续：

```text
13, 14 m/s
```

停止标准：

```text
连续两个速度点 success < 50%
或 crash > 40%
或 timeout/crash 已明显失控
```

该测试作为 stress test，不替代当前 2-10m/s 主协议。

### 6.3 v4.1 保守改进方向

优先级从高到低：

1. 固定 checkpoint 选择策略  
   不再只依赖训练 AR 选择最终 best。每 5000 iter 保存固定 checkpoint，用统一 eval 选择最终模型。

2. 低速 timeout 修正  
   可尝试更温和的损失：

   ```text
   coef_progress: 0.03
   coef_goal: 0.005
   coef_speed: 0.03 或 0.05
   ```

   目标是减少低速 timeout，但不显著增加 8m/s crash。

3. 显式速度段均衡训练  
   训练时不要只依赖 `speed_mtp` 随机场景分布，考虑让目标速度覆盖：

   ```text
   2, 4, 6, 8, 10 m/s
   ```

   或至少记录每个 episode 的速度分布。

4. 风险辅助头或风险重建  
   暂时不作为下一步首选，因为改动较大。只有当 v4.1 仍无法处理 8m/s crash 时再考虑。

### 6.4 当前不建议立刻做的事

- 不建议马上大改 CUDA kernel。
- 不建议马上把 recon depth 改成 recon risk。
- 不建议只根据训练日志 AR 继续追高速。
- 不建议用被覆盖过的旧 `checkpoint_best.pth` 做正式对比。

## 7. 后续更新模板

每次新训练或新测评后，追加如下记录。

```markdown
## YYYY-MM-DD: vX 实验记录

### 目的

### 代码版本

### 训练命令关键参数

### 产物路径

### 训练日志关键点

### 统一 eval 结果

| 模型 | 平均 Success | 平均 Crash | 平均 Timeout |
| --- | ---: | ---: | ---: |

### 分速度问题

### 结论

### 下一步
```

