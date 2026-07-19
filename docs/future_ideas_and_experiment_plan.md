
## 1. 当前现象

目前已经观察到几个稳定现象：

- 原版模型在统一 original-like CUDA eval 下并非所有速度都完美，低速可能 timeout，高速可能 crash。
- 未充分 phase2 微调的 memflight 权重，例如 `checkpoint_phase1_done`，在 original-like 满障碍测试中略低于原版。
- 从 `checkpoint_phase1_done` 做 full-density phase2 safety fine-tuning 后，`ft_best` 平均成功率超过原版：

| 模型 | 平均成功率 |
| --- | ---: |
| 原版 `checkpoint0004` | 81.67% |
| `ft_5000` | 82.47% |
| `ft_10000` | 82.56% |
| `ft_best` | 82.92% |

阶段性判断：

```text
memflight 结构本身不是无效，关键在于 phase1 权重还不是最终飞行最优模型。
full-density safety fine-tuning 能把记忆增强模型转化为更稳定的飞行策略。
```

## 2. 为什么低速 crash 低但 success 仍低

低速失败多数不是 crash，而是 timeout。

例如在 `target_speed=2` 时，如果 crash 率低但 timeout 高，说明模型不是避障失败，而是在给定 `timesteps=400` 内没到达目标。

原因主要有三点：

1. `target_speed` 只限制飞行速度，不重新生成场景。
2. `speed_mtp` 在 `env.reset()` 阶段已经影响目标点距离、障碍物纵向尺度和场景尺度。
3. 当场景按较高 `speed_mtp` 生成，但测评强制低 `target_speed` 飞行时，低速无人机可能来不及到达目标。

所以：

```text
低速成功率低 ≠ 低速避障能力差
低速成功率低可能主要是任务距离 / 时间预算 / 场景尺度不匹配。
```

## 3. 为什么高速也会 timeout

高速 timeout 看起来反直觉，但也是可能的。原因包括：

- 高速下模型更容易大幅绕障，路径变长，未必能及时进入 goal radius。
- 控制延迟和动作平滑会造成 overshoot 或来回修正。
- 模型可能为了避障降低实际前进速度，导致虽然 `target_speed` 高，但有效进度不足。
- 测评 success 要求到达目标，训练 success 主要看不碰撞，两者不完全一致。

因此高速失败可能分成两类：

```text
crash：安全性不足
timeout：到达效率不足或轨迹绕行过多
```

后续实验不能只看 success rate，应该同时报告：

- crash rate
- timeout rate
- avg active speed
- avg finish time
- avg min gap

## 4. “不要只靠 speed_mtp 随机生成”是什么意思

当前环境里速度分布大致来自：

```python
self.max_speed = (0.75 + 2.5 * rd) * self.speed_mtp
```

这意味着训练时的速度上限是由随机数 `rd` 和全局倍率 `speed_mtp` 间接生成的。

问题是：

```text
训练时并没有显式告诉模型：现在要测 2/4/6/8/10 m/s 哪一个速度档。
```

更稳的做法是显式采样目标速度：

```text
target_speed ∈ {2, 4, 6, 8, 10}
```

并把 `target_speed` 或速度档作为条件输入给模型，或者至少在训练中按速度档均衡采样。这样可以减少：

- 低速 timeout
- 高速 crash
- 对 `speed_mtp` 的隐式依赖

这个方向可以叫：

```text
speed-conditioned training / speed-aware evaluation alignment
```

## 5. 为什么训练时 speed_mtp=4.0，测试时常用 2.5

这不是说 2.5 一定更正确，而是两个协议目标不同。

使用 `speed_mtp=2.5`：

- 对齐 `eval_original_like.py` 的默认设置；
- 低速 2 m/s 不容易因为目标距离过长而 timeout；
- 更适合作为 original-like 复现评测。

使用 `speed_mtp=4.0`：

- 更接近原版训练配置和 memdpflight 训练配置；
- 场景尺度更大，对低速更苛刻；
- 更适合作为训练分布敏感性实验。

建议论文中写成两个 protocol：

```text
Protocol A: original-like eval, speed_mtp=2.5
Protocol B: train-distribution stress test, speed_mtp=4.0
```

不要把两者混在一个结论里。

## 6. 是否需要修改 CUDA env

从“少改动、可写论文”的角度，不建议大改物理模型或 CUDA kernel。

原论文的核心是：

- 简化点质点/加速度动力学；
- 控制延迟；
- 推力响应指数平滑；
- 速度目标接近；
- 障碍物避让；
- 控制平滑。

当前代码已经基本沿用了这套可微物理训练框架。直接改 physics kernel 风险较高，也会让对比不公平。

更推荐做保守改动：

### 6.1 解耦 scene scale 和 target speed

现状是 `speed_mtp` 同时影响：

- 训练速度上限；
- 目标距离；
- 障碍物尺度；
- 场景难度。

可以保守地拆成：

```text
scene_mtp: 控制场景尺度
target_speed: 控制飞行速度
```

这样低速测试不会被迫飞高速尺度场景。

### 6.2 增加 fixed-speed training mode

保留原始 `speed_mtp` 逻辑，同时新增可选参数：

```bash
--fixed_target_speeds 2,4,6,8,10
```

训练时按速度档采样，使训练分布和测评分布更一致。

### 6.3 强制 full-density phase2

这个已经初步验证有效。

建议把它正式变成训练阶段：

```text
Phase 0: base policy
Phase 1: memory reconstruction
Phase 2: full-density safety fine-tuning
```

这是目前最保守、收益最明确的改动。

## 7. Recon 从重建图像改成重建风险

当前 memflight 的 `recon_decoder` 更接近重建压缩视觉/深度特征。

问题是：

```text
重建深度图不等于知道哪里危险。
```

飞行真正需要的是：

- 最近障碍物距离；
- 前方 clearance；
- time-to-collision；
- 可通行方向；
- 未来若干步碰撞风险。

因此可以把辅助任务从：

```text
reconstruct depth / visual feature
```

改成：

```text
predict risk / clearance / collision probability
```

这是很有论文价值的方向，因为它更直接对应避障任务。

保守实现方案：

1. 不改主干模型。
2. 保留 GRU hidden state。
3. 把 `recon_decoder` 输出从 48 维深度特征改为几个风险量：

```text
min_gap_pred
front_clearance_pred
ttc_pred
collision_risk_pred
```

监督信号可以来自 CUDA env 已经能算的：

```python
vec_to_pt = env.find_vec_to_nearest_pt()
distance = norm(vec_to_pt) - env.margin
```

优点：

- 不需要真实标签；
- 不需要 ROS；
- 仍然是可微物理环境内生成监督；
- 更贴合飞行成功率。

这个方向可以叫：

```text
risk-aware memory supervision
```

## 8. Flightmare benchmark 和 CUDA eval 哪个成功率更高

不能简单判断固定谁更高。

一般来说：

- CUDA eval 是项目内部快速训练/评测环境，几何和深度由 CUDA kernel 生成；
- Flightmare benchmark 是 ROS + 渲染仿真环境，更接近论文正式比较场景；
- 两者的成功定义、轨迹、目标速度、视觉输入、场景结构可能都不同。

如果 Flightmare benchmark 是论文精心设定的同分布测试，原论文模型成功率会很高。

但如果 CUDA eval 的低速组合存在 timeout 问题，CUDA 下成功率可能低于论文图表。

所以更稳的表述是：

```text
当前 CUDA eval 结果不能直接对标 Figure 8。
它适合做训练机制、速度分布、场景分布和消融实验分析。
```

## 9. 可写论文的问题定义

一个更稳的论文问题不是“超过原论文 Figure 8”，而是：

```text
复现 DiffPhysDrone 的 CUDA 训练框架后，发现训练分布、速度条件和评测目标之间存在不一致。
这种不一致会导致低速 timeout 和高速 crash。
本文提出 memory + full-density safety fine-tuning / risk-aware auxiliary learning 来提升 original-like CUDA 多速度评测稳定性。
```

可以拆成三个贡献：

1. 分析原始可微物理训练在多速度测评下的 failure modes：
   - low-speed timeout
   - high-speed crash
   - speed_mtp sensitivity
2. 提出保守训练修正：
   - memory reconstruction
   - full-density phase2 safety fine-tuning
3. 提出进一步的 risk-aware memory supervision：
   - 用风险预测替代单纯视觉重建
   - 更直接服务避障

## 10. 后续实验优先级

### Priority 1: 固化当前有效结果

保留并记录：

```text
phase1_done
ft_5000
ft_10000
ft_best
original checkpoint0004
```

在同一协议下报告：

- success rate
- crash rate
- timeout rate
- avg speed

### Priority 2: speed_mtp 敏感性

比较：

```text
speed_mtp=2.5
speed_mtp=4.0
```

模型只测：

```text
original checkpoint0004
phase1_done
ft_best
```

目标是证明：

```text
phase2 safety fine-tuning 是否稳定改善，而不是只在一个 speed_mtp 下有效。
```

### Priority 3: full-density 训练消融

比较：

```text
phase1_done
phase2 fine-tune without force_full_density
phase2 fine-tune with force_full_density
```

这是目前最重要的消融。

### Priority 4: risk-aware auxiliary task

先不要大改模型。可以做一个最小版本：

```text
GRU hidden -> min_gap_pred
loss_risk = MSE(min_gap_pred, min_gap)
```

如果有效，再扩展到：

```text
front clearance
time-to-collision
collision probability
```

## 11. 现阶段最可行做法

综合“少改动、收益明确、容易写论文”，当前最推荐路线是：

```text
1. 不改 CUDA kernel。
2. 保留原可微物理损失。
3. 固化 full-density phase2 safety fine-tuning。
4. 做 speed_mtp 和 failure mode 分析。
5. 后续再尝试 risk-aware auxiliary head。
```

最保守版本的论文题目可以围绕：

```text
Speed- and Risk-Aware Fine-Tuning for Vision-Based Agile Flight in Differentiable Physics Simulation
```

或者中文：

```text
面向多速度视觉敏捷飞行的可微物理仿真安全微调方法
```

