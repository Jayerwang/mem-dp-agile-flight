# Mem-DP：基于历史深度重建的无人机敏捷避障

本仓库提供 Mem-DP 的训练、消融实验和配对评测代码。项目基于
《Learning Vision-based Agile Flight via Differentiable Physics》
公开的可微物理无人机训练框架进行扩展。

本文最终模型记为 **MemDP Past-6**。模型在 GRU 循环策略基础上，
引入历史深度重建辅助监督，以约束循环隐藏状态保留与避障相关的历史几何信息。

在实验使用的 15 Hz 控制频率下，6步历史偏移约对应0.4秒。

## 一、最终模型配置

```text
模型名称            MemDP Past-6
recon_members       [1, 0, 0]（历史/当前/未来）
recon_steps         6
recon_coef_past     0.15
recon_coef_cur      0.0
phase1_recon        on
phase2_recon        on
训练随机种子        20261028
```
重建解码器仅用于训练阶段的辅助监督。实际飞行动作由循环策略及动作头输出。
## 二、目录结构
```text
memdp/
  model.py
      最终MemDP网络结构
  env_cuda.py
      CUDA可微物理训练环境
  train.py
      分阶段训练入口

evaluation/
  paired_evaluation.py
      统一场景配对评测器
  scenes/large_obstacle.py
      随机大型障碍物场景
  scenes/staggered_double_wall.py
      错位双墙和偏置开口场景

scripts/
  train_memdp_past6.sh
      复现最终Past-6模型
  train_span_ablation.sh
      重建目标与历史跨度消融
  evaluate_all_scenes.sh
      三类环境正式评测

weights/
  checkpoint_phase0_done.pth
      各消融组共享的训练起点
  memdp_past6_k6_seed20261028.pth
      最终MemDP Past-6权重
  original_dp_checkpoint0004.pth
      原始DP基线权重
  SHA256SUMS
      权重文件校验值
```
仓库根目录的 model.py 和 env_cuda.py 分别用于原始 DP 基线模型
和统一评测环境。

memdpflight/ 是仓库早期实现，为保持历史兼容性而保留。新的训练和
评测实验应优先使用 memdp/、`evaluation/` 和 scripts/。
## 三、运行环境
本仓库已在以下环境中完成验证：
- Python 3.10.20
- PyTorch 2.2.2
- CUDA Runtime 11.8
- CUDA Toolkit 11.8
- NVIDIA GeForce RTX 4090
其他支持 CUDA 的 NVIDIA GPU 也可能运行，但所需显存和编译兼容性
取决于具体硬件环境。

安装 Python 依赖：
    pip install -r requirements.txt
编译可微物理 CUDA 扩展：
    pip install -e src
检查 CUDA 扩展：
    python src/test.py
## 四、校验发布权重
    cd weights
    sha256sum -c SHA256SUMS
    cd ..
正常情况下，三个文件均应显示：
    OK
## 五、训练最终 MemDP Past-6
### 5.1 冒烟测试
冒烟测试仅用于检查训练代码、阶段切换和权重保存能否正常运行：
    bash scripts/train_memdp_past6.sh smoke 20261102
该模式只执行少量迭代，不能用于评价模型性能。
### 5.2 正式配置
    bash scripts/train_memdp_past6.sh quick 20261028
正式实验包含两个阶段：
1. Phase 1：记忆适配，训练3000次更新；
2. Phase 2：联合优化，训练2000次更新。
主要配置为：
    batch_size = 64
    timesteps  = 150
    Past-K     = 6
    seed       = 20261028
所有训练从：
    weights/checkpoint_phase0_done.pth
开始。

训练结果保存在 runs/ 下。为防止误覆盖已有实验，脚本在检测到同名
输出目录时会主动停止。

请勿使用下面的裸命令复现最终模型：
    python -m memdp.train
该命令采用通用默认参数，并不等于论文实验配置。正式复现应使用
scripts/train_memdp_past6.sh。
## 六、历史深度跨度消融实验
执行完整消融：
    bash scripts/train_span_ablation.sh quick 20261028
该脚本从同一训练起点出发，对比以下配置：
| 配置名称 | 重建目标 | 历史偏移 |
|---|---|---:|
| No-Recon | 不使用重建监督 | — |
| Current | 当前深度 | 0步 |
| Past-6 | 历史深度 | 6步 |
| Past-12 | 历史深度 | 12步 |
| Past-24 | 历史深度 | 24步 |


所有组使用相同的：
- 初始化权重；
- 训练随机种子；
- 两阶段训练步数；
- 控制损失配置；
- 环境及相机参数。
详细定义见：

[历史跨度实验说明](docs/MEMORY_SPAN_EXPERIMENT.md)
## 七、三类环境配对评测
运行 Original DP 与 MemDP Past-6 的正式对比：
    bash scripts/evaluate_all_scenes.sh
评测包含三类环境：
1. 原始随机混合障碍环境；
2. Medium、Hard 难度的随机大型障碍物环境；
3. Medium、Hard 难度的错位双墙环境。
评测目标速度为：
    4 m/s、6 m/s、8 m/s
同一实验条件下，两种模型使用完全相同的初始场景。每个结果目录都会
生成 pairing_audit.json。有效的配对实验应满足：
    {
      "mismatch_count": 0,
      "pairing_passed": true
    }
评测器输出的主要指标包括：
- 避障成功率；
- 碰撞率；
- 超时率；
- 实际飞行速度；
- 朝目标方向的进展速度；
- 最终目标距离；
- 单轨迹评测记录。
生成结果保存在 results/ 下，该目录默认不提交到 Git。
## 八、已完成的复现检查
当前发布版本已经完成以下验证：
- Original DP 权重严格加载；
- MemDP Past-6 权重严格加载；
- Phase 1训练流程验证；
- Phase 2训练流程验证；
- 原始混合障碍环境推理验证；
- 随机大型障碍环境推理验证；
- 错位双墙环境推理验证；
- 场景哈希配对审计；
- 核心复现文件的服务器绝对路径清理；
- 权重 SHA-256 完整性校验。
## 九、实验范围说明
本仓库的配对实验运行于轻量级 CUDA 可微物理环境，适合进行：
- 控制变量消融；
- 历史重建跨度比较；
- 大规模配对轨迹测试；
- 快速训练与评测迭代。
该环境的结果不应直接表述为 Flightmare 高保真仿真结果或真实无人机实验结果。
## 十、致谢
本项目基于以下工作公开的训练框架：
    @article{zhang2025learning,
      title={Learning vision-based agile flight via differentiable physics},
      author={Zhang, Yuang and Hu, Yu and Song, Yunlong and Zou, Danping and Lin, Weiyao},
      journal={Nature Machine Intelligence},
      year={2025},
      publisher={Nature Publishing Group}
    }
历史重建辅助监督与分阶段训练的设计也参考了已有的记忆增强视觉导航研究，
包括 MAVRL 等工作。
