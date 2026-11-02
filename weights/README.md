# 模型权重说明

## 论文正式权重

- `memdp_past6_k6_seed20261028.pth`：论文最终MemDP Past-6模型。
- `original_dp_checkpoint0004.pth`：Original DP对比基线。
- `checkpoint_phase0_done.pth`：MemDP各消融组使用的共享训练起点。
- `SHA256SUMS`：模型文件完整性校验值。

最终MemDP配置为Past-6，即从GRU隐藏状态重建6步前的深度信息。在15 Hz
控制频率下，6步约对应0.4 s。历史重建系数为0.15，当前深度重建和速度
缩放分支均关闭。

目录中的其他历史权重仅用于旧实验追溯，不属于论文最终模型。
