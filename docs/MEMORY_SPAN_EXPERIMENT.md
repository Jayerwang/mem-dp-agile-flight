# MemDP单解码头历史跨度消融实验

共同起点：
weights/checkpoint_phase0_done.pth

共同训练日程：
- 共享起点：加载已经训练完成的Phase 0权重，本轮不重复训练Phase 0
- Phase 1：3000次有效更新，冻结动作头fc；训练stem、v_proj、GRU和decoder
- Phase 2：2000次有效更新，全参数联合训练
- 本轮新增训练总计5000次更新
- 控制损失在两个阶段始终启用
- No-Recon也采用相同阶段、步数和冻结日程

实验组：
- no_recon: 无重建
- current: 当前深度重建，coef=0.15
- past6: 前6步深度重建，coef=0.15
- past12: 前12步深度重建，coef=0.15
- past24: 前24步深度重建，coef=0.15

单头约束：
每个重建模型只拟合一种目标，不同时拟合当前帧和历史帧。

共同环境：
- batch_size=64
- timesteps=150
- fixed_curriculum_phase=2
- speed_mtp=4.0
- fov_x_half_tan=0.82
- cam_angle=20
- ground_voxels
- random_rotation
- yaw_drift
- single

控制损失：
- coef_v=1.0
- coef_v_pred=2.0
- coef_collide=7.5
- coef_obj_avoidance=3.0
- coef_d_acc=0.01
- coef_d_jerk=0.001
- coef_speed=0.0

第一轮训练种子：20261028


## 阶段定义

Phase 1和Phase 2表示参数冻结与解冻的优化阶段，不等同于环境难度课程。
正式实验通过`fixed_curriculum_phase=2`固定使用最高难度环境，不进行自适应
环境阶段切换。Phase 2对应的损失权重倍率仍按原训练实现生效。

在15 Hz控制频率下：

- Past-6约对应0.4 s历史跨度；
- Past-12约对应0.8 s历史跨度；
- Past-24约对应1.6 s历史跨度。
