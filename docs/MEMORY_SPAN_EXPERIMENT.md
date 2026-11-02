# V63 单头记忆重建跨度消融

共同起点：
weights/checkpoint_phase0_done.pth

共同训练日程：
- Phase 1：15000步，冻结动作头fc；训练stem、v_proj、GRU和decoder
- Phase 2：5000步，全参数联合训练
- 所有组总计20000步
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
