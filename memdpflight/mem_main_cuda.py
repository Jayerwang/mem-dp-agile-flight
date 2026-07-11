"""
版本：v3_infinite_curriculum
修改日期：2024-04-27
修改内容：
1. 课程调度器门槛从 0.85 降到了 0.70，窗口 50 -> 15。
2. 解除了 target_v 的环境限速，支持无限提速。
预期目标：解决模型低速过拟合问题，冲刺 4m/s 以上极速。
"""



"""
main_cuda.py  ——  修改版
在原始训练循环基础上新增：
  1. CurriculumScheduler   : MAVRL 风格三阶段课程调度
  2. set_training_phase()  : 阶段性冻结/解冻模型子模块
  3. 记忆重建损失           : past + current 帧重建（对应 MAVRL [1,1,0]）
  4. 动态损失权重           : 随课程阶段调整 coef_v / coef_collide
  5. v_scale_head 集成      : 目标速度动态缩放（可选，由 --enable_v_scale 控制）

原有可微物理引擎（CUDA kernel）、Env、Model 调用接口完全不变。
"""

from collections import defaultdict, deque
import math
from random import normalvariate
from matplotlib import pyplot as plt
from env_cuda import Env
import torch
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from collections import deque

import argparse
from model import Model

best_success_rate = 0.0

# =============================================================================
# ① CurriculumScheduler (无限极速突破版 - Infinite Curriculum)
# =============================================================================
from collections import deque

class CurriculumScheduler:
    def __init__(self, window: int = 15):
        self.phase   = 0
        self.window  = window
        self._hist   = deque(maxlen=window)
        
        # 🌟 1. 目标门槛：平均胜率必须达到 97%
        self.upgrade_threshold = 0.97  
        self.max_phase = 4  
        
        # 🌟 2. 稳定性判定参数：
        self.stable_count = 0  # 当前已连续达标的次数
        self.required_stable_count = 20  # 【核心】必须连续 20 次评估都 >= 97% 才算“真正稳定”

    def state_dict(self):
        return {
            'phase': self.phase,
            'hist': list(self._hist),
            'stable_count': self.stable_count  # 🌟 记得保存连续达标次数
        }

    def load_state_dict(self, state):
        self.phase = state.get('phase', 0)
        self._hist = deque(state.get('hist', []), maxlen=self.window)
        self.stable_count = state.get('stable_count', 0)

    def step(self, success_rate: float) -> int:
        self._hist.append(float(success_rate))
        
        # 历史记录没存满 window 长度前，按兵不动
        if len(self._hist) < self.window:
            return self.phase
            
        # 计算近期平均胜率
        avg = sum(self._hist) / len(self._hist)
        
        # 🌟 3. 极其严苛的稳定性检测逻辑
        if avg >= self.upgrade_threshold:
            self.stable_count += 1  # 达标了，积攒 1 点进度
        else:
            self.stable_count = 0   # ❌ 只要有一次掉下 97%，前功尽弃，重新憋！
            
        # 🌟 4. 只有当连续稳定次数攒满，且没到满级时，才允许升级
        if self.stable_count >= self.required_stable_count and self.phase < self.max_phase:
            self.phase += 1
            self._hist.clear() 
            self.stable_count = 0  # 升级后清空计数器，重新适应新速度
            print(f"\n🚀 [自适应突破] 胜率已在 {self.upgrade_threshold*100}% 连续稳定！难度升级！当前 Phase → {self.phase} (近期均值={avg:.3f})")
            
        return self.phase

    @property
    def v_coef_mult(self) -> float: return 1.0 
    
    @property
    def collide_coef_mult(self) -> float: 
        return 1.5 
        
    @property
    def speed_mult(self) -> float: 
        # Phase 0=1.0x, 1=1.3x, 2=1.6x, 3=1.9x, 4=2.2x
        return 1.0 + self.phase * 0.3
# =============================================================================
# ② set_training_phase
# =============================================================================

def set_training_phase(model: Model, phase: int, verbose: bool = True) -> None:
    """
    phase 0  (iter < phase0_iters)
        冻结 recon_decoder / v_scale_head，只训练原有参数。
        让策略先收敛到合理基线，不被新模块梯度干扰。

    phase 1  (phase0_iters <= iter < phase1_iters)
        解冻 recon_decoder，冻结动作头 fc。
        仅让 GRU + 解码器 接受重建损失，不改变已有策略输出层。

    phase 2  (iter >= phase1_iters)
        全部参数可训练，各模块已有合理初值，联合训练。
    """
    has_recon  = hasattr(model, 'recon_decoder')
    has_vscale = hasattr(model, 'v_scale_head')

    # 动作头 fc：phase 1 时冻结，其余开放
    for p in model.fc.parameters():
        p.requires_grad = (phase != 1)

    if has_recon:
        for p in model.recon_decoder.parameters():
            p.requires_grad = (phase >= 1)

    if has_vscale:
        for p in model.v_scale_head.parameters():
            p.requires_grad = (phase >= 2)

    if verbose:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in model.parameters())
        frozen = []
        if phase == 1:            frozen.append('fc')
        if has_recon  and phase == 0: frozen.append('recon_decoder')
        if has_vscale and phase < 2:  frozen.append('v_scale_head')
        print(f"[Phase {phase}] trainable {trainable:,}/{total:,}  "
              f"frozen: {frozen or 'none'}")


# =============================================================================
# argparse
# =============================================================================

parser = argparse.ArgumentParser()
# ── 原有参数（完全不动）────────────────────────────────────
parser.add_argument('--resume',             default=None)
parser.add_argument('--batch_size',         type=int,   default=64)
parser.add_argument('--num_iters',          type=int,   default=50000)
parser.add_argument('--coef_v',             type=float, default=1.0)
parser.add_argument('--coef_speed',         type=float, default=0.0)
parser.add_argument('--coef_v_pred',        type=float, default=2.0)
parser.add_argument('--coef_collide',       type=float, default=2.0)
parser.add_argument('--coef_obj_avoidance', type=float, default=1.5)
parser.add_argument('--coef_d_acc',         type=float, default=0.01)
parser.add_argument('--coef_d_jerk',        type=float, default=0.001)
parser.add_argument('--coef_d_snap',        type=float, default=0.0)
parser.add_argument('--coef_ground_affinity', type=float, default=0.)
parser.add_argument('--coef_bias',          type=float, default=0.0)
parser.add_argument('--lr',                 type=float, default=1e-3)
parser.add_argument('--grad_decay',         type=float, default=0.4)
parser.add_argument('--speed_mtp',          type=float, default=1.0)
parser.add_argument('--fov_x_half_tan',     type=float, default=0.53)
parser.add_argument('--timesteps',          type=int,   default=150)
parser.add_argument('--cam_angle',          type=int,   default=10)
parser.add_argument('--single',             default=False, action='store_true')
parser.add_argument('--gate',               default=False, action='store_true')
parser.add_argument('--ground_voxels',      default=False, action='store_true')
parser.add_argument('--scaffold',           default=False, action='store_true')
parser.add_argument('--random_rotation',    default=False, action='store_true')
parser.add_argument('--yaw_drift',          default=False, action='store_true')
parser.add_argument('--no_odom',            default=False, action='store_true')

# ── 新增参数 ────────────────────────────────────────────────
parser.add_argument('--recon_members',      nargs=3, type=int,   default=[1, 1, 0],
                    help='过去/当前/未来 重建开关，对齐 MAVRL 最优配置 [1,1,0]')
parser.add_argument('--recon_steps',        type=int,   default=12,
                    help='重建帧偏移步数，15Hz 下 12 步 ≈ 0.8s 历史')
parser.add_argument('--recon_coef_past',    type=float, default=0.3,
                    help='过去帧重建损失系数')
parser.add_argument('--recon_coef_cur',     type=float, default=0.2,
                    help='当前帧重建损失系数')
parser.add_argument('--curriculum_window',  type=int,   default=50,
                    help='课程升级滑动均值窗口大小')
parser.add_argument('--phase0_iters',       type=int,   default=10000,
                    help='阶段 0（冻结新模块）持续的迭代数')
parser.add_argument('--phase1_iters',       type=int,   default=25000,
                    help='阶段 0+1 合计迭代数，超过后进入 phase 2（全参数可训练）')
parser.add_argument('--enable_v_scale',     default=False, action='store_true',
                    help='是否启用速度缩放头（v_scale_head）')

args = parser.parse_args()
writer = SummaryWriter()
print(args)

# =============================================================================
# 模型 / 环境 / 优化器初始化
# =============================================================================

device = torch.device('cuda')

env = Env(
    args.batch_size, 64, 48, args.grad_decay, device,
    fov_x_half_tan=args.fov_x_half_tan,
    single=args.single, gate=args.gate,
    ground_voxels=args.ground_voxels, scaffold=args.scaffold,
    speed_mtp=args.speed_mtp, random_rotation=args.random_rotation,
    cam_angle=args.cam_angle,
)

obs_dim = 7 if args.no_odom else 10
model = Model(
    dim_obs=obs_dim,
    dim_action=6,
    recon_members=tuple(bool(x) for x in args.recon_members),
    recon_steps=args.recon_steps,
    enable_v_scale=args.enable_v_scale,
).to(device)

if args.resume:
    state_dict = torch.load(args.resume, map_location=device)
    missing, unexpected = model.load_state_dict(state_dict, False)
    if missing:     print("missing_keys:",    missing)
    if unexpected:  print("unexpected_keys:", unexpected)

optim = AdamW(model.parameters(), args.lr)
sched = CosineAnnealingLR(optim, args.num_iters, args.lr * 0.01)

# ── 新增初始化 ──────────────────────────────────────────────
curriculum    = CurriculumScheduler(window=args.curriculum_window)
RECON_STEPS   = args.recon_steps
RECON_MEMBERS = [bool(x) for x in args.recon_members]
depth_history: deque = deque(maxlen=RECON_STEPS + 5)

# 阶段 0：冻结新模块
set_training_phase(model, phase=0)
_last_phase   = 0
_success      = torch.tensor(0.0)   # 首次 iter 占位

ctl_dt_base   = 1 / 15

# =============================================================================
# 辅助函数（原有）
# =============================================================================

scaler_q = defaultdict(list)
def smooth_dict(ori_dict):
    for k, v in ori_dict.items():
        scaler_q[k].append(float(v))

def barrier(x: torch.Tensor, v_to_pt):
    return (v_to_pt * (1 - x).relu().pow(2)).mean()

def is_save_iter(i):
    if i < 2000:
        return (i + 1) % 250 == 0
    return (i + 1) % 1000 == 0

# =============================================================================
# 训练主循环
# =============================================================================

pbar = tqdm(range(args.num_iters), ncols=90)
B    = args.batch_size
best_ar = 0.0  # 初始化历史最高综合收益

for i in pbar:

    # ── ① 阶段切换（基于 iter 数，稳定可靠）──────────────────
    target_phase = 0
    if i >= args.phase1_iters:
        target_phase = 2
    elif i >= args.phase0_iters:
        target_phase = 1

    if target_phase != _last_phase:
        # 🌟 新增：在切换到新阶段之前，立刻把上一个阶段的完美毕业状态存下来！
        torch.save(model.state_dict(), f'checkpoint_phase{_last_phase}_done.pth')
        print(f"\n💾 [Checkpoint] Phase {_last_phase} 毕业！权重已保存为 checkpoint_phase{_last_phase}_done.pth\n")
        set_training_phase(model, target_phase)
        _last_phase = target_phase
        # 重建优化器，避免旧动量影响新激活的参数
        optim = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            args.lr * (0.5 ** target_phase)   # 每次升级时略微降低 lr
        )
        sched = CosineAnnealingLR(
            optim, args.num_iters - i, args.lr * 0.01 * (0.5 ** target_phase)
        )
        print(f"[iter {i}] Optimizer rebuilt for phase {target_phase}")

# ── ② 课程阶段更新（success_rate 驱动）────────────────────
    if i > 0:
        phase = curriculum.step(_success)
        # 🌟 新增保护：环境密度最多升到 index 2 (100% 满载)，但阶段 phase 本身继续无限升！
        env.curriculum_phase = min(phase, 2)

    # ── ③ Episode 初始化 ──────────────────────────────────────
    env.reset()
    model.reset()

    p_history          = []
    v_history          = []
    target_v_history   = []
    vec_to_pt_history  = []
    v_preds            = []
    vid                = []
    v_net_feats        = []
    h                  = None

    # 每个 episode 清空深度历史和重建 loss 缓存
    depth_history.clear()
    recon_past_losses = []
    recon_cur_losses  = []

    act_lag    = 1
    act_buffer = [env.act] * (act_lag + 1)
    target_v_raw = env.p_target - env.p

    if args.yaw_drift:
        drift_av = torch.randn(B, device=device) * (5 * math.pi / 180 / 15)
        zeros = torch.zeros_like(drift_av)
        ones  = torch.ones_like(drift_av)
        R_drift = torch.stack([
            torch.cos(drift_av), -torch.sin(drift_av), zeros,
            torch.sin(drift_av),  torch.cos(drift_av), zeros,
            zeros, zeros, ones,
        ], -1).reshape(B, 3, 3)

    # ── ④ 时间步循环 ──────────────────────────────────────────
    for t in range(args.timesteps):
        ctl_dt = normalvariate(1 / 15, 0.1 / 15)
        depth, flow = env.render(ctl_dt)
        p_history.append(env.p)
        vec_to_pt_history.append(env.find_vec_to_nearest_pt())

        if is_save_iter(i):
            vid.append(depth[4])

        with torch.no_grad():
            depth_norm = 3.0 / depth.clamp(0.3, 24) - 0.6       
            # 核大小从 4 改为 8，输出尺寸变为 (B, 6, 8) 即 48 像素
            depth_c = F.avg_pool2d(depth_norm[:, None], 8, 8).squeeze(1) 
            depth_history.append(depth_c.clone())
            
        if args.yaw_drift:
            target_v_raw = torch.squeeze(target_v_raw[:, None] @ R_drift, 1)
        else:
            target_v_raw = env.p_target - env.p.detach()
        env.run(act_buffer[t], ctl_dt, target_v_raw)

        # 构建局部旋转矩阵
        R   = env.R
        fwd = env.R[:, :, 0].clone()
        up  = torch.zeros_like(fwd)
        fwd[:, 2] = 0
        up[:, 2]  = 1
        fwd = F.normalize(fwd, 2, -1)
        R   = torch.stack([fwd, torch.cross(up, fwd), up], -1)

        target_v_norm = torch.norm(target_v_raw, 2, -1, keepdim=True)
        target_v_unit = target_v_raw / target_v_norm
        
        # 🌟 核心修复：让喂给网络的导航目标速度，也跟随课程同步放大！
        dynamic_max_speed_t = env.max_speed * curriculum.speed_mult
        target_v      = target_v_unit * torch.minimum(target_v_norm, dynamic_max_speed_t)

        state = [
            torch.squeeze(target_v[:, None] @ R, 1),
            env.R[:, 2],
            env.margin[:, None],
        ]
        local_v = torch.squeeze(env.v[:, None] @ R, 1)
        if not args.no_odom:
            state.insert(0, local_v)
        state = torch.cat(state, -1)

        # 深度图预处理（与原有完全一致）
        x = 3 / depth.clamp_(0.3, 24) - 0.6 + torch.randn_like(depth) * 0.02
        x = F.max_pool2d(x[:, None], 4, 4)

        # ── 前向传播（根据 enable_v_scale 选择分支）──────────
        if args.enable_v_scale:
            act, values, h, v_scale = model(x, state, h)
            # 用 v_scale 缩放目标速度（v_scale 通过 detach 隔离，不影响主任务）
            target_v_scaled = target_v_unit * (
                torch.minimum(target_v_norm, env.max_speed) * v_scale
            )
            target_v_history.append(target_v_scaled)
        else:
            act, values, h = model(x, state, h)
            target_v_history.append(target_v)

        # ── 记忆重建损失（phase >= 1 才计算）─────────────────
        if _last_phase >= 1:
            # 过去帧重建
            #if RECON_MEMBERS[0] and len(depth_history) > RECON_STEPS:
             #   past_target = depth_history[0].flatten(1)   # 最旧的帧 (B, 192)
             # 过去帧重建
            if RECON_MEMBERS[0] and len(depth_history) > RECON_STEPS:
                # 使用负索引精确提取 RECON_STEPS 之前的帧
                past_target = depth_history[-(RECON_STEPS + 1)].flatten(1)
                recon_past  = model.decode(h)               # (B, 192)
                recon_past_losses.append(
                    F.mse_loss(recon_past, past_target, reduction='mean'))

            # 当前帧重建
            if RECON_MEMBERS[1]:
                cur_target = depth_c.flatten(1)
                recon_cur  = model.decode(h)
                recon_cur_losses.append(
                    F.mse_loss(recon_cur, cur_target, reduction='mean'))

        a_pred, v_pred, *_ = (R @ act.reshape(B, 3, -1)).unbind(-1)
        v_preds.append(v_pred)
        act = (a_pred - v_pred - env.g_std) * env.thr_est_error[:, None] + env.g_std
        act_buffer.append(act)
        v_net_feats.append(torch.cat([act, local_v, h], -1))

        v_history.append(env.v)

    # ── ⑤ Loss 计算区域 ──────────────────────────────────────
    p_history         = torch.stack(p_history)
    loss_ground_affinity = p_history[..., 2].relu().pow(2).mean()
    act_buffer        = torch.stack(act_buffer)

    v_history         = torch.stack(v_history)
    v_history_cum     = v_history.cumsum(0)
    v_history_avg     = (v_history_cum[30:] - v_history_cum[:-30]) / 30
    target_v_history  = torch.stack(target_v_history)
    T, B_now, _       = v_history.shape

    delta_v  = torch.norm(v_history_avg - target_v_history[1:1-30], 2, -1)
    loss_v   = F.smooth_l1_loss(delta_v, torch.zeros_like(delta_v))

    v_preds      = torch.stack(v_preds)
    loss_v_pred  = F.mse_loss(v_preds, v_history.detach())

    target_v_norm_h     = torch.norm(target_v_history, 2, -1)
    target_v_history_nd = target_v_history / target_v_norm_h[..., None]
    fwd_v               = torch.sum(v_history * target_v_history_nd, -1)
    loss_bias           = F.mse_loss(
        v_history, fwd_v[..., None] * target_v_history_nd) * 3

    jerk_history = act_buffer.diff(1, 0).mul(15)
    snap_history = F.normalize(act_buffer - env.g_std).diff(1, 0).diff(1, 0).mul(15**2)
    loss_d_acc   = act_buffer.pow(2).sum(-1).mean()
    loss_d_jerk  = jerk_history.pow(2).sum(-1).mean()
    loss_d_snap  = snap_history.pow(2).sum(-1).mean()

    vec_to_pt_history = torch.stack(vec_to_pt_history)
    distance          = torch.norm(vec_to_pt_history, 2, -1) - env.margin
    with torch.no_grad():
        v_to_pt = (-torch.diff(distance, 1, 1) * 135).clamp_min(1)
    loss_obj_avoidance = barrier(distance[:, 1:], v_to_pt)
    loss_collide       = F.softplus(distance[:, 1:].mul(-32)).mul(v_to_pt).mean()

    speed_history = v_history.norm(2, -1)
    loss_speed    = F.smooth_l1_loss(fwd_v, target_v_norm_h)

    # ── 新增：记忆重建损失汇总 ──────────────────────────────
    loss_recon = torch.tensor(0.0, device=device)
    if recon_past_losses:
        loss_recon = loss_recon + args.recon_coef_past * torch.stack(recon_past_losses).mean()
    if recon_cur_losses:
        loss_recon = loss_recon + args.recon_coef_cur  * torch.stack(recon_cur_losses).mean()

    # ── 新增：课程动态权重 ──────────────────────────────────
    w_v       = args.coef_v       * curriculum.v_coef_mult
    w_collide = args.coef_collide * curriculum.collide_coef_mult

    # ── Loss 合并（新增 loss_recon，替换动态权重）─────────────
    loss = (w_v            * loss_v
            + w_collide    * loss_collide
            + args.coef_obj_avoidance   * loss_obj_avoidance
            + args.coef_bias            * loss_bias
            + args.coef_d_acc           * loss_d_acc
            + args.coef_d_jerk          * loss_d_jerk
            + args.coef_d_snap          * loss_d_snap
            + args.coef_speed           * loss_speed
            + args.coef_v_pred          * loss_v_pred
            + args.coef_ground_affinity * loss_ground_affinity
            + loss_recon)

    if torch.isnan(loss):
        print("loss is nan, exiting...")
        exit(1)

    pbar.set_description_str(
        f'loss:{loss:.3f} ph:{_last_phase} rc:{loss_recon:.4f}')

    optim.zero_grad()
    loss.backward()
    optim.step()
    sched.step()

    # ── ⑥ 无梯度统计 + TensorBoard ──────────────────────────
    with torch.no_grad():
        avg_speed = speed_history.mean(0)
        success   = torch.all(distance.flatten(0, 1) > 0, 0)
        _success  = success.sum() / B      # 供下次 iter 的 curriculum.step() 使用

        smooth_dict({
            # 原有指标
            'loss':                   loss,
            'loss_v':                 loss_v,
            'loss_v_pred':            loss_v_pred,
            'loss_obj_avoidance':     loss_obj_avoidance,
            'loss_d_acc':             loss_d_acc,
            'loss_d_jerk':            loss_d_jerk,
            'loss_d_snap':            loss_d_snap,
            'loss_bias':              loss_bias,
            'loss_speed':             loss_speed,
            'loss_collide':           loss_collide,
            'loss_ground_affinity':   loss_ground_affinity,
            'success':                _success,
            'max_speed':              speed_history.max(0).values.mean(),
            'avg_speed':              avg_speed.mean(),
            'ar':                     (success * avg_speed).mean(),
            # 新增监控指标
            'loss_recon':             loss_recon,
            'curriculum_phase':       float(_last_phase),
            'curriculum_phase_sr':    float(curriculum.phase),
            'w_v':                    w_v,
            'w_collide':              w_collide,
        })

        if is_save_iter(i):
            fig_p, ax = plt.subplots()
            ph = p_history[:, 4].cpu()
            ax.plot(ph[:, 0], label='x')
            ax.plot(ph[:, 1], label='y')
            ax.plot(ph[:, 2], label='z')
            ax.legend()
            fig_v, ax = plt.subplots()
            vh = v_history[:, 4].cpu()
            ax.plot(vh[:, 0], label='x')
            ax.plot(vh[:, 1], label='y')
            ax.plot(vh[:, 2], label='z')
            ax.legend()
            fig_a, ax = plt.subplots()
            ah = act_buffer[:, 4].cpu()
            ax.plot(ah[:, 0], label='x')
            ax.plot(ah[:, 1], label='y')
            ax.plot(ah[:, 2], label='z')
            ax.legend()
            writer.add_figure('p_history', fig_p, i + 1)
            writer.add_figure('v_history', fig_v, i + 1)
            writer.add_figure('a_reals',   fig_a, i + 1)
            plt.close('all')

        if (i + 1) % 5000 == 0:
            ckpt_idx = (i + 1) // 5000
            torch.save(model.state_dict(),      f'checkpoint{ckpt_idx:05d}.pth')
            torch.save(curriculum.state_dict(), f'curriculum{ckpt_idx:05d}.pth')
    # 🌟 新增：基于综合收益 (AR) 的最优权重保存逻辑
        current_sr = float(_success)
        current_speed = float(avg_speed.mean())
        current_ar = current_sr * current_speed  # 核心：胜率和速度综合考量

        # 只有在有一定速度基础，且【AR创新高】时才保存
        if current_ar > best_ar and current_speed > 1.0: # 记得在循环最外面初始化 best_ar = 0.0
            best_ar = current_ar
            torch.save(model.state_dict(), 'checkpoint_best.pth')
            torch.save(curriculum.state_dict(), 'curriculum_best.pth') # 把课表也存一份最好的
            print(f"👉 [New Best AR!] {current_ar:.3f} (SR: {current_sr:.3f}, Speed: {current_speed:.2f} m/s) -> Checkpoint saved!")
        if (i + 1) % 25 == 0:
            for k, v in scaler_q.items():
                writer.add_scalar(k, sum(v) / len(v), i + 1)
            scaler_q.clear()