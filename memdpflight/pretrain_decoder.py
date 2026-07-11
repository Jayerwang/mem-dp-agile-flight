"""
pretrain_decoder.py  ——  新增脚本
离线预热记忆解码器，对应 MAVRL 的 trainvae.py 流程。

流程：
  1. 用已有初始策略收集深度图序列 (真实飞行画面)
  2. 展开 GRU，收集各时刻隐状态
  3. 目标降采样至 6x8 (48像素)，只训练 recon_decoder
  4. 保存权重，正式训练时 --resume 加载
"""

import argparse
import math
from collections import deque

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from env_cuda import Env
from model import Model


# =============================================================================
# argparse
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--resume',       required=True,
                    help='已有 checkpoint 路径（只用于初始化 model 权重）')
parser.add_argument('--n_collect',    type=int,   default=5000,
                    help='收集深度图序列的时间步数')
parser.add_argument('--epochs',       type=int,   default=200,
                    help='解码器预热训练轮数')
parser.add_argument('--recon_steps',  type=int,   default=12,
                    help='重建帧偏移，与正式训练保持一致')
parser.add_argument('--lr',           type=float, default=1e-3)
parser.add_argument('--batch_size',   type=int,   default=64)
parser.add_argument('--save',         type=str,   default='decoder_pretrain.pth',
                    help='预热后的完整 model 权重保存路径')
parser.add_argument('--grad_decay',   type=float, default=0.4)
parser.add_argument('--speed_mtp',    type=float, default=1.0)
parser.add_argument('--no_odom',      default=False, action='store_true')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================================
# 初始化
# =============================================================================

env = Env(
    args.batch_size, 64, 48, args.grad_decay, device,
    speed_mtp=args.speed_mtp,
)
# 预热阶段用 easy 环境（基础策略能安全飞行的环境）
env.curriculum_phase = 0

obs_dim = 7 if args.no_odom else 10
model = Model(
    dim_obs=obs_dim,
    dim_action=6,
    recon_members=(True, True, False),
    recon_steps=args.recon_steps,
    enable_v_scale=False,   # 预热阶段不需要速度头
).to(device)

# 加载已有 checkpoint（strict=False，兼容新增模块）
if args.resume:
    state_dict = torch.load(args.resume, map_location=device)
    
    # 🌟 新增这一行：强行剔除 checkpoint 里所有和 recon_decoder 相关的旧权重
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith('recon_decoder')}
    
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, False)
    if missing_keys:
        print("missing_keys:", missing_keys)

# 冻结除 recon_decoder 之外的所有参数
for name, param in model.named_parameters():
    param.requires_grad = ('recon_decoder' in name)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / {total:,}  (only recon_decoder)")

optim = Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=args.lr,
)
sched = CosineAnnealingLR(optim, args.epochs, args.lr * 0.05)


# =============================================================================
# 第一步：收集深度图序列 (使用模型真实策略)
# =============================================================================

print(f"\n[Step 1] Collecting {args.n_collect} steps of depth sequences ...")

RECON_STEPS    = args.recon_steps
depth_buf: deque = deque(maxlen=RECON_STEPS + 5)

# 存储 (hx, past_target, cur_target) 三元组，供离线训练
dataset = []   # list of (hx_detach, past_flat, cur_flat)

env.reset()
h = None
B = args.batch_size

for t in tqdm(range(args.n_collect), ncols=70):
    depth, _ = env.render(1 / 15)

    with torch.no_grad():
        # 1. 构建重建目标：核大小改为 8，输出 6x8=48 像素
        depth_norm = 3.0 / depth.clamp(0.3, 24) - 0.6
        depth_c    = F.avg_pool2d(depth_norm[:, None], 8, 8).squeeze(1)  
        depth_buf.append(depth_c.clone())

        # 2. 构建模型输入：stem 仍然需要 12x16，所以核大小保持 4
        x = F.max_pool2d(depth_norm[:, None], 4, 4)
        v_dummy = torch.zeros(B, obs_dim, device=device)

        # 3. 策略前向传播，获取动作和隐状态
        act_pred, _, h = model(x, v_dummy, h)
        hx_save = h.clone()

    # 记录用于训练的样本（有足够历史时才记录）
    if len(depth_buf) > RECON_STEPS:
        dataset.append((
            hx_save,                                # (B, 192)
            depth_buf[0].flatten(1).clone(),        # past target  (B, 48)
            depth_c.flatten(1).clone(),             # current target (B, 48)
        ))

    # 4. 转化为物理引擎动作并驱动环境 (真实飞行)
    with torch.no_grad():
        R = env.R
        a_pred, v_pred, *_ = (R @ act_pred.reshape(B, 3, -1)).unbind(-1)
        policy_act = (a_pred - v_pred - env.g_std) * env.thr_est_error[:, None] + env.g_std
        
    dummy_target_v = torch.zeros((B, 3), device=device)
    env.run(policy_act, 1 / 15, dummy_target_v)

    # 每 200 步 reset，避免飞出边界或陷入死胡同
    if (t + 1) % 200 == 0:
        env.reset()
        h = None
        depth_buf.clear()

print(f"Collected {len(dataset)} training samples  "
      f"(batch_size={args.batch_size}, each sample has {args.batch_size} drones)")


# =============================================================================
# 第二步：离线训练 recon_decoder
# =============================================================================

print(f"\n[Step 2] Pretraining recon_decoder for {args.epochs} epochs ...")

model.train()

best_loss = float('inf')

for epoch in range(args.epochs):
    total_loss    = 0.0
    total_samples = 0

    # 打乱数据集顺序
    import random
    random.shuffle(dataset)

    for hx, past_target, cur_target in dataset:
        # hx, past_target, cur_target 都已在 device 上

        # 过去帧重建
        recon_past = model.decode(hx)
        loss_past  = F.mse_loss(recon_past, past_target, reduction='mean')

        # 当前帧重建
        recon_cur = model.decode(hx)
        loss_cur  = F.mse_loss(recon_cur,  cur_target,  reduction='mean')

        loss = 0.6 * loss_past + 0.4 * loss_cur   # past 权重略高，与正式训练一致

        optim.zero_grad()
        loss.backward()
        optim.step()

        total_loss    += loss.item()
        total_samples += 1

    sched.step()
    avg_loss = total_loss / max(total_samples, 1)

    if (epoch + 1) % 10 == 0:
        print(f"  epoch {epoch+1:>4d}/{args.epochs}  "
              f"avg_loss={avg_loss:.5f}  lr={sched.get_last_lr()[0]:.2e}")

    if avg_loss < best_loss:
        best_loss = avg_loss

print(f"\nPretraining done. Best loss: {best_loss:.5f}")


# =============================================================================
# 第三步：恢复全部参数可训练，保存完整权重
# =============================================================================

# 恢复所有参数的 requires_grad（正式训练时由 set_training_phase 重新管理）
for param in model.parameters():
    param.requires_grad = True

torch.save(model.state_dict(), args.save)
print(f"Saved pretrained model → {args.save}")

# =============================================================================
# 简单验证：检查重建质量
# =============================================================================

print("\n[Verify] Running quick reconstruction check ...")
model.eval()
env.reset()
h = None
depth_buf.clear()

with torch.no_grad():
    for t in range(RECON_STEPS + 10):
        depth, _ = env.render(1 / 15)
        
        depth_norm = 3.0 / depth.clamp(0.3, 24) - 0.6
        depth_c    = F.avg_pool2d(depth_norm[:, None], 8, 8).squeeze(1) # 核大小 8
        depth_buf.append(depth_c.clone())
        
        x = F.max_pool2d(depth_norm[:, None], 4, 4)
        v_dummy = torch.zeros(B, obs_dim, device=device)
        
        act_pred, _, h = model(x, v_dummy, h)
        
        R = env.R
        a_pred, v_pred, *_ = (R @ act_pred.reshape(B, 3, -1)).unbind(-1)
        policy_act = (a_pred - v_pred - env.g_std) * env.thr_est_error[:, None] + env.g_std
        
        env.run(policy_act, 1/15, torch.zeros(B, 3, device=device))

    if len(depth_buf) > RECON_STEPS:
        past_target = depth_buf[0].flatten(1)
        recon_past  = model.decode(h)
        mae = (recon_past - past_target).abs().mean().item()
        print(f"Past recon MAE: {mae:.5f}")
        print("Verification passed ✓" if mae < 0.15 else
              "⚠ MAE > 0.15, consider more epochs or larger dataset")