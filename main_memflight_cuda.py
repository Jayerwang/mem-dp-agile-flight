from collections import defaultdict, deque
import math
import os
from random import normalvariate

from matplotlib import pyplot as plt
import torch
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import argparse

from env_cuda import Env
from model_memflight import Model


parser = argparse.ArgumentParser()
parser.add_argument('--resume', default=None)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_iters', type=int, default=50000)
parser.add_argument('--coef_v', type=float, default=1.0)
parser.add_argument('--coef_speed', type=float, default=0.2)
parser.add_argument('--coef_v_pred', type=float, default=2.0)
parser.add_argument('--coef_collide', type=float, default=2.0)
parser.add_argument('--coef_obj_avoidance', type=float, default=1.5)
parser.add_argument('--coef_d_acc', type=float, default=0.01)
parser.add_argument('--coef_d_jerk', type=float, default=0.001)
parser.add_argument('--coef_d_snap', type=float, default=0.0)
parser.add_argument('--coef_ground_affinity', type=float, default=0.0)
parser.add_argument('--coef_bias', type=float, default=0.0)
parser.add_argument('--coef_mem', type=float, default=0.25)
parser.add_argument('--coef_sigma', type=float, default=0.05)
parser.add_argument('--coef_gate', type=float, default=0.02)
parser.add_argument('--sigma_min', type=float, default=0.6)
parser.add_argument('--sigma_max', type=float, default=1.1)
parser.add_argument('--sigma_target', type=float, default=0.5)
parser.add_argument('--memory_k', type=int, default=4)
parser.add_argument('--perception_noise', type=float, default=0.02)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--grad_decay', type=float, default=0.4)
parser.add_argument('--speed_mtp', type=float, default=1.0)
parser.add_argument('--fov_x_half_tan', type=float, default=0.53)
parser.add_argument('--timesteps', type=int, default=150)
parser.add_argument('--cam_angle', type=int, default=10)
parser.add_argument('--single', default=False, action='store_true')
parser.add_argument('--gate', default=False, action='store_true')
parser.add_argument('--ground_voxels', default=False, action='store_true')
parser.add_argument('--scaffold', default=False, action='store_true')
parser.add_argument('--random_rotation', default=False, action='store_true')
parser.add_argument('--yaw_drift', default=False, action='store_true')
parser.add_argument('--no_odom', default=False, action='store_true')
parser.add_argument('--output_root', default='/root/autodl-tmp/dp_memflight_outputs')
parser.add_argument('--ckpt_every', type=int, default=2000)
parser.add_argument('--ckpt_points', type=str, default='333,1068,2215,3777,6170,8574,11350')
args = parser.parse_args()

ckpt_points = set()
for raw in args.ckpt_points.split(','):
    raw = raw.strip()
    if not raw:
        continue
    try:
        point = int(raw)
    except ValueError:
        continue
    if point > 0:
        ckpt_points.add(point)

os.makedirs(args.output_root, exist_ok=True)
log_dir = os.path.join(args.output_root, 'runs')
ckpt_dir = os.path.join(args.output_root, 'checkpoints')
os.makedirs(log_dir, exist_ok=True)
os.makedirs(ckpt_dir, exist_ok=True)

writer = SummaryWriter(log_dir=log_dir)
print(args)

device = torch.device('cuda')

env = Env(
    args.batch_size,
    64,
    48,
    args.grad_decay,
    device,
    fov_x_half_tan=args.fov_x_half_tan,
    single=args.single,
    gate=args.gate,
    ground_voxels=args.ground_voxels,
    scaffold=args.scaffold,
    speed_mtp=args.speed_mtp,
    random_rotation=args.random_rotation,
    cam_angle=args.cam_angle,
)

if args.no_odom:
    model = Model(7, 6)
else:
    model = Model(10, 6)
model = model.to(device)

optim = AdamW(model.parameters(), args.lr)
sched = CosineAnnealingLR(optim, args.num_iters, args.lr * 0.01)

# Legacy behavior: only restore model weights, do not restore optimizer/scheduler/iter.
if args.resume:
    checkpoint = torch.load(args.resume, map_location=device)
    state = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint

    model_state = model.state_dict()
    shape_mismatch = []
    filtered_state = {}
    for k, v in state.items():
        if k not in model_state:
            continue
        if model_state[k].shape != v.shape:
            shape_mismatch.append((k, tuple(v.shape), tuple(model_state[k].shape)))
            continue
        filtered_state[k] = v

    missing_keys, unexpected_keys = model.load_state_dict(filtered_state, False)
    if missing_keys:
        print('missing_keys:', missing_keys)
    if unexpected_keys:
        print('unexpected_keys:', unexpected_keys)
    if shape_mismatch:
        print('shape_mismatch_keys:', [k for k, _, _ in shape_mismatch])
    print(f'resume_note: loaded {len(filtered_state)} matched tensors only')


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


pbar = tqdm(range(args.num_iters), ncols=100)
B = args.batch_size
for i in pbar:
    env.reset()
    model.reset()

    p_history = []
    v_history = []
    target_v_history = []
    vec_to_pt_history = []
    v_preds = []

    h = None
    depth_buffer = deque(maxlen=max(args.memory_k + 1, 1))
    mem_loss_acc = torch.tensor(0.0, device=device)
    mem_loss_count = 0
    sigma_hist = []
    gate_hist = []
    mem_k_hist = []

    act_lag = 1
    act_buffer = [env.act] * (act_lag + 1)
    target_v_raw = env.p_target - env.p

    if args.yaw_drift:
        drift_av = torch.randn(B, device=device) * (5 * math.pi / 180 / 15)
        zeros = torch.zeros_like(drift_av)
        ones = torch.ones_like(drift_av)
        R_drift = torch.stack([
            torch.cos(drift_av), -torch.sin(drift_av), zeros,
            torch.sin(drift_av), torch.cos(drift_av), zeros,
            zeros, zeros, ones,
        ], -1).reshape(B, 3, 3)

    for t in range(args.timesteps):
        ctl_dt = normalvariate(1 / 15, 0.1 / 15)
        depth, _ = env.render(ctl_dt)
        p_history.append(env.p)
        vec_to_pt_history.append(env.find_vec_to_nearest_pt())

        if args.yaw_drift:
            target_v_raw = torch.squeeze(target_v_raw[:, None] @ R_drift, 1)
        else:
            target_v_raw = env.p_target - env.p.detach()
        env.run(act_buffer[t], ctl_dt, target_v_raw)

        R = env.R
        fwd = env.R[:, :, 0].clone()
        up = torch.zeros_like(fwd)
        fwd[:, 2] = 0
        up[:, 2] = 1
        fwd = F.normalize(fwd, 2, -1)
        R = torch.stack([fwd, torch.cross(up, fwd), up], -1)

        target_v_norm = torch.norm(target_v_raw, 2, -1, keepdim=True)
        target_v_unit = target_v_raw / target_v_norm.clamp_min(1e-6)
        target_v = target_v_unit * torch.minimum(target_v_norm, env.max_speed)
        state = [
            torch.squeeze(target_v[:, None] @ R, 1),
            env.R[:, 2],
            env.margin[:, None],
        ]
        local_v = torch.squeeze(env.v[:, None] @ R, 1)
        if not args.no_odom:
            state.insert(0, local_v)
        state = torch.cat(state, -1)

        x = 3 / depth.clamp_(0.3, 24) - 0.6 + torch.randn_like(depth) * args.perception_noise
        x = F.max_pool2d(x[:, None], 4, 4)

        act, aux, h = model(x, state, h)
        sigma = aux['sigma']
        sigma_hist.append(sigma.squeeze(-1))
        memory_gate = aux.get('memory_gate')
        if memory_gate is not None:
            gate_hist.append(memory_gate.squeeze(-1))

        # Scale aggressiveness with sigma to favor adaptive high-speed behavior.
        sigma_scale = args.sigma_min + (args.sigma_max - args.sigma_min) * sigma
        act = act * sigma_scale

        recon = aux.get('recon')
        depth_buffer.append(x.detach())
        if recon is not None and len(depth_buffer) >= args.memory_k:
            mem_target = depth_buffer[-args.memory_k]
            mem_loss_acc = mem_loss_acc + F.smooth_l1_loss(recon, mem_target)
            mem_loss_count += 1

        a_pred, v_pred, *_ = (R @ act.reshape(B, 3, -1)).unbind(-1)
        v_preds.append(v_pred)
        act_real = (a_pred - v_pred - env.g_std) * env.thr_est_error[:, None] + env.g_std
        act_buffer.append(act_real)

        v_history.append(env.v)
        target_v_history.append(target_v)

    p_history = torch.stack(p_history)
    loss_ground_affinity = p_history[..., 2].relu().pow(2).mean()
    act_buffer = torch.stack(act_buffer)

    v_history = torch.stack(v_history)
    v_history_cum = v_history.cumsum(0)
    v_history_avg = (v_history_cum[30:] - v_history_cum[:-30]) / 30
    target_v_history = torch.stack(target_v_history)
    delta_v = torch.norm(v_history_avg - target_v_history[1:1-30], 2, -1)
    loss_v = F.smooth_l1_loss(delta_v, torch.zeros_like(delta_v))

    v_preds = torch.stack(v_preds)
    loss_v_pred = F.mse_loss(v_preds, v_history.detach())

    target_v_history_norm = torch.norm(target_v_history, 2, -1)
    target_v_history_normalized = target_v_history / target_v_history_norm[..., None].clamp_min(1e-6)
    fwd_v = torch.sum(v_history * target_v_history_normalized, -1)
    loss_bias = F.mse_loss(v_history, fwd_v[..., None] * target_v_history_normalized) * 3

    jerk_history = act_buffer.diff(1, 0).mul(15)
    snap_history = F.normalize(act_buffer - env.g_std).diff(1, 0).diff(1, 0).mul(15**2)
    loss_d_acc = act_buffer.pow(2).sum(-1).mean()
    loss_d_jerk = jerk_history.pow(2).sum(-1).mean()
    loss_d_snap = snap_history.pow(2).sum(-1).mean()

    vec_to_pt_history = torch.stack(vec_to_pt_history)
    distance = torch.norm(vec_to_pt_history, 2, -1)
    distance = distance - env.margin
    with torch.no_grad():
        v_to_pt = (-torch.diff(distance, 1, 1) * 135).clamp_min(1)
    loss_obj_avoidance = barrier(distance[:, 1:], v_to_pt)
    loss_collide = F.softplus(distance[:, 1:].mul(-32)).mul(v_to_pt).mean()

    speed_history = v_history.norm(2, -1)
    loss_speed = F.smooth_l1_loss(fwd_v, target_v_history_norm)

    if mem_loss_count > 0:
        loss_mem = mem_loss_acc / mem_loss_count
    else:
        loss_mem = mem_loss_acc

    sigma_stack = torch.stack(sigma_hist)
    sigma_mean = sigma_stack.mean()
    sigma_target = torch.full_like(sigma_stack, args.sigma_target)
    loss_sigma = F.smooth_l1_loss(sigma_stack, sigma_target)

    if gate_hist:
        gate_stack = torch.stack(gate_hist)
        gate_mean = gate_stack.mean()
        loss_gate = gate_mean
    else:
        gate_mean = torch.tensor(0.0, device=device)
        loss_gate = gate_mean

    loss = (
        args.coef_v * loss_v
        + args.coef_obj_avoidance * loss_obj_avoidance
        + args.coef_bias * loss_bias
        + args.coef_d_acc * loss_d_acc
        + args.coef_d_jerk * loss_d_jerk
        + args.coef_d_snap * loss_d_snap
        + args.coef_speed * loss_speed
        + args.coef_v_pred * loss_v_pred
        + args.coef_collide * loss_collide
        + args.coef_ground_affinity * loss_ground_affinity
        + args.coef_mem * loss_mem
        + args.coef_sigma * loss_sigma
        + args.coef_gate * loss_gate
    )

    if torch.isnan(loss):
        print('loss is nan, exiting...')
        raise SystemExit(1)

    pbar.set_description_str(f'loss: {loss:.3f}')
    optim.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
    optim.step()
    sched.step()

    with torch.no_grad():
        avg_speed = speed_history.mean(0)
        success = torch.all(distance.flatten(0, 1) > 0, 0)

        smooth_dict({
            'loss': loss,
            'loss_v': loss_v,
            'loss_v_pred': loss_v_pred,
            'loss_obj_avoidance': loss_obj_avoidance,
            'loss_d_acc': loss_d_acc,
            'loss_d_jerk': loss_d_jerk,
            'loss_d_snap': loss_d_snap,
            'loss_bias': loss_bias,
            'loss_speed': loss_speed,
            'loss_collide': loss_collide,
            'loss_ground_affinity': loss_ground_affinity,
            'loss_mem': loss_mem,
            'loss_sigma': loss_sigma,
            'sigma_mean': sigma_mean,
              'loss_gate': loss_gate,
              'gate_mean': gate_mean,
            'success': success.sum() / B,
            'max_speed': speed_history.max(0).values.mean(),
            'avg_speed': avg_speed.mean(),
            'ar': (success * avg_speed).mean(),
        })

        if is_save_iter(i):
            fig_p, ax = plt.subplots()
            vis_idx = min(4, p_history.shape[1] - 1)
            p_vis = p_history[:, vis_idx].cpu()
            ax.plot(p_vis[:, 0], label='x')
            ax.plot(p_vis[:, 1], label='y')
            ax.plot(p_vis[:, 2], label='z')
            ax.legend()
            writer.add_figure('p_history', fig_p, i + 1)
        iter_id = i + 1
        should_save = (iter_id % args.ckpt_every == 0) or (iter_id in ckpt_points)
        if should_save:
            ckpt_path = os.path.join(ckpt_dir, f'memflight_checkpoint_iter{i + 1:05d}.pth')
            torch.save(model.state_dict(), ckpt_path)
            print('saved:', ckpt_path)
        if (i + 1) % 25 == 0:
            for k, v in scaler_q.items():
                writer.add_scalar(k, sum(v) / len(v), i + 1)
            # Emit a plain line log so nohup/tail can show realtime progress reliably.
            print(f"iter={i + 1}/{args.num_iters} loss={float(loss):.4f} success={float(success.sum() / B):.3f} max_speed={float(speed_history.max(0).values.mean()):.3f}", flush=True)
            scaler_q.clear()
