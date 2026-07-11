import argparse
import os
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from env_cuda import Env
from model import Model

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, required=True)
    parser.add_argument('--trials', type=int, default=5)
    parser.add_argument('--target_speed', type=float, default=8.0, help="Forced target speed")
    parser.add_argument('--goal_radius', type=float, default=5.0)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--timesteps', type=int, default=400)
    parser.add_argument('--grad_decay', type=float, default=0.4)
    parser.add_argument('--fov_x_half_tan', type=float, default=0.53)
    parser.add_argument('--cam_angle', type=int, default=10)
    parser.add_argument('--single', default=True, action='store_true')
    parser.add_argument('--no_odom', default=False, action='store_true')
    parser.add_argument('--save_gif', default=True, action='store_true')
    
    # 记忆参数
    parser.add_argument('--recon_members', nargs=3, type=int, default=[1, 1, 0])
    parser.add_argument('--recon_steps', type=int, default=12)
    parser.add_argument('--enable_v_scale', action='store_true', default=False)
    return parser.parse_args()

def run_single_trial(env, model, args, trial_idx):
    ctl_dt = 1 / 15
    h = None
    frames = []

    reached_goal = False
    crashed = False
    speed_samples = []
    peak_speed = 0.0

    with torch.no_grad():
        env.reset()
        env.max_speed = torch.full_like(env.max_speed, float(args.target_speed))
        model.reset()

        act_buffer = [env.act] * 2

        for t in range(args.timesteps):
            depth, _ = env.render(ctl_dt)

            if args.save_gif and trial_idx == 0:
                frame = depth[0].cpu().numpy()
                frame = 3 / frame.clip(0.3, 24) - 0.6
                lo, hi = frame.min(), frame.max()
                frame = (frame - lo) / (hi - lo + 1e-6)
                frames.append((frame * 255).astype(np.uint8))

            target_v_raw = env.p_target - env.p
            env.run(act_buffer[-1], ctl_dt, target_v_raw)

            vec_to_pt = env.find_vec_to_nearest_pt()
            gap = torch.norm(vec_to_pt, 2, -1) - env.margin

            speed = env.v[0].norm().item()
            dist = (env.p_target[0] - env.p[0]).norm().item()
            gap0 = gap[0].item()

            speed_samples.append(speed)
            peak_speed = max(peak_speed, speed)

            if dist <= args.goal_radius:
                reached_goal = True
                print(f"✅ Trial {trial_idx+1} reached goal! (peak={peak_speed:.2f} m/s)")
                break

            if gap0 < 0:
                crashed = True
                print(f"💥 Trial {trial_idx+1} crashed! (gap={gap0:.3f}, dist={dist:.2f} m)")
                break

            fwd = env.R[:, :, 0].clone()
            fwd[:, 2] = 0
            up = torch.zeros_like(fwd)
            up[:, 2] = 1
            fwd = F.normalize(fwd, 2, -1)
            R = torch.stack([fwd, torch.cross(up, fwd, dim=-1), up], -1)

            target_v_norm = torch.norm(target_v_raw, 2, -1, keepdim=True)
            target_v_unit = target_v_raw / target_v_norm.clamp_min(1e-6)
            target_v = target_v_unit * torch.minimum(target_v_norm, env.max_speed)

            local_v = torch.squeeze(env.v[:, None] @ R, 1)
            state_parts = [torch.squeeze(target_v[:, None] @ R, 1), env.R[:, 2], env.margin[:, None]]
            if not args.no_odom: state_parts.insert(0, local_v)
            state = torch.cat(state_parts, -1)

            x = 3 / depth.clamp_(0.3, 24) - 0.6
            x = F.max_pool2d(x[:, None], 4, 4)

            if args.enable_v_scale:
                act, _, h, _ = model(x, state, h)
            else:
                act, _, h = model(x, state, h)
                
            a_pred, v_pred, *_ = (R @ act.reshape(args.batch_size, 3, -1)).unbind(-1)
            act = (a_pred - v_pred - env.g_std) * env.thr_est_error[:, None] + env.g_std
            act_buffer.append(act)

    mean_speed = float(np.mean(speed_samples)) if speed_samples else 0.0
    return {
        'success': reached_goal and (not crashed),
        'mean_speed': mean_speed,
        'peak_speed': peak_speed,
        'frames': frames,
    }

def main():
    args = get_args()
    device = torch.device('cuda')
    print(f"Loading checkpoint: {args.resume}")

    env = Env(args.batch_size, 64, 48, args.grad_decay, device,
              fov_x_half_tan=args.fov_x_half_tan, single=args.single,
              cam_angle=args.cam_angle)

    dim_obs = 7 if args.no_odom else 10
    model = Model(dim_obs, 6, recon_members=tuple(bool(x) for x in args.recon_members), recon_steps=args.recon_steps).to(device)

    model.load_state_dict(torch.load(args.resume, map_location=device), strict=False)
    model.eval()

    trial_results = []
    frames = []
    for trial_idx in range(args.trials):
        result = run_single_trial(env, model, args, trial_idx)
        trial_results.append(result)
        if args.save_gif and trial_idx == 0:
            frames = result['frames']

    if args.save_gif and frames:
        os.makedirs('gifs', exist_ok=True)
        imageio.mimsave(f'gifs/eval_{args.target_speed}ms.gif', frames, fps=15)
        print(f"🎥 First trial recorded and saved to gifs/eval_{args.target_speed}ms.gif")

if __name__ == '__main__':
    main()