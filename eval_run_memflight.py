import argparse
import os

import imageio
import numpy as np
import torch
import torch.nn.functional as F

from env_cuda import Env
from model_memflight import Model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, required=True)
    parser.add_argument('--trials', type=int, default=10)
    parser.add_argument('--goal_radius', type=float, default=5.0)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--timesteps', type=int, default=400)
    parser.add_argument('--grad_decay', type=float, default=0.4)
    parser.add_argument('--speed_mtp', type=float, default=1.0)
    parser.add_argument('--fov_x_half_tan', type=float, default=0.53)
    parser.add_argument('--cam_angle', type=int, default=10)
    parser.add_argument('--single', default=False, action='store_true')
    parser.add_argument('--gate', default=False, action='store_true')
    parser.add_argument('--ground_voxels', default=False, action='store_true')
    parser.add_argument('--scaffold', default=False, action='store_true')
    parser.add_argument('--random_rotation', default=False, action='store_true')
    parser.add_argument('--no_odom', default=False, action='store_true')
    parser.add_argument('--save_gif', default=False, action='store_true')
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
            env.run(act_buffer[t], ctl_dt, target_v_raw)

            vec_to_pt = env.find_vec_to_nearest_pt()
            gap = torch.norm(vec_to_pt, 2, -1) - env.margin

            speed = env.v[0].norm().item()
            dist = (env.p_target[0] - env.p[0]).norm().item()
            gap0 = gap[0].item()

            speed_samples.append(speed)
            peak_speed = max(peak_speed, speed)

            if dist <= args.goal_radius:
                reached_goal = True
                print(
                    f"Trial {trial_idx + 1}/{args.trials} reached goal at step {t + 1} "
                    f"(dist={dist:.2f} m, peak={peak_speed:.2f} m/s)"
                )
                break

            if gap0 < 0:
                crashed = True
                print(
                    f"Trial {trial_idx + 1}/{args.trials} crashed at step {t + 1} "
                    f"(gap={gap0:.3f}, dist={dist:.2f} m)"
                )
                break

            fwd = env.R[:, :, 0].clone()
            fwd[:, 2] = 0
            up = torch.zeros_like(fwd)
            up[:, 2] = 1
            fwd = F.normalize(fwd, 2, -1)
            R = torch.stack([fwd, torch.cross(up, fwd, dim=-1), up], -1)

            target_v_norm = torch.norm(target_v_raw, 2, -1, keepdim=True)
            target_v_unit = target_v_raw / target_v_norm
            target_v = target_v_unit * torch.minimum(target_v_norm, env.max_speed)

            local_v = torch.squeeze(env.v[:, None] @ R, 1)
            state_parts = [
                torch.squeeze(target_v[:, None] @ R, 1),
                env.R[:, 2],
                env.margin[:, None],
            ]
            if not args.no_odom:
                state_parts.insert(0, local_v)
            state = torch.cat(state_parts, -1)

            x = 3 / depth.clamp_(0.3, 24) - 0.6
            x = F.max_pool2d(x[:, None], 4, 4)

            act, _, h = model(x, state, h)
            a_pred, v_pred, *_ = (R @ act.reshape(args.batch_size, 3, -1)).unbind(-1)
            act = (a_pred - v_pred - env.g_std) * env.thr_est_error[:, None] + env.g_std
            act_buffer.append(act)

            if (t + 1) % 50 == 0:
                print(
                    f"Trial {trial_idx + 1}/{args.trials} | Step {t + 1:4d}/{args.timesteps} "
                    f"| speed={speed:.2f} m/s | dist_to_goal={dist:.2f} m | gap={gap0:.3f}"
                )

    mean_speed = float(np.mean(speed_samples)) if speed_samples else 0.0
    success = reached_goal and (not crashed)
    return {
        'success': success,
        'reached_goal': reached_goal,
        'crashed': crashed,
        'mean_speed': mean_speed,
        'peak_speed': peak_speed,
        'frames': frames,
    }


def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B = args.batch_size

    print(f"Device: {device}")
    print(f"Loading checkpoint: {args.resume}")

    env = Env(B, 64, 48, args.grad_decay, device,
              fov_x_half_tan=args.fov_x_half_tan,
              single=args.single,
              gate=args.gate,
              ground_voxels=args.ground_voxels,
              scaffold=args.scaffold,
              speed_mtp=args.speed_mtp,
              random_rotation=args.random_rotation,
              cam_angle=args.cam_angle)

    dim_obs = 7 if args.no_odom else 10
    model = Model(dim_obs, 6).to(device)

    checkpoint = torch.load(args.resume, map_location=device)
    state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print('missing_keys:', missing)
    if unexpected:
        print('unexpected_keys:', unexpected)

    model.eval()
    print('Checkpoint loaded. Starting evaluation...')
    print(f'Protocol: trials={args.trials}, goal_radius={args.goal_radius}m')

    trial_results = []
    frames = []
    for trial_idx in range(args.trials):
        result = run_single_trial(env, model, args, trial_idx)
        trial_results.append(result)
        if args.save_gif and trial_idx == 0:
            frames = result['frames']

    success_trials = [r for r in trial_results if r['success']]
    success_rate = 100.0 * len(success_trials) / max(1, len(trial_results))
    mean_speed_on_success = (
        float(np.mean([r['mean_speed'] for r in success_trials]))
        if success_trials else 0.0
    )
    peak_speed = float(np.max([r['peak_speed'] for r in trial_results])) if trial_results else 0.0

    print('\n=== Summary ===')
    print(f'success_rate: {success_rate:.1f}% ({len(success_trials)}/{len(trial_results)})')
    print(f'mean_speed_on_success: {mean_speed_on_success:.2f} m/s')
    print(f'peak_speed: {peak_speed:.2f} m/s')
    print('Evaluation complete.')

    if args.save_gif and frames:
        os.makedirs('gifs', exist_ok=True)
        save_path = 'gifs/result.gif'
        imageio.mimsave(save_path, frames, fps=15)
        print(f'GIF saved to {save_path}')


if __name__ == '__main__':
    main()
