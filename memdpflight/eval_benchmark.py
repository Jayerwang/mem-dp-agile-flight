import argparse
import os
import torch
import torch.nn.functional as F
from tqdm import trange
from env_cuda import Env
from model import Model

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Memory-Architecture Aggregate Evaluator")
    parser.add_argument("--resume", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--num_episodes", type=int, default=50, help="Number of batches to run")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--timesteps", type=int, default=400)
    parser.add_argument("--goal_radius", type=float, default=5.0)
    
    # 核心测试指标：目标速度
    parser.add_argument("--target_speed", type=float, default=8.0, help="Forced target speed (e.g. 2, 4, 6, 8)")

    parser.add_argument("--grad_decay", type=float, default=0.4)
    parser.add_argument("--fov_x_half_tan", type=float, default=0.53) # 保持与训练一致
    parser.add_argument("--cam_angle", type=float, default=10)        # 保持与训练一致
    parser.add_argument("--single", action="store_true", default=True)
    parser.add_argument("--gate", action="store_true", default=False)
    parser.add_argument("--ground_voxels", action="store_true", default=False) # 训练用的是False
    parser.add_argument("--scaffold", action="store_true", default=False)
    parser.add_argument("--random_rotation", action="store_true", default=False)
    parser.add_argument("--no_odom", action="store_true", default=False)
    parser.add_argument("--speed_mtp", type=float, default=1.0)
    
    # 记忆架构参数 (必须与训练时完全一致)
    parser.add_argument('--recon_members', nargs=3, type=int, default=[1, 1, 0])
    parser.add_argument('--recon_steps', type=int, default=12)
    parser.add_argument('--enable_v_scale', action='store_true', default=False)
    
    return parser.parse_args()

def load_policy(path: str, args, device: torch.device, dim_obs: int) -> Model:
    # 适配新的 Model 实例化参数
    model = Model(
        dim_obs=dim_obs, 
        dim_action=6,
        recon_members=tuple(bool(x) for x in args.recon_members),
        recon_steps=args.recon_steps,
        enable_v_scale=args.enable_v_scale
    ).to(device)
    
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

def run_episode(env: Env, model: Model, args: argparse.Namespace):
    bsz = args.batch_size
    device = env.device
    ctl_dt = 1.0 / 15.0

    env.reset()
    # 🌟 核心强制：将所有无人机的物理最高速度强制锁死为你想要测试的速度（例如 8.0）
    env.max_speed = torch.full_like(env.max_speed, float(args.target_speed))
    model.reset()

    h = None # GRU 初始隐藏状态
    done = torch.zeros(bsz, dtype=torch.bool, device=device)
    success = torch.zeros(bsz, dtype=torch.bool, device=device)
    act_buffer = [env.act] * 2
    margin = env.margin.reshape(-1)

    for t in range(args.timesteps):
        depth, _ = env.render(ctl_dt)
        target_v_raw = env.p_target - env.p
        env.run(act_buffer[-1], ctl_dt, target_v_raw)

        vec_to_pt = env.find_vec_to_nearest_pt()
        nearest_dist = torch.norm(vec_to_pt, 2, -1).min(dim=0).values
        gap = nearest_dist - margin
        dist = torch.norm(env.p_target - env.p, 2, -1)

        reached_goal = dist <= args.goal_radius
        crashed = gap < 0
        newly_done = (~done) & (reached_goal | crashed)
        success = success | (newly_done & reached_goal & (~crashed))
        done = done | newly_done

        if bool(torch.all(done)):
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
        state_parts = [
            torch.squeeze(target_v[:, None] @ R, 1),
            env.R[:, 2],
            margin[:, None],
        ]
        if not args.no_odom:
            state_parts.insert(0, local_v)
        state = torch.cat(state_parts, -1)

        x = 3.0 / depth.clamp_(0.3, 24.0) - 0.6
        x = F.max_pool2d(x[:, None], 4, 4)

        with torch.no_grad():
            if args.enable_v_scale:
                act, _, h, _ = model(x, state, h)
            else:
                act, _, h = model(x, state, h) # 接收记忆 h

        a_pred, v_pred, *_ = (R @ act.reshape(bsz, 3, -1)).unbind(-1)
        act = (a_pred - v_pred - env.g_std) * env.thr_est_error[:, None] + env.g_std
        act_buffer.append(act)
                
    return int(success.sum().item())

def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dim_obs = 7 if args.no_odom else 10

    env = Env(
        batch_size=args.batch_size, width=64, height=48, grad_decay=args.grad_decay,
        device=device, fov_x_half_tan=args.fov_x_half_tan, single=args.single,
        gate=args.gate, ground_voxels=args.ground_voxels, scaffold=args.scaffold,
        speed_mtp=args.speed_mtp, random_rotation=args.random_rotation, cam_angle=args.cam_angle,
    )
    model = load_policy(args.resume, args, device, dim_obs=dim_obs)

    total_trials = args.num_episodes * args.batch_size
    total_success = 0

    for idx in trange(args.num_episodes, desc=f"Testing at {args.target_speed}m/s"):
        total_success += run_episode(env, model, args)

    success_rate = total_success / total_trials
    print("\n" + "=" * 50)
    print(f"[{os.path.basename(args.resume)}] Evaluation Report")
    print(f"Target Speed: {args.target_speed:.1f} m/s")
    print(f"Total Trials: {total_trials}")
    print(f"Success Rate: {success_rate * 100:.2f}%")
    print("=" * 50)

if __name__ == "__main__":
    main()