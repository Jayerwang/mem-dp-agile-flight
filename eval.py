import argparse
import torch
import numpy as np
import imageio
import os
import sys

# 1. 导入环境和模型
try:
    from env_cuda import Env
    from model import Model
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    sys.exit(1)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, required=True, help='模型权重路径')
    parser.add_argument('--resource_dir', type=str, required=True, help='地图资源目录')
    parser.add_argument('--save_visualization', action='store_true', default=True)
    parser.add_argument('--target_speed', type=float, default=2.5)

    # 维度参数 (必须与 checkpoint 一致)
    parser.add_argument('--dim_obs', type=int, default=10, help='模型需要的状态维度')
    parser.add_argument('--dim_action', type=int, default=6, help='模型输出的动作维度')

    # 环境参数 (默认 64x48)
    parser.add_argument('--env_width', type=int, default=64)
    parser.add_argument('--env_height', type=int, default=48)
    
    # 其他默认参数
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--fov_x_half_tan', type=float, default=0.53)
    parser.add_argument('--cam_angle', type=float, default=10)
    parser.add_argument('--grad_decay', type=float, default=0.4)
    parser.add_argument('--single', action='store_true', default=True)
    parser.add_argument('--gate', action='store_true', default=False)
    parser.add_argument('--ground_voxels', action='store_true', default=False)
    parser.add_argument('--scaffold', action='store_true', default=False)
    parser.add_argument('--random_rotation', action='store_true', default=False)
    parser.add_argument('--speed_mtp', type=float, default=1.0)

    args = parser.parse_args()
    return args

def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"🚀 初始化环境 ({args.env_width}x{args.env_height})...")
    
    # 严格按照旧版签名初始化
    try:
        env = Env(
            batch_size=1, 
            width=args.env_width, 
            height=args.env_height, 
            grad_decay=args.grad_decay, 
            device=device,
            fov_x_half_tan=args.fov_x_half_tan,
            single=args.single,
            gate=args.gate,
            ground_voxels=args.ground_voxels,
            scaffold=args.scaffold,
            speed_mtp=args.speed_mtp,
            random_rotation=args.random_rotation,
            cam_angle=args.cam_angle
        )
    except Exception as e:
        print(f"❌ 环境初始化失败: {e}")
        return

    print(f"📥 加载模型 (Input={args.dim_obs}, Output={args.dim_action})...")
    
    # 初始化模型
    try:
        policy = Model(dim_obs=args.dim_obs, dim_action=args.dim_action).to(device)
    except TypeError:
        policy = Model().to(device)

    # 加载权重
    if not os.path.exists(args.resume):
        print(f"❌ 找不到文件: {args.resume}")
        return

    checkpoint = torch.load(args.resume, map_location=device)
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        policy.load_state_dict(checkpoint['model'])
    else:
        policy.load_state_dict(checkpoint)
    
    policy.eval()
    print("✅ 准备起飞...")

    frames = []
    
    # 重置环境
    try:
        obs = env.reset()
    except:
        obs = env.reset(1) # 尝试传参

    if obs is None:
        print("❌ 致命错误: env.reset() 返回 None。请确保 env_cuda.py 已恢复为原始版本。")
        return

    hx = None 
    
    for i in range(400):
        
        # --- 智能数据解包与补齐 ---
        if isinstance(obs, (list, tuple)):
            voxels_np, state_vec_np = obs[0], obs[1]
        else:
            voxels_np = obs
            state_vec_np = np.zeros(args.dim_obs)

        # 检查状态向量维度并自动补齐 (Padding)
        # 如果环境返回 9 维，但模型要 10 维，我们在末尾补 0
        current_dim = state_vec_np.shape[0]
        target_dim = args.dim_obs
        
        if current_dim < target_dim:
            pad_size = target_dim - current_dim
            state_vec_np = np.pad(state_vec_np, (0, pad_size), 'constant')
        elif current_dim > target_dim:
            state_vec_np = state_vec_np[:target_dim]

        # 转 Tensor
        x_tensor = torch.as_tensor(voxels_np).float().to(device)
        v_tensor = torch.as_tensor(state_vec_np).float().to(device)

        if x_tensor.ndim == 3: x_tensor = x_tensor.unsqueeze(0)
        if v_tensor.ndim == 1: v_tensor = v_tensor.unsqueeze(0)

        # 推理
        with torch.no_grad():
            action, _, hx = policy(x_tensor, v_tensor, hx)
        
        # 截取动作 (前4维)
        action_np = action.cpu().numpy()
        if action_np.shape[0] == 1: action_np = action_np[0]
        
        env_action = action_np[:4] if action_np.shape[0] > 4 else action_np
            
        obs, reward, done, info = env.step(env_action)
        
        # 渲染
        if hasattr(env, 'render_img'):
            img = env.render_img
            if isinstance(img, torch.Tensor): img = img.cpu().numpy()
            if img.ndim == 3 and img.shape[0] == 3: img = np.transpose(img, (1, 2, 0))
            if img.max() <= 1.5: img = (img * 255).astype(np.uint8)
            else: img = img.astype(np.uint8)
            frames.append(img)
            
        if (i+1) % 50 == 0:
            print(f"Step: {i+1}/400")

    if args.save_visualization and len(frames) > 0:
        if not os.path.exists('gifs'): os.makedirs('gifs')
        save_path = 'gifs/result.gif'
        print(f"💾 GIF 保存成功: {save_path}")
        imageio.mimsave(save_path, frames, fps=30)
    else:
        print("⚠️ 未生成图像")

if __name__ == '__main__':
    main()
