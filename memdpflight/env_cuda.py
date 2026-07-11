"""
env_cuda.py  ——  修改版
在原始 Env 基础上新增：
  1. curriculum_phase  属性：0/1/2 三阶段课程
  2. phase_density     配置：各阶段障碍密度
  3. reset() 根据当前阶段动态调整障碍数量

其余物理仿真逻辑（run / render / find_vec）完全不动。
"""

import math
import random
import time
import torch
import torch.nn.functional as F
import quadsim_cuda


class GDecay(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.alpha, None

g_decay = GDecay.apply


class RunFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, R, dg, z_drag_coef, drag_2, pitch_ctl_delay,
                act_pred, act, p, v, v_wind, a, grad_decay, ctl_dt, airmode):
        act_next, p_next, v_next, a_next = quadsim_cuda.run_forward(
            R, dg, z_drag_coef, drag_2, pitch_ctl_delay,
            act_pred, act, p, v, v_wind, a, ctl_dt, airmode)
        ctx.save_for_backward(
            R, dg, z_drag_coef, drag_2, pitch_ctl_delay, v, v_wind, act_next)
        ctx.grad_decay = grad_decay
        ctx.ctl_dt     = ctl_dt
        return act_next, p_next, v_next, a_next

    @staticmethod
    def backward(ctx, d_act_next, d_p_next, d_v_next, d_a_next):
        R, dg, z_drag_coef, drag_2, pitch_ctl_delay, v, v_wind, act_next = \
            ctx.saved_tensors
        d_act_pred, d_act, d_p, d_v, d_a = quadsim_cuda.run_backward(
            R, dg, z_drag_coef, drag_2, pitch_ctl_delay, v, v_wind, act_next,
            d_act_next, d_p_next, d_v_next, d_a_next,
            ctx.grad_decay, ctx.ctl_dt)
        return (None, None, None, None, None,
                d_act_pred, d_act, d_p, d_v, None, d_a, None, None, None)

run = RunFunction.apply


class Env:
    # ── 课程阶段配置 ─────────────────────────────────────────
    # phase 0 easy  : 30% 障碍密度  ≈  9 个
    # phase 1 medium: 60% 障碍密度  ≈ 18 个
    # phase 2 hard  : 100% 障碍密度 = 30 个（原始值）
    PHASE_DENSITY = [0.30, 0.60, 1.00]
    N_OBS_MAX     = 30          # 原始最大障碍数，不改变

    def __init__(
        self,
        batch_size, width, height, grad_decay,
        device='cpu', fov_x_half_tan=0.53,
        single=False, gate=False, ground_voxels=False,
        scaffold=False, speed_mtp=1,
        random_rotation=False, cam_angle=10,
    ) -> None:
        self.device       = device
        self.batch_size   = batch_size
        self.width        = width
        self.height       = height
        self.grad_decay   = grad_decay
        best_ar = 0.0  # 初始化历史最高综合收益
        # ── 新增：课程阶段（由训练循环外部设置）─────────────
        self.curriculum_phase: int = 0

        # ── 原有权重/偏置张量（完全不动）──────────────────────
        self.ball_w   = torch.tensor([8., 18, 6, 0.2],    device=device)
        self.ball_b   = torch.tensor([0., -9, -1, 0.4],   device=device)
        self.voxel_w  = torch.tensor([8., 18, 6, 0.1, 0.1, 0.1], device=device)
        self.voxel_b  = torch.tensor([0., -9, -1, 0.2, 0.2, 0.2], device=device)
        self.ground_voxel_w = torch.tensor([8., 18, 0, 2.9, 2.9, 1.9], device=device)
        self.ground_voxel_b = torch.tensor([0., -9, -1, 0.1, 0.1, 0.1], device=device)
        self.cyl_w    = torch.tensor([8., 18, 0.35], device=device)
        self.cyl_b    = torch.tensor([0., -9, 0.05], device=device)
        self.cyl_h_w  = torch.tensor([8., 6, 0.1],  device=device)
        self.cyl_h_b  = torch.tensor([0., 0, 0.05], device=device)
        self.gate_w   = torch.tensor([2., 2, 1.0, 0.5], device=device)
        self.gate_b   = torch.tensor([3., -1, 0.0, 0.5], device=device)
        self.v_wind_w = torch.tensor([1, 1, 0.2], device=device)
        self.g_std    = torch.tensor([0., 0, -9.80665], device=device)
        self.roof_add = torch.tensor([0., 0., 2.5, 1.5, 1.5, 1.5], device=device)
        self.sub_div  = torch.linspace(0, 1./15, 10, device=device).reshape(-1, 1, 1)

        self.p_init = torch.as_tensor([
            [-1.5, -3.,  1], [ 9.5, -3.,  1],
            [-0.5,  1.,  1], [ 8.5,  1.,  1],
            [ 0.0,  3.,  1], [ 8.0,  3.,  1],
            [-1.0, -1.,  1], [ 9.0, -1.,  1],
        ], device=device).repeat(batch_size // 8 + 7, 1)[:batch_size]

        self.p_end = torch.as_tensor([
            [8.,  3.,  1], [0.,  3.,  1],
            [8., -1.,  1], [0., -1.,  1],
            [8., -3.,  1], [0., -3.,  1],
            [8.,  1.,  1], [0.,  1.,  1],
        ], device=device).repeat(batch_size // 8 + 7, 1)[:batch_size]

        self.flow          = torch.empty((batch_size, 0, height, width), device=device)
        self.single        = single
        self.gate          = gate
        self.ground_voxels = ground_voxels
        self.scaffold      = scaffold
        self.speed_mtp     = speed_mtp
        self.random_rotation = random_rotation
        self.cam_angle     = cam_angle
        self.fov_x_half_tan = fov_x_half_tan
        self.reset()

    # ── 新增辅助方法：根据当前课程阶段计算障碍数量 ───────────
    def _n_obs(self) -> int:
        """
        根据 curriculum_phase 返回当前实际障碍数量。
        最少保留 5 个，保证场景不完全空旷。
        """
        density = self.PHASE_DENSITY[self.curriculum_phase]
        return max(5, int(self.N_OBS_MAX * density))

    def reset(self):
        B      = self.batch_size
        device = self.device

        # ── 新增：动态障碍数量 ────────────────────────────────
        n_obs = self._n_obs()   # phase=0 → 9, phase=1 → 18, phase=2 → 30

        cam_angle = (self.cam_angle + torch.randn(B, device=device)) * math.pi / 180
        zeros = torch.zeros_like(cam_angle)
        ones  = torch.ones_like(cam_angle)
        self.R_cam = torch.stack([
            torch.cos(cam_angle), zeros, -torch.sin(cam_angle),
            zeros, ones, zeros,
            torch.sin(cam_angle), zeros, torch.cos(cam_angle),
        ], -1).reshape(B, 3, 3)

        # ── 障碍物初始化：用 n_obs 替换原来固定的 30 ──────────
        # 注意：只改障碍数量，其余采样逻辑完全不变
        self.balls  = torch.rand((B, n_obs, 4), device=device) * self.ball_w  + self.ball_b
        self.voxels = torch.rand((B, n_obs, 6), device=device) * self.voxel_w + self.voxel_b
        self.cyl    = torch.rand((B, n_obs, 3), device=device) * self.cyl_w   + self.cyl_b
        self.cyl_h  = torch.rand((B, 2,     3), device=device) * self.cyl_h_w + self.cyl_h_b

        self._fov_x_half_tan   = (0.95 + 0.1 * random.random()) * self.fov_x_half_tan
        self.n_drones_per_group = random.choice([4, 8])
        self.drone_radius       = random.uniform(0.1, 0.15)
        if self.single:
            self.n_drones_per_group = 1

        rd = torch.rand((B // self.n_drones_per_group, 1), device=device)\
               .repeat_interleave(self.n_drones_per_group, 0)
        self.max_speed = (0.75 + 2.5 * rd) * self.speed_mtp
        scale = (self.max_speed - 0.5).clamp_min(1)

        self.thr_est_error = 1 + torch.randn(B, device=device) * 0.01

        roof = torch.rand((B,)) < 0.5
        half = min(15, n_obs // 2)
        self.balls[~roof, :half, :2] = self.cyl[~roof, :half, :2]
        self.voxels[~roof, :half, :2] = self.cyl[~roof, half : half * 2, :2]
        self.balls[~roof, :half]  += self.roof_add[:4]
        self.voxels[~roof, :half] += self.roof_add

        # 以下 clamp 逻辑不变
        self.balls[..., 0]  = torch.minimum(
            torch.maximum(self.balls[..., 0],  self.balls[..., 3]  + 0.3 / scale),
            8 - 0.3 / scale - self.balls[..., 3])
        self.voxels[..., 0] = torch.minimum(
            torch.maximum(self.voxels[..., 0], self.voxels[..., 3] + 0.3 / scale),
            8 - 0.3 / scale - self.voxels[..., 3])
        self.cyl[..., 0]    = torch.minimum(
            torch.maximum(self.cyl[..., 0],    self.cyl[..., 2]    + 0.3 / scale),
            8 - 0.3 / scale - self.cyl[..., 2])
        self.cyl_h[..., 0]  = torch.minimum(
            torch.maximum(self.cyl_h[..., 0],  self.cyl_h[..., 2]  + 0.3 / scale),
            8 - 0.3 / scale - self.cyl_h[..., 2])

        self.voxels[roof, 0, 2]  = self.voxels[roof, 0, 2] * 0.5 + 201
        self.voxels[roof, 0, 3:] = 200

        if self.ground_voxels:
            ground_balls_r        = 8 + torch.rand((B, 2), device=device) * 6
            ground_balls_r_ground = 2 + torch.rand((B, 2), device=device) * 4
            ground_balls_h        = ground_balls_r - (
                ground_balls_r.pow(2) - ground_balls_r_ground.pow(2)).sqrt()
            self.balls[:, :2, 3] = ground_balls_r
            self.balls[:, :2, 2] = ground_balls_h - ground_balls_r - 1
            ground_voxels = torch.rand((B, 10, 6), device=device) \
                            * self.ground_voxel_w + self.ground_voxel_b
            ground_voxels[:, :, 2] = ground_voxels[:, :, 5] - 1
            self.voxels = torch.cat([self.voxels, ground_voxels], 1)

        self.voxels[:, :, 1] *= (self.max_speed + 4) / scale
        self.balls[:, :, 1]  *= (self.max_speed + 4) / scale
        self.cyl[:, :, 1]    *= (self.max_speed + 4) / scale

        if self.gate:
            gate = torch.rand((B, 4), device=device) * self.gate_w + self.gate_b
            p    = gate[None, :, :3]
            nearest_pt = torch.empty_like(p)
            quadsim_cuda.find_nearest_pt(
                nearest_pt, self.balls, self.cyl, self.cyl_h, self.voxels,
                p, self.drone_radius, 1)
            gate_x, gate_y, gate_z, gate_r = gate.unbind(-1)
            gate_x[(nearest_pt - p).norm(2, -1)[0] < 0.5] = -50
            ones_ = torch.ones_like(gate_x)
            gate  = torch.stack([
                torch.stack([gate_x, gate_y + gate_r + 5, gate_z, ones_*0.05, ones_*5, ones_*5], -1),
                torch.stack([gate_x, gate_y,              gate_z + gate_r + 5, ones_*0.05, ones_*5, ones_*5], -1),
                torch.stack([gate_x, gate_y - gate_r - 5, gate_z, ones_*0.05, ones_*5, ones_*5], -1),
                torch.stack([gate_x, gate_y,              gate_z - gate_r - 5, ones_*0.05, ones_*5, ones_*5], -1),
            ], 1)
            self.voxels = torch.cat([self.voxels, gate], 1)

        self.voxels[..., 0] *= scale
        self.balls[..., 0]  *= scale
        self.cyl[..., 0]    *= scale
        self.cyl_h[..., 0]  *= scale

        if self.ground_voxels:
            self.balls[:, :2, 0] = torch.minimum(
                torch.maximum(self.balls[:, :2, 0], ground_balls_r_ground + 0.3),
                scale * 8 - 0.3 - ground_balls_r_ground)

        self.pitch_ctl_delay = 12 + 1.2 * torch.randn((B, 1), device=device)
        self.yaw_ctl_delay   = 6  + 0.6 * torch.randn((B, 1), device=device)

        rd = torch.rand((B // self.n_drones_per_group, 1), device=device)\
               .repeat_interleave(self.n_drones_per_group, 0)
        scale = torch.cat([
            scale,
            rd + 0.5,
            torch.rand_like(scale) - 0.5,
        ], -1)
        self.p        = self.p_init  * scale + torch.randn_like(scale) * 0.1
        self.p_target = self.p_end   * scale + torch.randn_like(scale) * 0.1

        if self.random_rotation:
            yaw_bias = torch.rand(B // self.n_drones_per_group, device=device)\
                         .repeat_interleave(self.n_drones_per_group, 0) * 1.5 - 0.75
            c = torch.cos(yaw_bias);  s = torch.sin(yaw_bias)
            l = torch.ones_like(yaw_bias);  o = torch.zeros_like(yaw_bias)
            R = torch.stack([c,-s,o, s,c,o, o,o,l], -1).reshape(B, 3, 3)
            self.p        = torch.squeeze(R @ self.p[..., None], -1)
            self.p_target = torch.squeeze(R @ self.p_target[..., None], -1)
            self.voxels[..., :3] = (R @ self.voxels[..., :3].transpose(1,2)).transpose(1,2)
            self.balls[...,  :3] = (R @ self.balls[...,  :3].transpose(1,2)).transpose(1,2)
            self.cyl[...,    :3] = (R @ self.cyl[...,    :3].transpose(1,2)).transpose(1,2)

        if self.scaffold and random.random() < 0.5:
            x   = torch.arange(1, 6, dtype=torch.float, device=device)
            y   = torch.arange(-3, 4, dtype=torch.float, device=device)
            z   = torch.arange(1, 4, dtype=torch.float, device=device)
            _x, _y = torch.meshgrid(x, y)
            scaf_v = torch.stack([_x, _y, torch.full_like(_x, 0.02)], -1).flatten(0,1)
            x_bias = torch.rand_like(self.max_speed) * self.max_speed
            scale2 = 1 + torch.rand((B, 1, 1), device=device)
            scaf_v = scaf_v * scale2 + torch.stack([
                x_bias, torch.randn_like(self.max_speed),
                torch.rand_like(self.max_speed) * 0.01], -1)
            self.cyl = torch.cat([self.cyl, scaf_v], 1)
            _x2, _z2 = torch.meshgrid(x, z)
            scaf_h = torch.stack([_x2, _z2, torch.full_like(_x2, 0.02)], -1).flatten(0,1)
            scaf_h = scaf_h * scale2 + torch.stack([
                x_bias, torch.randn_like(self.max_speed) * 0.1,
                torch.rand_like(self.max_speed) * 0.01], -1)
            self.cyl_h = torch.cat([self.cyl_h, scaf_h], 1)

        self.v      = torch.randn((B, 3), device=device) * 0.2
        self.v_wind = torch.randn((B, 3), device=device) * self.v_wind_w
        self.act    = torch.randn_like(self.v) * 0.1
        self.a      = self.act
        self.dg     = torch.randn((B, 3), device=device) * 0.2

        R   = torch.zeros((B, 3, 3), device=device)
        self.R = quadsim_cuda.update_state_vec(
            R, self.act,
            torch.randn((B, 3), device=device) * 0.2 + F.normalize(self.p_target - self.p),
            torch.zeros_like(self.yaw_ctl_delay), 5)
        self.R_old = self.R.clone()
        self.p_old = self.p
        self.margin = torch.rand((B,), device=device) * 0.2 + 0.1

        self.drag_2      = torch.rand((B, 2), device=device) * 0.15 + 0.3
        self.drag_2[:, 0] = 0
        self.z_drag_coef = torch.ones((B, 1), device=device)

    # ── 以下方法完全不变 ──────────────────────────────────────

    @staticmethod
    @torch.no_grad()
    def update_state_vec(R, a_thr, v_pred, alpha, yaw_inertia=5):
        self_forward_vec = R[..., 0]
        g_std   = torch.tensor([0, 0, -9.80665], device=R.device)
        a_thr   = a_thr - g_std
        thrust  = torch.norm(a_thr, 2, -1, True)
        self_up_vec = a_thr / thrust
        forward_vec = self_forward_vec * yaw_inertia + v_pred
        forward_vec = self_forward_vec * alpha + F.normalize(forward_vec, 2, -1) * (1 - alpha)
        forward_vec[:, 2] = (forward_vec[:, 0] * self_up_vec[:, 0]
                             + forward_vec[:, 1] * self_up_vec[:, 1]) / -self_up_vec[2]
        self_forward_vec = F.normalize(forward_vec, 2, -1)
        self_left_vec    = torch.cross(self_up_vec, self_forward_vec)
        return torch.stack([self_forward_vec, self_left_vec, self_up_vec], -1)

    def render(self, ctl_dt):
        canvas = torch.empty((self.batch_size, self.height, self.width), device=self.device)
        quadsim_cuda.render(
            canvas, self.flow, self.balls, self.cyl, self.cyl_h,
            self.voxels, self.R @ self.R_cam, self.R_old, self.p,
            self.p_old, self.drone_radius, self.n_drones_per_group,
            self._fov_x_half_tan)
        return canvas, None

    def find_vec_to_nearest_pt(self):
        p = self.p + self.v * self.sub_div
        nearest_pt = torch.empty_like(p)
        quadsim_cuda.find_nearest_pt(
            nearest_pt, self.balls, self.cyl, self.cyl_h, self.voxels,
            p, self.drone_radius, self.n_drones_per_group)
        return nearest_pt - p

    def run(self, act_pred, ctl_dt=1/15, v_pred=None):
        self.dg = (self.dg * math.sqrt(1 - ctl_dt / 4)
                   + torch.randn_like(self.dg) * 0.2 * math.sqrt(ctl_dt / 4))
        self.p_old = self.p
        self.act, self.p, self.v, self.a = run(
            self.R, self.dg, self.z_drag_coef, self.drag_2, self.pitch_ctl_delay,
            act_pred, self.act, self.p, self.v, self.v_wind, self.a,
            self.grad_decay, ctl_dt, 0.5)
        alpha      = torch.exp(-self.yaw_ctl_delay * ctl_dt)
        self.R_old = self.R.clone()
        self.R     = quadsim_cuda.update_state_vec(self.R, self.act, v_pred, alpha, 5)