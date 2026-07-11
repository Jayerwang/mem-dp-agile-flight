# ============================================================
# training_modules.py
#
# 使用方式：将本文件放在 main_cuda.py 同目录，然后：
#   from training_modules import CurriculumScheduler, \
#       set_training_phase, build_optimizer
# 或直接将内容粘贴到 main_cuda.py import 块之后。
#
# 对照来源：
#   CurriculumScheduler <- MAVRL ppo_recurrent.py collect_rollouts()
#                          三档难度 + eval() easy/medium/hard radius
#   set_training_phase  <- MAVRL only_lstm_training 标志
#                          + RecurrentPPO._setup_model() 分阶段冻结
#   build_optimizer     <- MAVRL 各阶段重建 optimizer 策略
# ============================================================

from collections import deque
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


# ──────────────────────────────────────────────────────────────
# 模块一：CurriculumScheduler
# ──────────────────────────────────────────────────────────────

class CurriculumScheduler:
    """
    渐进式课程调度器。
    根据最近 window 次 iter 的 success_rate 滑动平均自动升级难度阶段。

    三个阶段（对照 MAVRL easy/medium/hard 三档）：
        phase 0 → easy   : 障碍密度 30%，损失系数鼓励速度
        phase 1 → medium : 障碍密度 60%，损失系数均衡
        phase 2 → hard   : 障碍密度 100%，损失系数重点惩罚碰撞

    典型用法：
        curriculum = CurriculumScheduler()
        for i in range(num_iters):
            ...训练...
            phase = curriculum.step(float(_success), current_iter=i)
            env.curriculum_phase = phase
            w_v, w_c = curriculum.get_loss_coefs(args.coef_v, args.coef_collide)
    """

    def __init__(
        self,
        phase_thresholds=(0.70, 0.82),
        # 触发升级所需的 success_rate 滑动均值（两个阈值对应两次升级）
        window=50,
        # 滑动平均窗口（单位：iter）
        min_phase_iters=500,
        # 每阶段至少经过多少 iter 才允许升级，防止早期噪声触发
        allow_downgrade=False,
        # 是否允许 success_rate 过低时自动降级
        downgrade_threshold=0.40,
        # 触发降级的 success_rate 下限
    ):
        assert len(phase_thresholds) == 2, \
            "固定三阶段设计，需要恰好两个阈值"
        self.phase_thresholds    = phase_thresholds
        self.window              = window
        self.min_phase_iters     = min_phase_iters
        self.allow_downgrade     = allow_downgrade
        self.downgrade_threshold = downgrade_threshold

        self.phase            = 0
        self.history          = deque(maxlen=window)
        self.phase_iter_count = 0      # 当前阶段已过的 iter 数
        self.phase_log        = []     # [(iter, new_phase, 'upgrade'/'downgrade')]

    # ----------------------------------------------------------
    def step(self, success_rate: float, current_iter: int = -1) -> int:
        """
        每个 iter 结束时调用一次，返回当前 phase（0/1/2）。

        Args:
            success_rate : 本 iter 成功率（0.0 ~ 1.0）
            current_iter : 当前 iter 号（仅用于日志）

        Returns:
            int: 当前 phase
        """
        self.history.append(float(success_rate))
        self.phase_iter_count += 1

        # 窗口未满时不做判断，避免早期噪声触发
        if len(self.history) < self.window:
            return self.phase

        avg = sum(self.history) / len(self.history)

        # ── 升级 ──
        if (
            self.phase < 2
            and avg >= self.phase_thresholds[self.phase]
            and self.phase_iter_count >= self.min_phase_iters
        ):
            self.phase += 1
            self.phase_iter_count = 0
            self.history.clear()        # 清空历史，让新阶段重新积累
            self.phase_log.append((current_iter, self.phase, "upgrade"))
            _label = {1: "medium", 2: "hard"}[self.phase]
            print(
                f"\n[Curriculum] iter={current_iter}  "
                f"avg_success={avg:.3f}  "
                f"→ 升级到 phase {self.phase} ({_label})\n"
            )

        # ── 降级（可选） ──
        elif (
            self.allow_downgrade
            and self.phase > 0
            and avg < self.downgrade_threshold
            and self.phase_iter_count >= self.min_phase_iters
        ):
            self.phase -= 1
            self.phase_iter_count = 0
            self.history.clear()
            self.phase_log.append((current_iter, self.phase, "downgrade"))
            print(
                f"\n[Curriculum] iter={current_iter}  "
                f"avg_success={avg:.3f}  "
                f"→ 降级到 phase {self.phase}\n"
            )

        return self.phase

    # ----------------------------------------------------------
    def get_loss_coefs(
        self, base_coef_v: float, base_coef_collide: float
    ):
        """
        根据当前 phase 返回调整后的损失权重。

        对照 MAVRL：密集障碍时 collision 权重更高，speed 权重更低。

        Args:
            base_coef_v       : args.coef_v（命令行基础值）
            base_coef_collide : args.coef_collide

        Returns:
            (w_v, w_collide): 调整后的权重
        """
        # (w_v 倍率, w_collide 倍率)，按 phase 0/1/2
        multipliers = [
            (1.5, 0.7),   # phase 0 easy:   鼓励速度
            (1.0, 1.0),   # phase 1 medium: 均衡
            (0.6, 1.8),   # phase 2 hard:   重点惩罚碰撞
        ]
        m_v, m_c = multipliers[self.phase]
        return base_coef_v * m_v, base_coef_collide * m_c

    # ----------------------------------------------------------
    def state_dict(self) -> dict:
        """保存状态，与 torch.save 配合使用"""
        return {
            "phase":            self.phase,
            "history":          list(self.history),
            "phase_iter_count": self.phase_iter_count,
            "phase_log":        self.phase_log,
        }

    def load_state_dict(self, d: dict):
        """从 checkpoint 恢复"""
        self.phase            = d["phase"]
        self.history          = deque(d["history"], maxlen=self.window)
        self.phase_iter_count = d["phase_iter_count"]
        self.phase_log        = d.get("phase_log", [])

    def __repr__(self):
        avg = sum(self.history) / len(self.history) if self.history else 0.0
        return (
            f"CurriculumScheduler("
            f"phase={self.phase}, "
            f"avg={avg:.3f}, "
            f"phase_iter={self.phase_iter_count}/{self.min_phase_iters})"
        )


# ──────────────────────────────────────────────────────────────
# 模块二：set_training_phase
# ──────────────────────────────────────────────────────────────

def set_training_phase(model, phase: int, verbose: bool = True) -> dict:
    """
    按阶段冻结/解冻模型各子模块的参数。

    对照 MAVRL only_lstm_training 的三阶段拆分：

        Phase 0（基础预热）
            可训练: stem / v_proj / gru / fc
            冻结:   recon_decoder / v_scale_head
            目的:   主任务先收敛，防止新模块梯度早期干扰策略

        Phase 1（记忆预热）
            可训练: stem / v_proj / gru / recon_decoder / v_scale_head
            冻结:   fc（动作头）
            目的:   对照 MAVRL only_lstm_training=True 冻结 action_net，
                    让 GRU 学会压缩历史，重建梯度不改变动作分布

        Phase 2（联合训练）
            可训练: 全部参数
            目的:   对照 MAVRL retrain PPO 阶段，端到端优化

    Args:
        model   : Model 实例（来自 model.py）
        phase   : 0 / 1 / 2
        verbose : 是否打印冻结状态

    Returns:
        dict: {模块名: requires_grad}，可用于断言和日志
    """
    assert phase in (0, 1, 2), f"phase 必须是 0/1/2，收到 {phase}"

    # ── 子模块分组 ──

    # 原始核心模块：始终可训练
    core_modules = {
        "stem":   model.stem,
        "v_proj": model.v_proj,
        "gru":    model.gru,
    }

    # 动作输出头：phase 1 时冻结
    action_modules = {
        "fc": model.fc,
    }

    # 新增辅助模块：phase 0 全冻结，phase 1/2 开启
    aux_modules = {}
    if hasattr(model, "recon_decoder"):
        aux_modules["recon_decoder"] = model.recon_decoder
    if hasattr(model, "v_scale_head"):
        aux_modules["v_scale_head"] = model.v_scale_head

    # ── 设置 requires_grad ──
    status = {}

    for name, mod in core_modules.items():
        grad = True
        _set_module_grad(mod, grad)
        status[name] = grad

    for name, mod in action_modules.items():
        grad = (phase != 1)             # phase 1 时冻结
        _set_module_grad(mod, grad)
        status[name] = grad

    for name, mod in aux_modules.items():
        grad = (phase >= 1)             # phase 0 时冻结
        _set_module_grad(mod, grad)
        status[name] = grad

    # ── 打印状态 ──
    if verbose:
        phase_names = {
            0: "easy   (基础预热)",
            1: "medium (记忆预热)",
            2: "hard   (联合训练)",
        }
        print(f"\n[set_training_phase] → phase {phase} {phase_names[phase]}")
        for name, grad in status.items():
            mark = "✓ 可训练" if grad else "✗ 冻结  "
            print(f"  {name:<20} {mark}")
        all_mods = {**core_modules, **action_modules, **aux_modules}
        n_trainable = sum(
            p.numel()
            for mod_name, mod in all_mods.items()
            for p in mod.parameters()
            if status[mod_name]
        )
        print(f"  可训练参数总量: {n_trainable:,}\n")

    return status


def _set_module_grad(module, requires_grad: bool):
    """将模块所有参数的 requires_grad 统一设置"""
    for p in module.parameters():
        p.requires_grad = requires_grad


# ──────────────────────────────────────────────────────────────
# 模块三：build_optimizer
# ──────────────────────────────────────────────────────────────

def build_optimizer(model, lr: float, phase: int):
    """
    只将 requires_grad=True 的参数加入 optimizer。

    每次 set_training_phase 之后必须重建 optimizer，否则被冻结的参数
    仍在原 optimizer 的 param_group 中，梯度更新会出错。

    Args:
        model : Model 实例（已调用过 set_training_phase）
        lr    : 基础学习率（args.lr）
        phase : 当前阶段，用于自动调整 lr

    Returns:
        AdamW optimizer（只含当前可训练参数）
    """
    # phase 1 记忆预热时用更小 lr，保护已有 GRU 权重
    # 对照 MAVRL train_lstm_from_dataset 的保守 lr 策略
    lr_multipliers = {0: 1.0, 1: 0.3, 2: 1.0}
    actual_lr = lr * lr_multipliers[phase]

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in trainable_params)

    optimizer = AdamW(trainable_params, lr=actual_lr)
    print(
        f"[build_optimizer] phase={phase}  "
        f"可训练参数={n_params:,}  "
        f"lr={actual_lr:.2e}"
    )
    return optimizer


# ──────────────────────────────────────────────────────────────
# main_cuda.py 集成模板（注释形式，对照位置插入）
# ──────────────────────────────────────────────────────────────

"""
# ── ① import 区域顶部 ─────────────────────────────────────────

from training_modules import (
    CurriculumScheduler,
    set_training_phase,
    build_optimizer,
)
from collections import deque

# ── ② argparse 解析之后，Env/Model 实例化之前 ────────────────

# 阶段切换节点（可按比例或绝对 iter 数调整）
PHASE_1_START = int(args.num_iters * 0.20)   # 20% 处开始记忆预热
PHASE_2_START = int(args.num_iters * 0.50)   # 50% 处开始联合训练

# 记忆参数（对照 MAVRL reconstruction_steps=10 @10Hz → 12 @15Hz）
RECON_STEPS      = 12
RECON_COEF_PAST  = 0.30
RECON_COEF_CUR   = 0.20

# ── ③ Model 实例化之后立即调用 ──────────────────────────────

model = Model(7, 6)                              # 根据 no_odom 参数调整
model = model.to(device)

set_training_phase(model, phase=0)               # 初始冻结新模块
optim = build_optimizer(model, args.lr, phase=0) # 只含可训练参数
sched = CosineAnnealingLR(optim, args.num_iters, args.lr * 0.01)

curriculum = CurriculumScheduler(
    phase_thresholds=(0.70, 0.82),
    window=50,
    min_phase_iters=max(200, args.num_iters // 50),
    allow_downgrade=False,
)

# 深度历史缓冲（记忆重建用，每 episode 清空）
depth_history = deque(maxlen=RECON_STEPS + 5)

# ── ④ 训练主循环内，for i in pbar: 的最开始 ─────────────────

    # 阶段切换检查（在 env.reset() 之前）
    if i == PHASE_1_START:
        set_training_phase(model, phase=1)
        optim = build_optimizer(model, args.lr, phase=1)
        sched = CosineAnnealingLR(optim, args.num_iters - i, args.lr * 0.01)

    elif i == PHASE_2_START:
        set_training_phase(model, phase=2)
        optim = build_optimizer(model, args.lr, phase=2)
        sched = CosineAnnealingLR(optim, args.num_iters - i, args.lr * 0.01)

    env.reset()
    model.reset()

    # 每 episode 的重建 loss 收集
    recon_past_losses = []
    recon_cur_losses  = []
    depth_history.clear()   # 不跨 episode 保留历史

# ── ⑤ 内层 t 步循环内，depth, flow = env.render() 之后 ──────

        depth, flow = env.render(ctl_dt)

        depth_norm = (3.0 / depth.clamp_(0.3, 24) - 0.6).detach()
        depth_c    = F.avg_pool2d(depth_norm[:, None], 4, 4).squeeze(1)
        depth_history.append(depth_c)

        act, _, h = model(x, state, h)   # 根据实际 forward 签名调整

        # 记忆重建（phase >= 1 时计算）
        if i >= PHASE_1_START and hasattr(model, "recon_decoder"):
            if len(depth_history) == depth_history.maxlen:
                past_target = depth_history[0].flatten(1).detach()
                recon_past  = model.decode(h)
                recon_past_losses.append(F.mse_loss(recon_past, past_target))

            cur_target = depth_c.flatten(1).detach()
            recon_cur  = model.decode(h)
            recon_cur_losses.append(F.mse_loss(recon_cur, cur_target))

# ── ⑥ 内层循环结束后，loss 计算区域 ─────────────────────────

    # 动态损失系数
    w_v, w_collide = curriculum.get_loss_coefs(args.coef_v, args.coef_collide)

    loss = w_v       * loss_v       + \\
           w_collide * loss_collide + \\
           args.coef_obj_avoidance * loss_obj_avoidance + \\
           args.coef_d_acc  * loss_d_acc  + \\
           args.coef_d_jerk * loss_d_jerk + \\
           args.coef_v_pred * loss_v_pred

    # 记忆重建损失
    loss_recon = 0.0
    if i >= PHASE_1_START:
        if recon_past_losses:
            loss_recon += RECON_COEF_PAST * torch.stack(recon_past_losses).mean()
        if recon_cur_losses:
            loss_recon += RECON_COEF_CUR  * torch.stack(recon_cur_losses).mean()
        loss = loss + loss_recon

    optim.zero_grad()
    loss.backward()
    optim.step()
    sched.step()

# ── ⑦ with torch.no_grad() 块内，smooth_dict 之前 ────────────

        phase = curriculum.step(float(_success), current_iter=i)
        env.curriculum_phase = phase

        smooth_dict({
            # ... 原有指标 ...
            "curriculum/phase":     float(phase),
            "curriculum/w_v":       w_v,
            "curriculum/w_collide": w_collide,
            "loss/recon":           float(loss_recon),
        })

# ── ⑧ Checkpoint 保存，加入 curriculum 状态 ──────────────────

    if (i + 1) % 10000 == 0:
        torch.save({
            "model":      model.state_dict(),
            "curriculum": curriculum.state_dict(),
            "phase":      phase,
            "iter":       i,
        }, f"checkpoint{i//10000:04d}.pth")

    # 从 checkpoint 恢复时：
    # ckpt = torch.load("checkpointXXXX.pth")
    # model.load_state_dict(ckpt["model"])
    # curriculum.load_state_dict(ckpt["curriculum"])
    # set_training_phase(model, phase=ckpt["phase"])
    # optim = build_optimizer(model, args.lr, phase=ckpt["phase"])
"""