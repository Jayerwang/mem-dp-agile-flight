# Next Code Change Plan: Progress Loss and Risk-Aware Memory

## 1. Current Finding

The latest unified original-like CUDA evaluation shows:

- The original model has relatively high timeout at low speed and increasing crash at high speed.
- The memory-enhanced model after full-density phase2 safety fine-tuning improves overall success rate.
- The strongest observed gain of the improved model is lower timeout, especially in medium/high-speed settings.
- Crash is not uniformly lower, which means safety margin is still the main weakness.

Working interpretation:

```text
Explicit memory helps the policy keep a more continuous local scene/goal representation, so it reduces hesitation and timeout.
However, reconstruction alone does not directly teach collision risk, so high-speed crash remains.
```

## 2. Design Principle

Prefer small, conservative changes:

1. Do not modify CUDA kernels first.
2. Do not remove the existing `recon_decoder`.
3. Add new losses and heads as optional arguments.
4. Keep the original training path reproducible when all new coefficients are zero.
5. Verify each change with the same unified evaluation protocol.

## 3. Priority Order

Recommended implementation order:

```text
Step 1: Add progress / goal-reaching loss.
Step 2: Add optional risk_head.
Step 3: Add explicit speed-bin sampling.
Step 4: Only if necessary, split scene scale from target speed in env reset.
```

The safest first experiment is Step 1 only.

## 4. Step 1: Add Progress and Goal Loss

### Problem

Low-speed failures are mostly timeout, not crash. The current training success metric mainly checks whether the rollout avoids collision, while evaluation success requires reaching the goal.

### Code Target

Modify:

```text
memdpflight/mem_main_cuda.py
```

### New Arguments

Add:

```python
parser.add_argument('--coef_progress', type=float, default=0.0)
parser.add_argument('--coef_goal', type=float, default=0.0)
parser.add_argument('--progress_margin', type=float, default=0.02)
```

Default `0.0` keeps existing behavior unchanged.

### Loss Definition

During rollout, store distance-to-goal:

```python
goal_dist_history.append(torch.norm(env.p_target - env.p, 2, -1))
```

After rollout:

```python
goal_dist_history = torch.stack(goal_dist_history)  # (T, B)
progress = goal_dist_history[:-1] - goal_dist_history[1:]
loss_progress = F.relu(args.progress_margin - progress).mean()
loss_goal = F.smooth_l1_loss(goal_dist_history[-1], torch.zeros_like(goal_dist_history[-1]))
```

Add to total loss:

```python
loss = loss + args.coef_progress * loss_progress + args.coef_goal * loss_goal
```

### First Training Command

Use small weights:

```bash
python mem_main_cuda.py \
  --resume checkpoint_phase1_done.pth \
  --single \
  --speed_mtp 2.5 \
  --batch_size 64 \
  --timesteps 150 \
  --num_iters 10000 \
  --lr 3e-4 \
  --ground_voxels \
  --random_rotation \
  --yaw_drift \
  --cam_angle 20 \
  --fov_x_half_tan 0.82 \
  --coef_collide 10.0 \
  --coef_obj_avoidance 4.0 \
  --coef_progress 0.05 \
  --coef_goal 0.01 \
  --progress_margin 0.02 \
  --coef_d_acc 0.01 \
  --coef_d_jerk 0.001 \
  --recon_members 1 1 0 \
  --recon_steps 12 \
  --recon_coef_past 0.03 \
  --recon_coef_cur 0.02 \
  --phase0_iters 0 \
  --phase1_iters 0 \
  --force_full_density
```

### Expected Effect

Expected improvement:

- lower timeout at 2/4 m/s;
- no obvious crash increase at 8/9/10 m/s.

Failure signal:

- crash rises sharply, meaning progress loss is pushing the model to rush through obstacles.

If crash rises, reduce:

```text
coef_progress -> 0.02
coef_goal -> 0.005
```

## 5. Step 2: Add Risk-Aware Auxiliary Head

### Problem

The current `recon_decoder` reconstructs compressed visual/depth features. This helps memory keep local geometry, which may explain lower timeout, but it does not directly supervise safety risk.

### Conservative Choice

Do not replace `recon_decoder`.

Add:

```text
risk_head
```

in parallel:

```text
GRU hidden
  -> action head
  -> recon_decoder
  -> risk_head
```

### Code Target

Modify:

```text
memdpflight/model.py
memdpflight/mem_main_cuda.py
```

### Model Change

In `Model.__init__`, add optional head:

```python
self.enable_risk_head = enable_risk_head
if enable_risk_head:
    self.risk_head = nn.Sequential(
        nn.Linear(192, 64),
        nn.LeakyReLU(0.05),
        nn.Linear(64, 1),
    )
```

Add method:

```python
def predict_risk(self, hx: torch.Tensor) -> torch.Tensor:
    return self.risk_head(hx)
```

### Training Arguments

Add:

```python
parser.add_argument('--enable_risk_head', default=False, action='store_true')
parser.add_argument('--coef_risk', type=float, default=0.0)
```

### Supervision Signal

Use nearest obstacle clearance already computed in training:

```python
distance = torch.norm(vec_to_pt_history, 2, -1) - env.margin
min_gap = distance.min(dim=2).values
```

Target can be clipped for stable regression:

```python
risk_target = min_gap.clamp(-1.0, 3.0)
loss_risk = F.smooth_l1_loss(risk_pred.squeeze(-1), risk_target)
```

Alternative binary target:

```text
future_collision_label = any gap < 0 in next K steps
loss_risk = BCEWithLogitsLoss(risk_logit, future_collision_label)
```

Start with min-gap regression because it is simpler.

### Training Schedule

Recommended:

```text
phase1: keep current recon training.
phase2: enable risk loss with low weight.
```

This preserves the current timeout advantage from memory reconstruction.

First trial:

```text
coef_risk = 0.05
```

## 6. Step 3: Explicit Low/High-Speed Training

### Meaning

Currently speed is implicit:

```python
self.max_speed = (0.75 + 2.5 * rd) * self.speed_mtp
```

This randomly generates a speed range, but it does not guarantee balanced training at the exact test speeds:

```text
2, 4, 6, 7, 8, 9, 10 m/s
```

Explicit speed-bin training means sampling target speeds directly from fixed bins.

### Minimal Code Change

In `mem_main_cuda.py`, after:

```python
env.reset()
```

and before target velocity is calculated, optionally override:

```python
speed_bins = torch.tensor(args.speed_bins, device=device)
sampled = speed_bins[torch.randint(0, len(speed_bins), (B,), device=device)]
env.max_speed = sampled
```

### New Arguments

```python
parser.add_argument('--speed_bins', type=str, default='')
```

Parser helper:

```python
speed_bins = [float(x) for x in args.speed_bins.split(',') if x]
```

### Recommended First Bins

```text
2,4,6,8,10
```

This directly targets:

- low-speed timeout;
- high-speed crash;
- speed generalization.

### Caveat

This minimal change changes target speed but not scene scale. A cleaner future version should split:

```text
scene_mtp
target_speed
```

inside `env_cuda.py`.

## 7. Step 4: Split Scene Scale and Target Speed

This is more invasive and should not be the first step.

Current `speed_mtp` affects both:

- scene generation;
- target velocity scale.

Better design:

```text
scene_mtp: controls target distance and obstacle layout scale
target_speed: controls commanded flight speed
```

This would make low-speed testing more meaningful because `target_speed=2` would not be forced to fly a scene generated for high speed.

Recommended only after Step 1-3 are tested.

## 8. Evaluation Plan After Each Change

Use the same unified evaluator.

Main protocol:

```bash
python memdpflight/experiments/round1_unified_eval.py \
  --repo-root /root/autodl-tmp/.autodl/DiffPhysDrone-master/DiffPhysDrone-master \
  --no-default-models \
  --model original_checkpoint0004:original:/root/autodl-tmp/.autodl/DiffPhysDrone-master/DiffPhysDrone-master/backup_checkpoints/20260413_131854/checkpoint0004.pth \
  --model ft_best:memdp:memdpflight_ft_phase2_safe/checkpoint_best.pth \
  --model new_best:memdp:<new_training_dir>/checkpoint_best.pth \
  --speeds 2,4,6,7,8,9,10 \
  --speed-mtp 2.5 \
  --num-episodes 15 \
  --batch-size 32 \
  --env-kind original
```

Report:

- success rate;
- crash rate;
- timeout rate;
- average over speeds;
- low-speed subset average: 2/4 m/s;
- high-speed subset average: 8/9/10 m/s.

## 9. Paper-Oriented Hypotheses

### Hypothesis A: Memory Reduces Timeout

Memory reconstruction helps the policy maintain a short-term local geometry / direction representation, reducing hesitation and unnecessary detours.

Expected evidence:

```text
memflight timeout < original timeout
```

especially at medium/high speeds.

### Hypothesis B: Progress Loss Reduces Low-Speed Timeout

Low-speed timeout comes from weak goal-reaching pressure under long-distance scenes.

Expected evidence:

```text
2/4 m/s timeout decreases
success increases without crash increasing too much
```

### Hypothesis C: Risk Head Reduces High-Speed Crash

Depth reconstruction is not directly risk-aware. Predicting clearance or collision risk should improve high-speed safety margin.

Expected evidence:

```text
8/9/10 m/s crash decreases
```

### Hypothesis D: Explicit Speed Bins Improve Speed Generalization

Training directly on evaluation speed bins reduces sensitivity to `speed_mtp`.

Expected evidence:

```text
performance gap between speed_mtp=2.5 and speed_mtp=4.0 narrows
```

## 10. Recommended Immediate Next Step

Implement only Step 1 first:

```text
Add progress loss + goal loss.
Keep recon unchanged.
Keep risk_head disabled.
Keep speed-bin sampling disabled.
```

Why:

- smallest code change;
- directly targets timeout;
- low risk of destroying current memory advantage;
- easy to write as an ablation.

If Step 1 improves timeout without raising crash, keep it.

Then implement Step 2 risk head to target high-speed crash.

