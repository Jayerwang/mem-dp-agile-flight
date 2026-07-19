# Round 1 Unified Evaluation Plan

## Goal

This round tests whether the memflight checkpoints really improve over the original model under the same lightweight CUDA original-like protocol.

The main risk in previous comparisons is that model differences were mixed with script differences:

- different evaluation entry points;
- different action-buffer delay logic;
- different `speed_mtp`;
- different camera setting;
- different obstacle density in `memdpflight/env_cuda.py`;
- different number of episodes.

Round 1 removes those differences as much as possible.

## Main Question

Does the memory-augmented memflight model outperform the original DiffPhysDrone model when both are evaluated in the same CUDA simulator, with the same camera, speed list, obstacle setting, and success definition?

## Models

Evaluate the following models:

| Model name | Type | Purpose |
| --- | --- | --- |
| `original_checkpoint0004` | original | Original DiffPhysDrone baseline. |
| `v3gru_no_pre` | memdp | Early explicit-GRU + reconstruction model without separate decoder pretraining. |
| `phase0_done` | memdp | End of phase 0, before full memory-reconstruction training. |
| `checkpoint_best` | memdp | Best training-time AR checkpoint, where `AR = success_rate * avg_speed`. |
| `phase1_done` | memdp | End of memory/reconstruction phase; current representative improved checkpoint. |
| `checkpoint00005` | memdp | Periodic checkpoint around 25000 iterations, near phase 1 end. |

## Default Protocol

Use the same scene protocol for every model:

```text
env_kind = original
target_speed = 2, 4, 6, 7, 8, 9, 10
speed_mtp = 2.5
batch_size = 32
num_episodes = 15
timesteps = 400
goal_radius = 5.0
ground_voxels = true
random_rotation = true
cam_angle = 20
fov_x_half_tan = 0.82
action_mode = original_lag
```

Why `env_kind = original`:

- root `env_cuda.py` always uses 30 obstacles;
- it avoids accidentally testing memdp checkpoints in a lower-density curriculum environment;
- both original and memdp models receive the same observation/state format.

Why `speed_mtp = 2.5`:

- `eval_original_like.py` uses `speed_mtp = 2.5` by default;
- this keeps the first comparison closer to the project's original-like evaluation script.

Also run `speed_mtp = 4.0` as a follow-up sensitivity test, because both the original single-agent training config and the memdpflight training config used `speed_mtp = 4.0`.

Why `action_mode = original_lag`:

- it matches the action-buffer behavior in `eval_original_like.py`;
- this keeps the protocol closer to the original-like evaluation script.

## Important Caveat About `memdpflight/env_cuda.py`

`memdpflight/env_cuda.py` adds curriculum obstacle density:

```text
phase 0 -> 30% obstacles
phase 1 -> 60% obstacles
phase 2 -> 100% obstacles
```

If evaluation uses the memdp environment and does not explicitly set `curriculum_phase = 2`, the scene can be easier than the original fixed-30-obstacle environment. For strict comparison, use root `env_cuda.py` or set:

```text
--env-kind memdp --curriculum-phase 2
```

Round 1 defaults to root original env.

## Metrics

For each model and target speed, record:

- `success_rate`;
- `success_count`;
- `crash_count`;
- `timeout_count`;
- `avg_active_speed`;
- `avg_finish_time_s`;
- `avg_min_gap`.

The primary metric is `success_rate`; the others help explain why a model succeeds or fails.

## Command

Run from repository root after building CUDA ops with `pip install -e src`.

Example:

```bash
python experiments/round1_unified_eval.py \
  --model original_checkpoint0004:original:/path/to/original/checkpoint0004.pth \
  --model v3gru_no_pre:memdp:/path/to/checkpoint0000_v3gru_no_pre.pth \
  --model phase0_done:memdp:weights/checkpoint_phase0_done.pth \
  --model checkpoint_best:memdp:weights/checkpoint_best.pth \
  --model phase1_done:memdp:weights/checkpoint_phase1_done.pth \
  --model checkpoint00005:memdp:weights/checkpoint00005.pth \
  --speeds 2,4,6,7,8,9,10 \
  --speed-mtp 2.5 \
  --num-episodes 15 \
  --batch-size 32 \
  --env-kind original
```

If a model path is missing, the script skips it and prints a warning.

## Expected Output

The script creates:

```text
experiments/results/round1_<timestamp>/round1_results.csv
experiments/results/round1_<timestamp>/round1_summary.md
experiments/results/round1_<timestamp>/round1_config.json
```

The summary table has one row per model and one column per speed, plus average success rate.

## Decision Criteria

Strong improvement:

- `phase1_done` has higher average success rate than `original_checkpoint0004`;
- it also stays better at high speeds 8, 9, and 10 m/s;
- lower crash rate supports the success-rate improvement.

Weak or conditional improvement:

- memflight improves only at low speed or only under a specific scene setting;
- `checkpoint_best` is worse than `phase1_done`, meaning training AR is not a perfect proxy for benchmark success.

No reliable improvement:

- original is equal or better after using the same `num_episodes`, `speed_mtp`, action lag, and scene generator.

## Follow-Up Experiments

After Round 1:

1. Run the same protocol with `speed_mtp = 4.0`.
2. Repeat top-2 models with `env_kind = memdp --curriculum-phase 2`.
3. Add scene ablations:
   - no ground obstacles / no random rotation;
   - ground obstacles only;
   - random rotation only;
   - both ground obstacles and random rotation.
4. Add camera sensitivity:
   - original narrow view: `cam_angle=10`, `fov_x_half_tan=0.53`;
   - memflight wide view: `cam_angle=20`, `fov_x_half_tan=0.82`.
