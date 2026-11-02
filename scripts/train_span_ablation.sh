#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXP="$ROOT"
SOURCE="$ROOT/weights/checkpoint_phase0_done.pth"

MODE="${1:-smoke}"
SEED="${2:-20261028}"

test -s "$SOURCE"

case "$MODE" in
  smoke)
    PHASE1_ITERS=2
    PHASE1_TOTAL=3
    PHASE2_ITERS=2
    BATCH=8
    TIMESTEPS=40
    ;;
  quick)
    PHASE1_ITERS=3000
    PHASE1_TOTAL=3001
    PHASE2_ITERS=2000
    BATCH=64
    TIMESTEPS=150
    ;;
  *)
    echo "用法：bash scripts/train_span_ablation.sh smoke|quick [seed]"
    exit 2
    ;;
esac

COMMON=(
  --batch_size "$BATCH"
  --coef_v 1.0
  --coef_speed 0.0
  --coef_v_pred 2.0
  --coef_collide 7.5
  --coef_obj_avoidance 3.0
  --coef_d_acc 0.01
  --coef_d_jerk 0.001
  --coef_d_snap 0.0
  --coef_ground_affinity 0.0
  --coef_bias 0.0
  --lr 0.001
  --grad_decay 0.4
  --speed_mtp 4.0
  --fov_x_half_tan 0.82
  --timesteps "$TIMESTEPS"
  --cam_angle 20
  --fixed_curriculum_phase 2
  --seed "$SEED"
  --ground_voxels
  --random_rotation
  --yaw_drift
  --single
)

run_group () {
  local group="$1"
  local past="$2"
  local current="$3"
  local steps="$4"
  local coef_past="$5"
  local coef_current="$6"
  local phase1_recon="$7"
  local phase2_recon="$8"

  local phase1_dir="$EXP/runs/${MODE}_${group}_phase1_seed${SEED}"
  local phase2_dir="$EXP/runs/${MODE}_${group}_phase2_seed${SEED}"

  if [[ -e "$phase1_dir" || -e "$phase2_dir" ]]; then
    echo "[ERROR] 输出目录已存在，拒绝覆盖：$group"
    echo "  $phase1_dir"
    echo "  $phase2_dir"
    exit 3
  fi

  mkdir -p "$phase1_dir" "$phase2_dir"

  {
    echo "group=$group"
    echo "mode=$MODE"
    echo "seed=$SEED"
    echo "recon_members=$past $current 0"
    echo "recon_steps=$steps"
    echo "recon_coef_past=$coef_past"
    echo "recon_coef_cur=$coef_current"
    echo "phase1_recon=$phase1_recon"
    echo "phase2_recon=$phase2_recon"
    echo "phase1_updates=$PHASE1_ITERS"
    echo "phase2_updates=$PHASE2_ITERS"
  } > "$phase1_dir/config_audit.txt"

  cp "$phase1_dir/config_audit.txt" "$phase2_dir/config_audit.txt"

  echo
  echo "=================================================="
  echo "[START] $group Phase 1"
  echo "=================================================="

  (
    cd "$phase1_dir"

    PYTHONPATH="$ROOT" OMP_NUM_THREADS=1 python -u -m memdp.train \
      --resume "$SOURCE" \
      --num_iters "$PHASE1_TOTAL" \
      --phase0_iters 0 \
      --phase1_iters "$PHASE1_ITERS" \
      --stop_after_phase1 \
      --phase1_recon "$phase1_recon" \
      --phase2_recon "$phase2_recon" \
      --recon_members "$past" "$current" 0 \
      --recon_steps "$steps" \
      --recon_coef_past "$coef_past" \
      --recon_coef_cur "$coef_current" \
      "${COMMON[@]}" \
      > train.log 2>&1
  )

  test -s "$phase1_dir/checkpoint_phase1_done.pth"

  echo "[DONE] $group Phase 1"

  echo
  echo "=================================================="
  echo "[START] $group Phase 2"
  echo "=================================================="

  (
    cd "$phase2_dir"

    PYTHONPATH="$ROOT" OMP_NUM_THREADS=1 python -u -m memdp.train \
      --resume "../${MODE}_${group}_phase1_seed${SEED}/checkpoint_phase1_done.pth" \
      --num_iters "$PHASE2_ITERS" \
      --phase0_iters 0 \
      --phase1_iters 0 \
      --phase1_recon "$phase1_recon" \
      --phase2_recon "$phase2_recon" \
      --recon_members "$past" "$current" 0 \
      --recon_steps "$steps" \
      --recon_coef_past "$coef_past" \
      --recon_coef_cur "$coef_current" \
      "${COMMON[@]}" \
      > train.log 2>&1
  )

  local final_ckpt
  final_ckpt=$(printf '%s/checkpoint_iter%06d.pth' \
    "$phase2_dir" "$PHASE2_ITERS")

  test -s "$final_ckpt"

  sha256sum \
    "$SOURCE" \
    "$phase1_dir/checkpoint_phase1_done.pth" \
    "$final_ckpt" \
    > "$phase2_dir/checkpoint_sha256.txt"

  echo "[DONE] $group Phase 2"
  echo "[FINAL] $final_ckpt"
}

# group       past current K   coef_past coef_cur P1   P2
run_group no_recon 0    0      12  0.00      0.00     off  off
run_group current  0    1      12  0.00      0.15     on   on
run_group past6    1    0       6  0.15      0.00     on   on
run_group past12   1    0      12  0.15      0.00     on   on
run_group past24   1    0      24  0.15      0.00     on   on

echo
echo "[ALL DONE] mode=$MODE seed=$SEED"
