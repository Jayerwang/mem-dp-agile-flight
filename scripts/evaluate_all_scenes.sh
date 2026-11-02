#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RESULT_ROOT="$ROOT/results/final_three_environment"

BASE_EVAL="$ROOT/evaluation/paired_evaluation.py"
LARGE_WRAPPER="$ROOT/evaluation/scenes/large_obstacle.py"
COMPOSITE_WRAPPER="$ROOT/evaluation/scenes/staggered_double_wall.py"

ORIGINAL="$ROOT/weights/original_dp_checkpoint0004.pth"
PAST6="$ROOT/weights/memdp_past6_k6_seed20261028.pth"

for FILE in \
  "$BASE_EVAL" \
  "$LARGE_WRAPPER" \
  "$COMPOSITE_WRAPPER" \
  "$ORIGINAL" \
  "$PAST6"
do
  if [ ! -s "$FILE" ]; then
    echo "[ERROR] missing or empty: $FILE" >&2
    exit 1
  fi
done

if [ -e "$RESULT_ROOT" ]; then
  echo "[ERROR] output exists: $RESULT_ROOT" >&2
  exit 1
fi

mkdir -p "$RESULT_ROOT"

sha256sum \
  "$BASE_EVAL" \
  "$LARGE_WRAPPER" \
  "$COMPOSITE_WRAPPER" \
  "$ORIGINAL" \
  "$PAST6" \
  > "$RESULT_ROOT/input_sha256.txt"

MODEL_ARGS=(
  --model "original_dp:original:$ORIGINAL"
  --model "memdp_past6:memdp:$PAST6"
)

COMMON_ARGS=(
  --repo-root "$ROOT"
  --no-default-models
  "${MODEL_ARGS[@]}"
  --speeds 4,6,8
  --speed-mtp 2.5
  --num-episodes 15
  --batch-size 32
  --timesteps 400
  --goal-radius 5.0
  --cam-angle 20
  --fov-x-half-tan 0.82
  --action-mode original_lag
  --env-kind original
  --seed 20261103
  --device cuda
  --hidden-reset-interval 0
  --depth-occlusion-frames 0
)

audit_pairing () {
  python - "$1" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text(encoding="utf-8"))

passed = (
    data.get("pairing_passed") is True
    and data.get("mismatch_count", 0) == 0
)

print(
    f"[PAIRING] {path.parent.name}: "
    f"scenes={data.get('scene_count')} "
    f"mismatch={data.get('mismatch_count')} "
    f"passed={passed}"
)

if not passed:
    raise SystemExit(f"Pairing audit failed: {path}")
PY
}

echo "[START] mixed"

OUT="$RESULT_ROOT/mixed"
mkdir -p "$OUT"

OMP_NUM_THREADS=1 python -u "$BASE_EVAL" \
  "${COMMON_ARGS[@]}" \
  --ground-voxels \
  --random-rotation \
  --output-dir "$OUT" \
  > "$OUT/eval.log" 2>&1

audit_pairing "$OUT/pairing_audit.json"
echo "[DONE] mixed"

for LEVEL in medium hard
do
  echo "[START] large_${LEVEL}"

  OUT="$RESULT_ROOT/large_${LEVEL}"
  mkdir -p "$OUT"

  LARGE_OBSTACLE_DIFFICULTY="$LEVEL" \
  LARGE_OBSTACLE_BASE_EVALUATOR="$BASE_EVAL" \
  OMP_NUM_THREADS=1 \
  python -u "$LARGE_WRAPPER" \
    "${COMMON_ARGS[@]}" \
    --no-ground-voxels \
    --no-random-rotation \
    --output-dir "$OUT" \
    > "$OUT/eval.log" 2>&1

  audit_pairing "$OUT/pairing_audit.json"
  echo "[DONE] large_${LEVEL}"
done

for LEVEL in medium hard
do
  echo "[START] staggered_${LEVEL}"

  OUT="$RESULT_ROOT/staggered_${LEVEL}"
  mkdir -p "$OUT"

  COMPOSITE_SCENE=staggered_double_wall \
  COMPOSITE_DIFFICULTY="$LEVEL" \
  COMPOSITE_BASE_EVALUATOR="$BASE_EVAL" \
  OMP_NUM_THREADS=1 \
  python -u "$COMPOSITE_WRAPPER" \
    "${COMMON_ARGS[@]}" \
    --no-ground-voxels \
    --no-random-rotation \
    --output-dir "$OUT" \
    > "$OUT/eval.log" 2>&1

  audit_pairing "$OUT/pairing_audit.json"
  echo "[DONE] staggered_${LEVEL}"
done

echo "[ALL DONE] $RESULT_ROOT"
