#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
# Resolve torch/lib and python env lib dynamically so this works across conda envs.
PY_PREFIX="$(python - << 'PY'
import sys
print(sys.prefix)
PY
)"
TORCH_LIB_DIR="$(python - << 'PY'
import os
import torch
print(os.path.join(os.path.dirname(torch.__file__), 'lib'))
PY
)"
export LD_LIBRARY_PATH="${TORCH_LIB_DIR}:${PY_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

echo "[1/5] Checking NVIDIA driver and GPU visibility..."
if ! nvidia-smi >/dev/null 2>&1; then
  echo "[ERROR] nvidia-smi failed. GPU may be unavailable in this session."
  exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

echo "[2/5] Checking PyTorch CUDA runtime..."
python - << 'PY'
import torch
print('torch_version:', torch.__version__)
print('torch_cuda_build:', torch.version.cuda)
print('cuda_available:', torch.cuda.is_available())
print('device_count:', torch.cuda.device_count())
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise SystemExit('[ERROR] PyTorch cannot access CUDA device.')
print('device_0:', torch.cuda.get_device_name(0))
PY

echo "[3/5] Checking quadsim_cuda extension import..."
python - << 'PY'
import importlib
importlib.import_module('quadsim_cuda')
print('quadsim_cuda_import: ok')
PY

echo "[4/5] Checking config file..."
if [ ! -f "configs/memflight_single.args" ]; then
  echo "[ERROR] Missing config file: configs/memflight_single.args"
  exit 1
fi

if [ "${1:-}" = "--check-only" ]; then
  echo "[5/5] Check-only mode complete. Environment is ready."
  exit 0
fi

echo "[5/5] Launching training..."
python main_memflight_cuda.py $(cat configs/memflight_single.args) "$@"
