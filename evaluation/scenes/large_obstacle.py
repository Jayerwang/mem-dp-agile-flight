"""Run the existing V60 paired evaluator in a deterministic large-obstacle scene.

Place this file next to ``paired_memory_diagnostic_eval_mlp.py``.  All command
line arguments are forwarded to that evaluator unchanged.  The wrapper only
replaces ``Env.reset`` after construction, so model loading, controller logic,
metrics, paired seeds, and checkpoint handling remain untouched.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import torch


DIFFICULTY = os.environ.get("LARGE_OBSTACLE_DIFFICULTY", "stress").lower()
DIFFICULTY_PROFILES = {
    "easy": {
        "block_half_x_range": [0.8, 1.1],
        "block_half_y_range": [1.4, 2.0],
    },
    "medium": {
        "block_half_x_range": [0.9, 1.3],
        "block_half_y_range": [2.0, 2.6],
    },
    "hard": {
        "block_half_x_range": [1.0, 1.4],
        "block_half_y_range": [2.6, 3.2],
    },
    # Preserves the exact V2 range used by the completed formal experiment.
    "stress": {
        "block_half_x_range": [1.0, 1.5],
        "block_half_y_range": [2.4, 3.6],
    },
}
if DIFFICULTY not in DIFFICULTY_PROFILES:
    raise ValueError(
        f"Unknown LARGE_OBSTACLE_DIFFICULTY={DIFFICULTY!r}; "
        f"choose one of {sorted(DIFFICULTY_PROFILES)}"
    )

SCENE_CONFIG = {
    "scene": "randomized_large_block_natural_occlusion_v3",
    "difficulty": DIFFICULTY,
    "start_x": -2.0,
    "start_z": 1.5,
    "start_abs_y_range": [0.35, 0.85],
    "target_x_range": [15.0, 17.0],
    "target_y_range": [-0.5, 0.5],
    "target_z": 1.5,
    # Axis-aligned block ranges. Sizes are half extents.
    "block_center_x_range": [4.1, 4.9],
    "block_center_y_range": [-0.35, 0.35],
    "block_center_z": 2.0,
    "block_half_x_range": DIFFICULTY_PROFILES[DIFFICULTY]["block_half_x_range"],
    "block_half_y_range": DIFFICULTY_PROFILES[DIFFICULTY]["block_half_y_range"],
    "block_half_z": 4.0,
    "artificial_depth_occlusion": False,
}


def _load_base_evaluator():
    configured = os.environ.get("LARGE_OBSTACLE_BASE_EVALUATOR")
    path = (
        Path(configured).expanduser().resolve()
        if configured
        else Path(__file__).with_name("paired_memory_diagnostic_eval_mlp.py")
    )
    spec = importlib.util.spec_from_file_location("paired_eval_mlp_base", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import evaluator from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _move_unused_obstacles_far_away(env) -> None:
    """Keep CUDA renderer inputs non-empty while disabling random obstacles."""
    if torch.is_tensor(getattr(env, "balls", None)):
        env.balls[..., :3] = 1000.0
        env.balls[..., 3] = 0.1
    if torch.is_tensor(getattr(env, "cyl", None)):
        env.cyl[..., :2] = 1000.0
        env.cyl[..., 2] = 0.1
    if torch.is_tensor(getattr(env, "cyl_h", None)):
        env.cyl_h[..., :2] = 1000.0
        env.cyl_h[..., 2] = 0.1


def _configure_large_obstacle_scene(env) -> None:
    batch_size = int(env.batch_size)
    device = env.device
    dtype = env.p.dtype

    def uniform(low: float, high: float) -> torch.Tensor:
        return low + (high - low) * torch.rand(batch_size, device=device, dtype=dtype)

    env.p = torch.zeros((batch_size, 3), device=device, dtype=dtype)
    env.p_target = torch.zeros_like(env.p)
    env.p[:, 0] = float(SCENE_CONFIG["start_x"])
    env.p[:, 2] = float(SCENE_CONFIG["start_z"])
    env.p_target[:, 0] = uniform(*SCENE_CONFIG["target_x_range"])
    env.p_target[:, 1] = uniform(*SCENE_CONFIG["target_y_range"])
    env.p_target[:, 2] = float(SCENE_CONFIG["target_z"])

    # Alternate the nearer side across paired trials and randomize its
    # magnitude. The torch seed is reset by the base evaluator before reset,
    # so every compared model receives bitwise-identical scene samples.
    side = torch.where(
        torch.arange(batch_size, device=device) % 2 == 0,
        torch.ones(batch_size, device=device, dtype=dtype),
        -torch.ones(batch_size, device=device, dtype=dtype),
    )
    env.p[:, 1] = side * uniform(*SCENE_CONFIG["start_abs_y_range"])
    env.p_old = env.p.clone()

    # Face along +x toward the target.  Columns are forward, left, and up.
    env.R = torch.eye(3, device=device, dtype=dtype).repeat(batch_size, 1, 1)
    env.R_old = env.R.clone()

    _move_unused_obstacles_far_away(env)
    block = torch.zeros((batch_size, 1, 6), device=device, dtype=dtype)
    block[:, 0, 0] = uniform(*SCENE_CONFIG["block_center_x_range"])
    block[:, 0, 1] = uniform(*SCENE_CONFIG["block_center_y_range"])
    block[:, 0, 2] = float(SCENE_CONFIG["block_center_z"])
    block[:, 0, 3] = uniform(*SCENE_CONFIG["block_half_x_range"])
    block[:, 0, 4] = uniform(*SCENE_CONFIG["block_half_y_range"])
    block[:, 0, 5] = float(SCENE_CONFIG["block_half_z"])
    env.voxels = block


def _install_reset_wrapper(base) -> None:
    original_build_env = base.build_env

    def build_env_with_large_obstacle(*args, **kwargs):
        env = original_build_env(*args, **kwargs)
        original_reset = env.reset

        def reset_large_obstacle():
            original_reset()
            _configure_large_obstacle_scene(env)

        env.reset = reset_large_obstacle
        return env

    base.build_env = build_env_with_large_obstacle


def _write_scene_manifest() -> None:
    try:
        index = sys.argv.index("--output-dir")
        output_dir = Path(sys.argv[index + 1])
    except (ValueError, IndexError):
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "large_obstacle_scene.json").write_text(
        json.dumps(SCENE_CONFIG, indent=2), encoding="utf-8"
    )


def main() -> None:
    if "--depth-occlusion-frames" in sys.argv:
        index = sys.argv.index("--depth-occlusion-frames")
        if index + 1 < len(sys.argv) and int(sys.argv[index + 1]) != 0:
            raise ValueError(
                "The primary large-obstacle test must use "
                "--depth-occlusion-frames 0 (natural occlusion only)."
            )
    base = _load_base_evaluator()
    _install_reset_wrapper(base)
    print("[Large obstacle scene]", json.dumps(SCENE_CONFIG, sort_keys=True))
    base.main()
    _write_scene_manifest()


if __name__ == "__main__":
    main()
