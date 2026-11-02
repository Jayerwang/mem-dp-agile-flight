"""Paired evaluation in deterministic composite-box obstacle scenes.

Supported scenes:
  staggered_double_wall:
      Two longitudinally separated blocks with opposite lateral offsets.
  offset_gate:
      Two wide blocks form a laterally offset opening.

The wrapper changes only Env.reset. Model loading, controller execution,
metrics, scene seeding, and paired evaluation remain in the base evaluator.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import torch


SCENE = os.environ.get(
    "COMPOSITE_SCENE",
    "staggered_double_wall",
).lower()

DIFFICULTY = os.environ.get(
    "COMPOSITE_DIFFICULTY",
    "medium",
).lower()

SUPPORTED_SCENES = {
    "staggered_double_wall",
    "offset_gate",
}

if SCENE not in SUPPORTED_SCENES:
    raise ValueError(
        f"Unknown COMPOSITE_SCENE={SCENE!r}; "
        f"choose one of {sorted(SUPPORTED_SCENES)}"
    )


PROFILES = {
    "easy": {
        "block_half_x_range": [0.75, 0.95],
        "stagger_half_y_range": [1.4, 1.8],
        "stagger_offset_range": [1.2, 1.6],
        "stagger_x1_range": [3.6, 4.2],
        "stagger_x2_range": [7.2, 8.0],
        "gate_half_gap_range": [1.25, 1.55],
        "gate_offset_range": [0.5, 0.9],
        "gate_wall_half_y": 4.0,
    },
    "medium": {
        "block_half_x_range": [0.85, 1.10],
        "stagger_half_y_range": [2.0, 2.4],
        "stagger_offset_range": [0.9, 1.3],
        "stagger_x1_range": [3.6, 4.2],
        "stagger_x2_range": [7.0, 7.8],
        "gate_half_gap_range": [0.90, 1.20],
        "gate_offset_range": [0.8, 1.2],
        "gate_wall_half_y": 4.5,
    },
    "hard": {
        "block_half_x_range": [0.95, 1.25],
        "stagger_half_y_range": [2.5, 3.0],
        "stagger_offset_range": [0.6, 1.0],
        "stagger_x1_range": [3.5, 4.2],
        "stagger_x2_range": [6.6, 7.4],
        "gate_half_gap_range": [0.60, 0.85],
        "gate_offset_range": [1.0, 1.5],
        "gate_wall_half_y": 5.0,
    },
}

if DIFFICULTY not in PROFILES:
    raise ValueError(
        f"Unknown COMPOSITE_DIFFICULTY={DIFFICULTY!r}; "
        f"choose one of {sorted(PROFILES)}"
    )

PROFILE = PROFILES[DIFFICULTY]

SCENE_CONFIG = {
    "scene": SCENE,
    "difficulty": DIFFICULTY,
    "representation": "axis_aligned_box_union",
    "start_x": -2.0,
    "start_z": 1.5,
    "start_abs_y_range": [0.35, 0.85],
    "target_x_range": [15.0, 17.0],
    "target_y_range": [-0.5, 0.5],
    "target_z": 1.5,
    "block_center_z": 2.0,
    "block_half_z": 4.0,
    "profile": PROFILE,
    "artificial_depth_occlusion": False,
}


def _load_base_evaluator():
    configured = os.environ.get("COMPOSITE_BASE_EVALUATOR")
    if not configured:
        raise ValueError(
            "COMPOSITE_BASE_EVALUATOR must point to the paired evaluator."
        )

    path = Path(configured).expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(
        "paired_eval_composite_base",
        path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import evaluator from {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _move_unused_obstacles_far_away(env) -> None:
    if torch.is_tensor(getattr(env, "balls", None)):
        env.balls[..., :3] = 1000.0
        env.balls[..., 3] = 0.1

    if torch.is_tensor(getattr(env, "cyl", None)):
        env.cyl[..., :2] = 1000.0
        env.cyl[..., 2] = 0.1

    if torch.is_tensor(getattr(env, "cyl_h", None)):
        env.cyl_h[..., :2] = 1000.0
        env.cyl_h[..., 2] = 0.1


def _configure_start_and_target(env):
    batch_size = int(env.batch_size)
    device = env.device
    dtype = env.p.dtype

    def uniform(low, high):
        return low + (high - low) * torch.rand(
            batch_size,
            device=device,
            dtype=dtype,
        )

    # Mirror half of the batch to prevent left/right selection bias.
    mirror = torch.where(
        torch.arange(batch_size, device=device) % 2 == 0,
        torch.ones(batch_size, device=device, dtype=dtype),
        -torch.ones(batch_size, device=device, dtype=dtype),
    )

    env.p = torch.zeros(
        (batch_size, 3),
        device=device,
        dtype=dtype,
    )
    env.p_target = torch.zeros_like(env.p)

    env.p[:, 0] = float(SCENE_CONFIG["start_x"])
    env.p[:, 1] = mirror * uniform(
        *SCENE_CONFIG["start_abs_y_range"]
    )
    env.p[:, 2] = float(SCENE_CONFIG["start_z"])

    env.p_target[:, 0] = uniform(
        *SCENE_CONFIG["target_x_range"]
    )
    env.p_target[:, 1] = uniform(
        *SCENE_CONFIG["target_y_range"]
    )
    env.p_target[:, 2] = float(
        SCENE_CONFIG["target_z"]
    )

    env.p_old = env.p.clone()

    env.R = torch.eye(
        3,
        device=device,
        dtype=dtype,
    ).repeat(batch_size, 1, 1)
    env.R_old = env.R.clone()

    return mirror, uniform


def _build_staggered_double_wall(env, mirror, uniform):
    """Two blocks with opposite lateral offsets.

    The first block pushes the vehicle toward one side; the second block
    is mirrored and requires a subsequent direction correction.
    """
    batch_size = int(env.batch_size)
    device = env.device
    dtype = env.p.dtype

    blocks = torch.zeros(
        (batch_size, 2, 6),
        device=device,
        dtype=dtype,
    )

    x1 = uniform(*PROFILE["stagger_x1_range"])
    x2 = uniform(*PROFILE["stagger_x2_range"])

    offset1 = uniform(*PROFILE["stagger_offset_range"])
    offset2 = uniform(*PROFILE["stagger_offset_range"])

    half_x1 = uniform(*PROFILE["block_half_x_range"])
    half_x2 = uniform(*PROFILE["block_half_x_range"])

    half_y1 = uniform(*PROFILE["stagger_half_y_range"])
    half_y2 = uniform(*PROFILE["stagger_half_y_range"])

    # Start and first block are on the same side, encouraging the vehicle
    # to cross toward the opposite side. The second block is mirrored.
    blocks[:, 0, 0] = x1
    blocks[:, 0, 1] = mirror * offset1
    blocks[:, 0, 2] = float(SCENE_CONFIG["block_center_z"])
    blocks[:, 0, 3] = half_x1
    blocks[:, 0, 4] = half_y1
    blocks[:, 0, 5] = float(SCENE_CONFIG["block_half_z"])

    blocks[:, 1, 0] = x2
    blocks[:, 1, 1] = -mirror * offset2
    blocks[:, 1, 2] = float(SCENE_CONFIG["block_center_z"])
    blocks[:, 1, 3] = half_x2
    blocks[:, 1, 4] = half_y2
    blocks[:, 1, 5] = float(SCENE_CONFIG["block_half_z"])

    return blocks


def _build_offset_gate(env, mirror, uniform):
    """Two wide boxes form a laterally offset opening."""
    batch_size = int(env.batch_size)
    device = env.device
    dtype = env.p.dtype

    blocks = torch.zeros(
        (batch_size, 2, 6),
        device=device,
        dtype=dtype,
    )

    gate_x = uniform(4.2, 5.0)
    gate_offset = mirror * uniform(
        *PROFILE["gate_offset_range"]
    )
    half_gap = uniform(
        *PROFILE["gate_half_gap_range"]
    )
    wall_half_y = float(PROFILE["gate_wall_half_y"])
    half_x = uniform(*PROFILE["block_half_x_range"])

    # The lower and upper boxes have inner faces at:
    # gate_offset-half_gap and gate_offset+half_gap.
    lower_center_y = gate_offset - half_gap - wall_half_y
    upper_center_y = gate_offset + half_gap + wall_half_y

    blocks[:, 0, 0] = gate_x
    blocks[:, 0, 1] = lower_center_y
    blocks[:, 0, 2] = float(SCENE_CONFIG["block_center_z"])
    blocks[:, 0, 3] = half_x
    blocks[:, 0, 4] = wall_half_y
    blocks[:, 0, 5] = float(SCENE_CONFIG["block_half_z"])

    blocks[:, 1, 0] = gate_x
    blocks[:, 1, 1] = upper_center_y
    blocks[:, 1, 2] = float(SCENE_CONFIG["block_center_z"])
    blocks[:, 1, 3] = half_x
    blocks[:, 1, 4] = wall_half_y
    blocks[:, 1, 5] = float(SCENE_CONFIG["block_half_z"])

    return blocks


def _configure_scene(env) -> None:
    mirror, uniform = _configure_start_and_target(env)
    _move_unused_obstacles_far_away(env)

    if SCENE == "staggered_double_wall":
        env.voxels = _build_staggered_double_wall(
            env,
            mirror,
            uniform,
        )
    elif SCENE == "offset_gate":
        env.voxels = _build_offset_gate(
            env,
            mirror,
            uniform,
        )
    else:
        raise AssertionError(SCENE)


def _install_reset_wrapper(base) -> None:
    original_build_env = base.build_env

    def build_env_with_composite_scene(*args, **kwargs):
        env = original_build_env(*args, **kwargs)
        original_reset = env.reset

        def reset_composite_scene():
            original_reset()
            _configure_scene(env)

            if env.voxels.ndim != 3:
                raise RuntimeError(
                    f"env.voxels must be 3D, got {env.voxels.shape}"
                )
            if env.voxels.shape[1] != 2:
                raise RuntimeError(
                    "Composite scene must contain exactly two boxes; "
                    f"got shape={tuple(env.voxels.shape)}"
                )
            if env.voxels.shape[2] != 6:
                raise RuntimeError(
                    f"Last voxel dimension must be 6; "
                    f"got shape={tuple(env.voxels.shape)}"
                )

        env.reset = reset_composite_scene
        return env

    base.build_env = build_env_with_composite_scene


def _write_scene_manifest() -> None:
    try:
        index = sys.argv.index("--output-dir")
        output_dir = Path(sys.argv[index + 1])
    except (ValueError, IndexError):
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / "composite_obstacle_scene.json"
    path.write_text(
        json.dumps(SCENE_CONFIG, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    if "--depth-occlusion-frames" in sys.argv:
        index = sys.argv.index("--depth-occlusion-frames")
        if (
            index + 1 < len(sys.argv)
            and int(sys.argv[index + 1]) != 0
        ):
            raise ValueError(
                "Composite tests require "
                "--depth-occlusion-frames 0."
            )

    base = _load_base_evaluator()
    _install_reset_wrapper(base)

    print(
        "[Composite obstacle scene]",
        json.dumps(SCENE_CONFIG, sort_keys=True),
    )

    base.main()
    _write_scene_manifest()


if __name__ == "__main__":
    main()
