import argparse
import csv
import hashlib
import importlib.util
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import trange


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str
    path: Path


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_bool_csv(raw: str) -> List[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_model_spec(raw: str, repo_root: Path) -> ModelSpec:
    parts = raw.split(":", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "--model must use name:kind:path, e.g. phase1_done:memdp:weights/checkpoint_phase1_done.pth"
        )
    name, kind, path_raw = parts
    kind = kind.lower()
    if kind not in {"original", "memdp"}:
        raise argparse.ArgumentTypeError("model kind must be original or memdp")
    path = Path(path_raw)
    if not path.is_absolute():
        path = repo_root / path
    return ModelSpec(name=name, kind=kind, path=path)


def default_model_specs(repo_root: Path) -> List[ModelSpec]:
    candidates = [
        ("phase0_done", "memdp", repo_root / "weights" / "checkpoint_phase0_done.pth"),
        ("checkpoint_best", "memdp", repo_root / "weights" / "checkpoint_best.pth"),
        ("phase1_done", "memdp", repo_root / "weights" / "checkpoint_phase1_done.pth"),
        ("checkpoint00005", "memdp", repo_root / "weights" / "checkpoint00005.pth"),
        ("original_checkpoint0004", "original", repo_root / "weights" / "original_checkpoint0004.pth"),
        ("v3gru_no_pre", "memdp", repo_root / "weights" / "v3gru_no_pre.pth"),
    ]
    return [ModelSpec(name, kind, path) for name, kind, path in candidates]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Round 1 unified original-like CUDA evaluator")
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Model spec in name:kind:path format. kind is original or memdp.",
    )
    parser.add_argument("--no-default-models", action="store_true")
    parser.add_argument("--speeds", type=str, default="2,4,6,7,8,9,10")
    parser.add_argument("--num-episodes", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--timesteps", type=int, default=400)
    parser.add_argument("--goal-radius", type=float, default=5.0)
    parser.add_argument("--speed-mtp", type=float, default=2.5)
    parser.add_argument("--grad-decay", type=float, default=0.4)
    parser.add_argument("--fov-x-half-tan", type=float, default=0.82)
    parser.add_argument("--cam-angle", type=float, default=20.0)
    parser.add_argument("--env-kind", choices=["original", "memdp"], default="original")
    parser.add_argument("--curriculum-phase", type=int, default=2)
    parser.add_argument("--action-mode", choices=["original_lag", "immediate"], default="original_lag")
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")

    parser.add_argument("--single", action="store_true", default=True)
    parser.add_argument("--gate", action="store_true", default=False)
    parser.add_argument("--scaffold", action="store_true", default=False)
    parser.add_argument("--no-odom", action="store_true", default=False)

    parser.add_argument("--ground-voxels", dest="ground_voxels", action="store_true", default=True)
    parser.add_argument("--no-ground-voxels", dest="ground_voxels", action="store_false")
    parser.add_argument("--random-rotation", dest="random_rotation", action="store_true", default=True)
    parser.add_argument("--no-random-rotation", dest="random_rotation", action="store_false")

    parser.add_argument("--recon-members", nargs=3, type=int, default=[1, 1, 0])
    parser.add_argument("--recon-steps", type=int, default=12)
    parser.add_argument("--enable-v-scale", action="store_true", default=False)
    parser.add_argument(
        "--hidden-reset-interval",
        type=int,
        default=0,
        help="Reset recurrent hidden state every K policy steps; 0 keeps normal memory.",
    )
    parser.add_argument(
        "--depth-occlusion-frames",
        type=int,
        default=0,
        help="Number of consecutive depth-input frames replaced in each occlusion window.",
    )
    parser.add_argument("--depth-occlusion-start", type=int, default=60)
    parser.add_argument(
        "--depth-occlusion-interval",
        type=int,
        default=0,
        help="Repeat occlusion every K steps; 0 applies one window only.",
    )
    parser.add_argument("--depth-occlusion-value", type=float, default=0.0)
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def build_env(args: argparse.Namespace, repo_root: Path, device: torch.device):
    if args.env_kind == "original":
        env_module = load_module("round1_original_env_cuda", repo_root / "env_cuda.py")
    else:
        env_module = load_module("round1_memdp_env_cuda", repo_root / "memdp" / "env_cuda.py")

    env = env_module.Env(
        batch_size=args.batch_size,
        width=64,
        height=48,
        grad_decay=args.grad_decay,
        device=device,
        fov_x_half_tan=args.fov_x_half_tan,
        single=args.single,
        gate=args.gate,
        ground_voxels=args.ground_voxels,
        scaffold=args.scaffold,
        speed_mtp=args.speed_mtp,
        random_rotation=args.random_rotation,
        cam_angle=args.cam_angle,
    )
    if hasattr(env, "curriculum_phase"):
        env.curriculum_phase = int(args.curriculum_phase)
    return env


def build_model(spec: ModelSpec, args: argparse.Namespace, repo_root: Path, device: torch.device, dim_obs: int):
    if spec.kind == "original":
        model_module = load_module(f"round1_model_{spec.name}", repo_root / "model.py")
        model = model_module.Model(dim_obs, 6).to(device)
    else:
        model_module = load_module(f"round1_model_{spec.name}", repo_root / "memdp" / "model.py")
        model = model_module.Model(
            dim_obs=dim_obs,
            dim_action=6,
            recon_members=tuple(bool(x) for x in args.recon_members),
            recon_steps=args.recon_steps,
            enable_v_scale=args.enable_v_scale,
        ).to(device)

    checkpoint = torch.load(spec.path, map_location=device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    load_result = model.load_state_dict(state_dict, strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        print(
            f"[load_state_dict] {spec.name}: missing={load_result.missing_keys}, "
            f"unexpected={load_result.unexpected_keys}"
        )
    model.eval()
    return model


def policy_forward(model, x: torch.Tensor, state: torch.Tensor, h, args: argparse.Namespace):
    with torch.no_grad():
        out = model(x, state, h)
    if isinstance(out, tuple) and len(out) == 4:
        act, _, h, _ = out
    elif isinstance(out, tuple) and len(out) == 3:
        act, _, h = out
    elif isinstance(out, tuple) and len(out) == 2:
        act, h = out
    else:
        raise RuntimeError(f"Unexpected model output type/length: {type(out)}")
    return act, h


def active_mean(values: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    if bool(active.any()):
        return values[active].mean()
    return torch.tensor(0.0, device=values.device)


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if bool(mask.any()):
        return values[mask].mean()
    return torch.tensor(0.0, device=values.device)


def validate_diagnostic_args(args: argparse.Namespace) -> None:
    if args.hidden_reset_interval < 0:
        raise ValueError("--hidden-reset-interval must be >= 0")
    if args.depth_occlusion_frames < 0:
        raise ValueError("--depth-occlusion-frames must be >= 0")
    if args.depth_occlusion_start < 0:
        raise ValueError("--depth-occlusion-start must be >= 0")
    if args.depth_occlusion_interval < 0:
        raise ValueError("--depth-occlusion-interval must be >= 0")
    if 0 < args.depth_occlusion_interval < args.depth_occlusion_frames:
        raise ValueError("occlusion interval must be 0 or >= occlusion frames")


def is_occluded_step(t: int, args: argparse.Namespace) -> bool:
    frames = args.depth_occlusion_frames
    start = args.depth_occlusion_start
    if frames <= 0 or t < start:
        return False
    if args.depth_occlusion_interval <= 0:
        return t < start + frames
    return ((t - start) % args.depth_occlusion_interval) < frames


def _tensor_slice_for_trial(value: torch.Tensor, trial_index: int, batch_size: int) -> torch.Tensor:
    value = value.detach()
    if value.ndim > 0 and value.shape[0] == batch_size:
        value = value[trial_index]
    elif value.ndim > 1 and value.shape[1] == batch_size:
        value = value[:, trial_index]
    return value.contiguous().cpu()


def scene_hashes(env, batch_size: int) -> List[str]:
    """Best-effort hash of reset geometry; used to audit pairing, not as a security hash."""
    names = (
        "p_init", "p_target", "balls", "cyl", "cyl_h", "voxels",
        "ground_voxel_w", "ground_voxel_b", "gate_w", "gate_b",
    )
    hashes: List[str] = []
    for trial_index in range(batch_size):
        digest = hashlib.sha256()
        found = False
        for name in names:
            value = getattr(env, name, None)
            if not torch.is_tensor(value):
                continue
            found = True
            selected = _tensor_slice_for_trial(value, trial_index, batch_size)
            digest.update(name.encode("utf-8"))
            digest.update(str(tuple(selected.shape)).encode("ascii"))
            digest.update(selected.numpy().tobytes())
        hashes.append(digest.hexdigest() if found else "unavailable")
    return hashes


def run_episode(
    env,
    model,
    args: argparse.Namespace,
    target_speed: float,
    episode_seed: int,
) -> Tuple[Dict[str, float], List[Dict[str, object]]]:
    torch.manual_seed(episode_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(episode_seed)

    bsz = args.batch_size
    device = env.device
    ctl_dt = 1.0 / 15.0

    if hasattr(env, "curriculum_phase"):
        env.curriculum_phase = int(args.curriculum_phase)
    env.reset()
    env.max_speed = torch.full_like(env.max_speed, float(target_speed))
    model.reset()
    reset_scene_hashes = scene_hashes(env, bsz)

    h = None
    done = torch.zeros(bsz, dtype=torch.bool, device=device)
    success = torch.zeros(bsz, dtype=torch.bool, device=device)
    crashed_final = torch.zeros(bsz, dtype=torch.bool, device=device)
    finish_step = torch.full((bsz,), args.timesteps, dtype=torch.float32, device=device)
    min_gap = torch.full((bsz,), float("inf"), dtype=torch.float32, device=device)
    speed_sum = torch.zeros(bsz, dtype=torch.float32, device=device)
    speed_count = torch.zeros(bsz, dtype=torch.float32, device=device)
    start_goal_dist = torch.norm(env.p_target - env.p, 2, -1).detach()
    prev_goal_dist = start_goal_dist.clone()
    final_goal_dist = start_goal_dist.clone()
    progress_sum = torch.zeros(bsz, dtype=torch.float32, device=device)
    forward_speed_sum = torch.zeros(bsz, dtype=torch.float32, device=device)
    active_time_s = torch.zeros(bsz, dtype=torch.float32, device=device)
    goal_alignment_sum = torch.zeros(bsz, dtype=torch.float32, device=device)
    positive_progress_count = torch.zeros(bsz, dtype=torch.float32, device=device)
    stall_count = torch.zeros(bsz, dtype=torch.float32, device=device)
    active_step_count = torch.zeros(bsz, dtype=torch.float32, device=device)

    act_buffer = [env.act] * 2
    margin = env.margin.reshape(-1)

    for t in range(args.timesteps):
        depth, _ = env.render(ctl_dt)
        target_v_raw = env.p_target - env.p

        if args.action_mode == "original_lag":
            action_to_apply = act_buffer[t]
        else:
            action_to_apply = act_buffer[-1]
        env.run(action_to_apply, ctl_dt, target_v_raw)

        active = ~done
        speed = torch.norm(env.v, 2, -1)
        speed_sum[active] += speed[active]
        speed_count[active] += 1
        active_time_s[active] += ctl_dt

        vec_to_pt = env.find_vec_to_nearest_pt()
        nearest_dist = torch.norm(vec_to_pt, 2, -1).min(dim=0).values
        gap = nearest_dist - margin
        min_gap = torch.minimum(min_gap, gap.detach())
        dist = torch.norm(env.p_target - env.p, 2, -1)
        step_progress = (prev_goal_dist - dist).detach()
        progress_sum[active] += step_progress[active]
        forward_speed_sum[active] += step_progress[active] / ctl_dt
        goal_unit = (env.p_target - env.p) / dist[:, None].clamp_min(1e-6)
        velocity_unit = env.v / speed[:, None].clamp_min(1e-6)
        alignment = (goal_unit * velocity_unit).sum(-1)
        moving = speed > 1e-4
        goal_alignment_sum[active & moving] += alignment[active & moving]
        positive_progress_count[active] += (step_progress[active] > 0).float()
        stall_count[active] += ((speed[active] < 0.2) | (step_progress[active].abs() < 1e-3)).float()
        active_step_count[active] += 1
        final_goal_dist[active] = dist[active].detach()
        prev_goal_dist = dist.detach()

        reached_goal = dist <= args.goal_radius
        crashed = gap < 0
        newly_done = (~done) & (reached_goal | crashed)
        success = success | (newly_done & reached_goal & (~crashed))
        crashed_final = crashed_final | (newly_done & crashed)
        finish_step[newly_done] = float(t + 1)
        done = done | newly_done

        if bool(torch.all(done)):
            break

        fwd = env.R[:, :, 0].clone()
        fwd[:, 2] = 0
        up = torch.zeros_like(fwd)
        up[:, 2] = 1
        fwd = F.normalize(fwd, 2, -1)
        R = torch.stack([fwd, torch.cross(up, fwd, dim=-1), up], -1)

        target_v_norm = torch.norm(target_v_raw, 2, -1, keepdim=True)
        target_v_unit = target_v_raw / target_v_norm.clamp_min(1e-6)
        target_v = target_v_unit * torch.minimum(target_v_norm, env.max_speed)

        local_v = torch.squeeze(env.v[:, None] @ R, 1)
        state_parts = [
            torch.squeeze(target_v[:, None] @ R, 1),
            env.R[:, 2],
            margin[:, None],
        ]
        if not args.no_odom:
            state_parts.insert(0, local_v)
        state = torch.cat(state_parts, -1)

        x = 3.0 / depth.clamp(0.3, 24.0) - 0.6
        x = F.max_pool2d(x[:, None], 4, 4)

        if is_occluded_step(t, args):
            x = torch.full_like(x, args.depth_occlusion_value)

        if args.hidden_reset_interval > 0 and t > 0 and t % args.hidden_reset_interval == 0:
            h = None

        act, h = policy_forward(model, x, state, h, args)
        a_pred, v_pred, *_ = (R @ act.reshape(bsz, 3, -1)).unbind(-1)
        act = (a_pred - v_pred - env.g_std) * env.thr_est_error[:, None] + env.g_std
        act_buffer.append(act)

    timeout = ~done
    avg_speed_per_trial = speed_sum / speed_count.clamp_min(1)
    avg_forward_speed_per_trial = forward_speed_sum / speed_count.clamp_min(1)
    normalized_progress = progress_sum / start_goal_dist.clamp_min(1e-6)
    goal_alignment = goal_alignment_sum / active_step_count.clamp_min(1)
    positive_progress_ratio = positive_progress_count / active_step_count.clamp_min(1)
    stall_ratio = stall_count / active_step_count.clamp_min(1)

    aggregate = {
        "success_count": int(success.sum().item()),
        "crash_count": int(crashed_final.sum().item()),
        "timeout_count": int(timeout.sum().item()),
        "avg_active_speed": float(avg_speed_per_trial.mean().item()),
        "avg_forward_speed": float(avg_forward_speed_per_trial.mean().item()),
        "avg_progress": float(progress_sum.mean().item()),
        "avg_normalized_progress": float(normalized_progress.mean().item()),
        "avg_start_goal_dist": float(start_goal_dist.mean().item()),
        "avg_final_goal_dist": float(final_goal_dist.mean().item()),
        "timeout_final_goal_dist": float(masked_mean(final_goal_dist, timeout).item()),
        "success_final_goal_dist": float(masked_mean(final_goal_dist, success).item()),
        "crash_final_goal_dist": float(masked_mean(final_goal_dist, crashed_final).item()),
        "avg_active_time_s": float(active_time_s.mean().item()),
        "avg_finish_time_s": float((finish_step * ctl_dt).mean().item()),
        "avg_min_gap": float(min_gap.mean().item()),
        "avg_goal_alignment": float(goal_alignment.mean().item()),
        "avg_positive_progress_ratio": float(positive_progress_ratio.mean().item()),
        "avg_stall_ratio": float(stall_ratio.mean().item()),
    }

    trial_rows: List[Dict[str, object]] = []
    for trial_index in range(bsz):
        if bool(success[trial_index]):
            outcome = "success"
        elif bool(crashed_final[trial_index]):
            outcome = "crash"
        else:
            outcome = "timeout"
        trial_rows.append(
            {
                "episode_seed": episode_seed,
                "batch_index": trial_index,
                "scene_id": f"{target_speed:g}_{episode_seed}_{trial_index:03d}",
                "scene_hash": reset_scene_hashes[trial_index],
                "outcome": outcome,
                "success": int(bool(success[trial_index])),
                "crash": int(bool(crashed_final[trial_index])),
                "timeout": int(bool(timeout[trial_index])),
                "start_goal_dist": float(start_goal_dist[trial_index].item()),
                "final_goal_dist": float(final_goal_dist[trial_index].item()),
                "progress": float(progress_sum[trial_index].item()),
                "normalized_progress": float(normalized_progress[trial_index].item()),
                "avg_active_speed": float(avg_speed_per_trial[trial_index].item()),
                "avg_forward_speed": float(avg_forward_speed_per_trial[trial_index].item()),
                "active_time_s": float(active_time_s[trial_index].item()),
                "finish_time_s": float((finish_step[trial_index] * ctl_dt).item()),
                "min_gap": float(min_gap[trial_index].item()),
                "goal_alignment": float(goal_alignment[trial_index].item()),
                "positive_progress_ratio": float(positive_progress_ratio[trial_index].item()),
                "stall_ratio": float(stall_ratio[trial_index].item()),
            }
        )
    return aggregate, trial_rows


def evaluate_model_speed(
    env,
    model,
    spec: ModelSpec,
    args: argparse.Namespace,
    speed: float,
    speed_index: int,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    total = {
        "success_count": 0,
        "crash_count": 0,
        "timeout_count": 0,
        "avg_active_speed": 0.0,
        "avg_forward_speed": 0.0,
        "avg_progress": 0.0,
        "avg_normalized_progress": 0.0,
        "avg_start_goal_dist": 0.0,
        "avg_final_goal_dist": 0.0,
        "timeout_final_goal_dist": 0.0,
        "success_final_goal_dist": 0.0,
        "crash_final_goal_dist": 0.0,
        "avg_active_time_s": 0.0,
        "avg_finish_time_s": 0.0,
        "avg_min_gap": 0.0,
        "avg_goal_alignment": 0.0,
        "avg_positive_progress_ratio": 0.0,
        "avg_stall_ratio": 0.0,
    }
    trajectory_rows: List[Dict[str, object]] = []

    for ep in trange(args.num_episodes, desc=f"{spec.name} @ {speed:g}m/s", leave=False):
        episode_seed = args.seed + speed_index * 10000 + ep
        metrics, episode_rows = run_episode(env, model, args, speed, episode_seed)
        for key in total:
            total[key] += metrics[key]
        for trial_row in episode_rows:
            trial_row.update(
                {
                    "model": spec.name,
                    "model_kind": spec.kind,
                    "checkpoint": str(spec.path),
                    "target_speed": speed,
                    "hidden_reset_interval": args.hidden_reset_interval,
                    "depth_occlusion_frames": args.depth_occlusion_frames,
                    "depth_occlusion_start": args.depth_occlusion_start,
                    "depth_occlusion_interval": args.depth_occlusion_interval,
                }
            )
            trajectory_rows.append(trial_row)

    total_trials = args.num_episodes * args.batch_size
    row: Dict[str, object] = {
        "model": spec.name,
        "model_kind": spec.kind,
        "checkpoint": str(spec.path),
        "target_speed": speed,
        "total_trials": total_trials,
        "success_count": total["success_count"],
        "crash_count": total["crash_count"],
        "timeout_count": total["timeout_count"],
        "success_rate": total["success_count"] / total_trials,
        "crash_rate": total["crash_count"] / total_trials,
        "timeout_rate": total["timeout_count"] / total_trials,
        "avg_active_speed": total["avg_active_speed"] / args.num_episodes,
        "avg_forward_speed": total["avg_forward_speed"] / args.num_episodes,
        "avg_progress": total["avg_progress"] / args.num_episodes,
        "avg_normalized_progress": total["avg_normalized_progress"] / args.num_episodes,
        "avg_start_goal_dist": total["avg_start_goal_dist"] / args.num_episodes,
        "avg_final_goal_dist": total["avg_final_goal_dist"] / args.num_episodes,
        "timeout_final_goal_dist": total["timeout_final_goal_dist"] / args.num_episodes,
        "success_final_goal_dist": total["success_final_goal_dist"] / args.num_episodes,
        "crash_final_goal_dist": total["crash_final_goal_dist"] / args.num_episodes,
        "avg_active_time_s": total["avg_active_time_s"] / args.num_episodes,
        "avg_finish_time_s": total["avg_finish_time_s"] / args.num_episodes,
        "avg_min_gap": total["avg_min_gap"] / args.num_episodes,
        "avg_goal_alignment": total["avg_goal_alignment"] / args.num_episodes,
        "avg_positive_progress_ratio": total["avg_positive_progress_ratio"] / args.num_episodes,
        "avg_stall_ratio": total["avg_stall_ratio"] / args.num_episodes,
        "hidden_reset_interval": args.hidden_reset_interval,
        "depth_occlusion_frames": args.depth_occlusion_frames,
        "depth_occlusion_start": args.depth_occlusion_start,
        "depth_occlusion_interval": args.depth_occlusion_interval,
    }
    return row, trajectory_rows


def write_csv(rows: List[Dict[str, object]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(rows: List[Dict[str, object]], speeds: List[float], path: Path) -> None:
    by_model: Dict[str, Dict[float, Dict[str, object]]] = {}
    for row in rows:
        by_model.setdefault(str(row["model"]), {})[float(row["target_speed"])] = row

    lines = [
        "# Round 1 Summary",
        "",
        "Success rate table:",
        "",
    ]
    header = ["Model"] + [f"{speed:g} m/s" for speed in speeds] + ["Avg"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] + ["---:"] * (len(header) - 1)) + " |")

    for model_name, speed_rows in by_model.items():
        values: List[Optional[float]] = []
        cells = [model_name]
        for speed in speeds:
            row = speed_rows.get(float(speed))
            if row is None:
                cells.append("")
                values.append(None)
            else:
                rate = float(row["success_rate"])
                cells.append(f"{rate * 100:.2f}%")
                values.append(rate)
        valid = [v for v in values if v is not None]
        avg = sum(valid) / len(valid) if valid else 0.0
        cells.append(f"{avg * 100:.2f}%")
        lines.append("| " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "Crash, timeout, and low-speed diagnosis details are in `round1_results.csv`.",
            "",
            "Additional diagnostic columns:",
            "",
            "- `avg_final_goal_dist`: mean final distance to the target for all trials.",
            "- `timeout_final_goal_dist`: mean final target distance among timeout trials only.",
            "- `avg_progress`: mean distance reduction from start to finish/timeout.",
            "- `avg_normalized_progress`: progress divided by initial target distance.",
            "- `avg_forward_speed`: mean target-distance reduction speed while active.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def save_config(args: argparse.Namespace, model_specs: Iterable[ModelSpec], output_dir: Path) -> None:
    config = vars(args).copy()
    config["repo_root"] = str(args.repo_root)
    config["output_dir"] = str(output_dir)
    config["model_specs"] = [
        {"name": spec.name, "kind": spec.kind, "path": str(spec.path)} for spec in model_specs
    ]
    (output_dir / "round1_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def write_pairing_audit(rows: List[Dict[str, object]], path: Path) -> None:
    by_scene: Dict[str, Dict[str, str]] = {}
    for row in rows:
        scene_id = str(row["scene_id"])
        by_scene.setdefault(scene_id, {})[str(row["model"])] = str(row["scene_hash"])
    mismatches = []
    unavailable = 0
    for scene_id, model_hashes in by_scene.items():
        hashes = set(model_hashes.values())
        if "unavailable" in hashes:
            unavailable += 1
        if len(hashes) != 1:
            mismatches.append({"scene_id": scene_id, "model_hashes": model_hashes})
    audit = {
        "scene_count": len(by_scene),
        "mismatch_count": len(mismatches),
        "unavailable_hash_count": unavailable,
        "pairing_passed": len(mismatches) == 0 and unavailable == 0,
        "mismatches": mismatches[:100],
    }
    path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    validate_diagnostic_args(args)
    repo_root = args.repo_root.resolve()
    speeds = parse_bool_csv(args.speeds)
    device = resolve_device(args.device)

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    if args.output_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = repo_root / "experiments" / "results" / f"round1_{timestamp}"
    else:
        output_dir = args.output_dir
        if not output_dir.is_absolute():
            output_dir = repo_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    model_specs: List[ModelSpec] = []
    if not args.no_default_models and not args.model:
        model_specs.extend(default_model_specs(repo_root))
    for raw in args.model:
        model_specs.append(parse_model_spec(raw, repo_root))

    existing_specs = []
    for spec in model_specs:
        if spec.path.exists():
            existing_specs.append(spec)
        else:
            print(f"[skip] Missing checkpoint for {spec.name}: {spec.path}")
    if not existing_specs:
        raise FileNotFoundError("No model checkpoint exists. Pass at least one --model name:kind:path.")

    dim_obs = 7 if args.no_odom else 10
    rows: List[Dict[str, object]] = []
    trajectory_rows: List[Dict[str, object]] = []

    print(f"Repo root: {repo_root}")
    print(f"Output dir: {output_dir}")
    print(f"Device: {device}")
    print(f"Env kind: {args.env_kind}")
    print(f"Speeds: {speeds}")
    print(f"Models: {[spec.name for spec in existing_specs]}")

    env = build_env(args, repo_root, device)

    for spec in existing_specs:
        model = build_model(spec, args, repo_root, device, dim_obs)
        for speed_index, speed in enumerate(speeds):
            row, speed_trajectory_rows = evaluate_model_speed(env, model, spec, args, speed, speed_index)
            rows.append(row)
            trajectory_rows.extend(speed_trajectory_rows)
            print(
                f"[done] {spec.name} @ {speed:g}m/s: "
                f"success={row['success_rate'] * 100:.2f}% "
                f"crash={row['crash_rate'] * 100:.2f}% "
                f"timeout={row['timeout_rate'] * 100:.2f}% "
                f"final_goal={row['avg_final_goal_dist']:.2f}m "
                f"timeout_goal={row['timeout_final_goal_dist']:.2f}m"
            )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_csv(rows, output_dir / "round1_results.csv")
    write_csv(trajectory_rows, output_dir / "per_trajectory_results.csv")
    write_pairing_audit(trajectory_rows, output_dir / "pairing_audit.json")
    write_summary(rows, speeds, output_dir / "round1_summary.md")
    save_config(args, existing_specs, output_dir)
    print(f"\nWrote: {output_dir / 'round1_results.csv'}")
    print(f"Wrote: {output_dir / 'per_trajectory_results.csv'}")
    print(f"Wrote: {output_dir / 'pairing_audit.json'}")
    print(f"Wrote: {output_dir / 'round1_summary.md'}")
    print(f"Wrote: {output_dir / 'round1_config.json'}")


if __name__ == "__main__":
    main()
