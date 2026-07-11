# Mem-DP Agile Flight

Memory-augmented vision-based agile flight built on the differentiable-physics drone training code from **Learning Vision-based Agile Flight via Differentiable Physics**.

This repository keeps the lightweight CUDA differentiable simulation workflow and the `memdpflight` improvement branch. The improvement idea is to add temporal memory / reconstruction-style auxiliary learning inspired by MAVRL, then evaluate whether the policy is more stable under near-original CUDA simulation settings.

## What Is Included

- `src/`: CUDA extension source for the differentiable quadrotor simulator.
- `env_cuda.py`, `model.py`, `main_cuda.py`, `eval_original_like.py`: original-style CUDA training and evaluation entry points used as baselines.
- `main_memflight_cuda.py`, `model_memflight.py`: earlier memory-augmented single-branch experiment.
- `memdpflight/`: main improved branch, including memory model, curriculum training scripts, pretraining, visual evaluation, and benchmark evaluation.
- `configs/`, `memdpflight/configs/`: reproducible command-line argument files.
- `weights/`: selected representative checkpoints and auxiliary artifacts for evaluation.
- `docs/`: notes about artifacts, evaluation protocol, and fair-comparison caveats.

Large simulators and generated files are intentionally excluded: `validation/`, `LinuxNoEditor/`, build folders, TensorBoard logs, caches, videos, and temporary checkpoints.

## Environment

The upstream code was tested with:

- Python 3.11
- PyTorch 2.2.2
- CUDA 11.8
- NVIDIA GPU with a working CUDA toolkit

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Build the CUDA extension:

```bash
pip install -e src
```

Quick import check:

```bash
python src/test.py
```

## Training

Original-style single-agent training:

```bash
python main_cuda.py $(cat configs/single_agent.args)
```

Memory-augmented training:

```bash
cd memdpflight
python mem_main_cuda.py $(cat configs/mem_ultimate.args)
```

The `memdpflight` branch adds:

- temporal memory / recurrent state for partially observable depth inputs;
- reconstruction auxiliary losses over current/past perception features;
- MAVRL-inspired staged training and curriculum scheduling;
- wider camera setting and ground-obstacle evaluation options used in the local CUDA benchmark.

## Evaluation

Improved model, near-original CUDA benchmark style:

```bash
cd memdpflight
python eval_benchmark.py \
  --resume ../weights/checkpoint_phase1_done.pth \
  --batch_size 32 \
  --num_episodes 15 \
  --target_speed 8 \
  --ground_voxels \
  --random_rotation \
  --cam_angle 20 \
  --fov_x_half_tan 0.82 \
  --recon_members 1 1 0 \
  --recon_steps 12
```

Original-style baseline evaluation:

```bash
python eval_original_like.py \
  --resume path/to/original/checkpoint0004.pth \
  --batch_size 32 \
  --num_episodes 15 \
  --target_speed 8 \
  --ground_voxels \
  --random_rotation \
  --cam_angle 20 \
  --fov_x_half_tan 0.82
```

See [docs/evaluation_notes.md](docs/evaluation_notes.md) for the measured local CUDA results and the limits of comparing them with the paper's Flightmare/Agile benchmark.

## Checkpoints

Selected artifacts in `weights/`:

- `checkpoint_phase1_done.pth`: strongest overall checkpoint in the local CUDA sweep.
- `checkpoint00005.pth`: another stable high-performing checkpoint.
- `checkpoint_best.pth`: best checkpoint from the training run naming convention.
- `checkpoint_phase0_done.pth`: phase-0 curriculum checkpoint for ablation.
- `decoder_pretrain.pth`: auxiliary decoder pretraining artifact.
- `curriculum_best.pth`: small curriculum state artifact.

## Fair-Comparison Note

The paper's Figure 8 success-rate result is from the Flightmare/Agile high-speed benchmark, not from the lightweight CUDA training/evaluation scripts in this repository. The CUDA benchmark here is useful for controlled ablation and fast iteration, but it is not a substitute for the same ROS + Flightmare validation environment used for paper-level comparison.

## Acknowledgements

This project is based on the released training code for:

```bibtex
@article{zhang2025learning,
  title={Learning vision-based agile flight via differentiable physics},
  author={Zhang, Yuang and Hu, Yu and Song, Yunlong and Zou, Danping and Lin, Weiyao},
  journal={Nature Machine Intelligence},
  pages={1--13},
  year={2025},
  publisher={Nature Publishing Group}
}
```

The memory/curriculum exploration is inspired by MAVRL:

```bibtex
@inproceedings{yu2024mavrl,
  title={MAVRL: Learn to Fly in Cluttered Environments with Varying Speed},
  year={2024}
}
```

