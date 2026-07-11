# Repository Artifact Policy

## Keep In Git

- Source code needed to train and evaluate in the CUDA differentiable simulator.
- Config files used by the original-style and memdpflight experiments.
- A small set of representative `.pth` checkpoints under `weights/`.
- Documentation explaining evaluation protocol and limitations.

## Exclude From Git

- `validation/`: ROS, AirSim, Flightmare, Unreal, and external validation packages. These are large and environment-specific.
- `LinuxNoEditor/`: packaged simulator binary, too large for a normal source repository.
- `runs/`, `retrain_runs/`, `backup_checkpoints/`: TensorBoard logs, temporary checkpoints, and repeated experimental output.
- `src/build/`, `*.so`, `*.egg-info`: machine-specific compiled artifacts.
- `__pycache__/`, editor folders, temporary media, and generated plots.

## Selected Weights

The selected checkpoints are small enough for normal GitHub storage and are useful for direct reproduction:

- `checkpoint_phase1_done.pth`: best average local CUDA success-rate sweep among the tested improved checkpoints.
- `checkpoint00005.pth`: stable alternative checkpoint.
- `checkpoint_best.pth`: training-run best checkpoint.
- `checkpoint_phase0_done.pth`: curriculum ablation checkpoint.
- `decoder_pretrain.pth`: pretraining artifact for reconstruction/memory modules.
- `curriculum_best.pth`: curriculum state artifact.

If the project later grows beyond GitHub's practical repository size, move checkpoints to GitHub Releases, Hugging Face, Google Drive, or another artifact store, and keep only download instructions in Git.

