# Evaluation Notes

## Local CUDA Benchmark

The local CUDA benchmark uses the repository's differentiable simulator and synthetic depth rendering. It is fast, scriptable on a CUDA server, and suitable for ablations.

Improved checkpoints were evaluated with:

```bash
python memdpflight/eval_benchmark.py \
  --batch_size 32 \
  --num_episodes 15 \
  --target_speed <2|4|6|7|8|9|10> \
  --ground_voxels \
  --random_rotation \
  --cam_angle 20 \
  --fov_x_half_tan 0.82 \
  --recon_members 1 1 0 \
  --recon_steps 12
```

This gives 480 trials per speed.

Observed average success rates across target speeds 2, 4, 6, 7, 8, 9, and 10 m/s:

| Checkpoint | Average success |
| --- | ---: |
| `checkpoint_phase1_done.pth` | 87.77% |
| `checkpoint00005.pth` | 87.23% |
| `checkpoint_phase0_done.pth` | 87.05% |
| `checkpoint_best.pth` | 86.81% |

The original baseline checkpoint previously tested under a related original-style script reached about 78.75% average in the same speed list, but that run used fewer episodes and a different entry script. The result suggests the memory-enhanced branch improved robustness in this local CUDA setup, while it should not be overstated as a paper-level fair comparison.

## Relation To The Paper Benchmark

The high success rates in the original paper's Figure 8 correspond to the Flightmare/Agile high-speed validation benchmark. That benchmark uses a ROS-driven 3D simulator with rendered visual observations and a fixed Agile benchmark scene/protocol.

The lightweight CUDA simulator here is different:

- CUDA benchmark: procedural geometry and depth are computed by project CUDA kernels; useful for training and fast evaluation.
- Flightmare/ROS benchmark: a 3D rendering/simulation stack provides camera images/depth in a more realistic validation environment; useful for fair comparison with Agile and the paper.

Therefore, a fair comparison with the paper should run both the original checkpoint and the improved checkpoint in the same Flightmare/Agile validation environment, with identical scene, camera, target-speed schedule, random seeds where possible, and success/failure definitions.

