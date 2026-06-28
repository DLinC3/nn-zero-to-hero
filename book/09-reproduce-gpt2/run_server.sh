#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/build-nanogpt"

# Server/cloud training template.
# 1. Start from a clean server environment and record GPU, driver, CUDA, and Python versions.
# 2. Create a fresh virtual environment or conda environment.
# 3. Install dependencies, choosing the PyTorch build that matches the server CUDA driver/runtime.
# 4. Run a short smoke test before launching longer runs.
# 5. Keep logs, generated datasets, checkpoints, and experiment outputs out of git.
# 6. For longer runs, use tmux/systemd/slurm as appropriate and write checkpoints frequently.
# 7. Optionally enable wandb for tracking after the local run is stable.

# Example starting point; tune for the actual server before use.
python train_gpt2.py
