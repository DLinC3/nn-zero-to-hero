#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/build-nanogpt"

# Local RTX 5090 laptop path.
# Start with smoke tests / small runs first.
# Before training, verify the active environment sees CUDA and PyTorch correctly:
#   python - <<'PY'
#   import torch
#   print(torch.__version__)
#   print(torch.cuda.is_available())
#   print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')
#   PY
# Adjust batch size, gradient accumulation, dtype, and compile settings after verifying CUDA + PyTorch.
python train_gpt2.py
