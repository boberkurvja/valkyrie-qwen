#!/bin/bash
# Project Valkyrie — Environment Setup
# Hardware: RTX 3090 (24GB), AMD EPYC 32-Core, 128GB RAM
# CUDA: 12.9 (driver), PyTorch CUDA 12.8 wheels
set -e

echo "=== Project Valkyrie: Environment Setup ==="

# 1. Install PyTorch 2.x with CUDA 12.8
echo "[1/6] Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 2. Install causal-conv1d from source (must be before mamba-ssm)
echo "[2/6] Installing causal-conv1d from source..."


# 3. Install mamba-ssm from local source
echo "[3/6] Installing mamba-ssm from source..."
cd /root/model/main

# 4. Install core ML dependencies
echo "[4/6] Installing transformers, datasets, trl..."
pip install transformers datasets trl accelerate peft

# 5. Install optimizer and utilities
echo "[5/6] Installing adam-atan2-pytorch and utilities..."
pip install adam-atan2-pytorch einops wandb tqdm pydantic omegaconf hydra-core

# 6. Install HRM dependencies
echo "[6/6] Installing HRM dependencies..."
pip install -r /root/model/HRM/requirements.txt

echo "=== Environment setup complete ==="
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}')"
