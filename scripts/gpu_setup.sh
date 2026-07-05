#!/bin/bash
# One-shot setup for a fresh Ubuntu + NVIDIA GPU box (e.g. Vast.ai / RunPod /
# Lambda). Installs Go, a Python venv with CUDA PyTorch, and builds the Go
# self-play binary. Run from the repo root:  bash scripts/gpu_setup.sh
set -euo pipefail

GO_VERSION=${GO_VERSION:-1.23.4}
CUDA_TAG=${CUDA_TAG:-cu124}     # match the box's CUDA (cu121/cu124/...); nvidia-smi to check

echo "=== system deps ==="
if command -v apt-get >/dev/null; then
  sudo apt-get update -y
  sudo apt-get install -y python3 python3-venv python3-pip build-essential wget git
fi

echo "=== Go $GO_VERSION ==="
if ! command -v go >/dev/null; then
  wget -q "https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz"
  sudo rm -rf /usr/local/go && sudo tar -C /usr/local -xzf "go${GO_VERSION}.linux-amd64.tar.gz"
  export PATH=$PATH:/usr/local/go/bin
  echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc
fi
go version

echo "=== python venv + CUDA torch ($CUDA_TAG) ==="
python3 -m venv .venv
. .venv/bin/activate
pip install -q --upgrade pip
pip install -q torch --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
pip install -q numpy tqdm pytest

echo "=== build go self-play binary ==="
(cd go && go build ./...)

echo "=== sanity checks ==="
.venv/bin/python -c "import torch; print('cuda available:', torch.cuda.is_available(), '| device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
.venv/bin/python -m pytest tests/ -q 2>&1 | tail -1
(cd go && go test ./engine/ 2>&1 | tail -1)

echo "=== setup complete. Next: docs/GPU_RUNBOOK.md ==="
