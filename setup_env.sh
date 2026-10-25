#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="${ENV_DIR:-/data/nestor_env/envs/nestor}"
BASE_PYTHON="${BASE_PYTHON:-python3}"

mkdir -p "$(dirname "$ENV_DIR")"
cd "$ROOT"

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  echo "[env] creating $ENV_DIR using the image Python"
  "$BASE_PYTHON" -m venv --system-site-packages "$ENV_DIR"
fi

PY="$ENV_DIR/bin/python"
export PATH="$ENV_DIR/bin:$PATH"

"$PY" - <<'PY'
import sys
if sys.version_info[:2] != (3, 10):
    raise SystemExit(f"expected Python 3.10 image, got {sys.version.split()[0]}")
PY

if ! "$PY" -c 'import torch, torchvision; assert torch.__version__ >= "2.0"' 2>/dev/null; then
  echo "[env] image has no compatible PyTorch; installing PyTorch 2.1.2 CUDA 11.8"
  "$PY" -m pip install \
    --index-url https://download.pytorch.org/whl/cu118 \
    torch==2.1.2 torchvision==0.16.2
fi

"$PY" -m pip install --upgrade "pip<25" setuptools wheel
"$PY" -m pip install -r "$ROOT/requirements-moepot.txt"

export ENV_DIR
"$PY" scripts/check_env.py
printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$ENV_DIR/.moepot-ready"
echo "Environment ready: $ENV_DIR"
