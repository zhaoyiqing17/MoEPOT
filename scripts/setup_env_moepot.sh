#!/usr/bin/env bash
set -euo pipefail

ENV_DIR="${ENV_DIR:-/data/nestor_env/envs/nestor}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p "$(dirname "$ENV_DIR")"
if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  echo "[env] creating virtual environment at $ENV_DIR"
  "$PYTHON_BIN" -m venv "$ENV_DIR"
else
  echo "[env] reusing $ENV_DIR"
fi

PY="$ENV_DIR/bin/python"
"$PY" - <<'PY'
import sys
if not ((3, 8) <= sys.version_info[:2] <= (3, 11)):
    raise SystemExit(
        f"Python 3.8-3.11 is required for the pinned PyTorch 2.0 environment; got {sys.version}"
    )
print("[env] python", sys.version.split()[0])
PY

"$PY" -m pip install --upgrade "pip<25" setuptools wheel

if ! "$PY" -c 'import torch; assert torch.__version__.startswith("2.0")' 2>/dev/null; then
  echo "[env] installing PyTorch 2.0.0 CUDA 11.7 wheels"
  "$PY" -m pip install \
    --index-url https://download.pytorch.org/whl/cu117 \
    torch==2.0.0 torchvision==0.15.0
fi

"$PY" -m pip install -r "$ROOT/requirements-moepot.txt"
cd "$ROOT"
"$PY" scripts/check_env.py
printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$ENV_DIR/.moepot-ready"

echo "[env] ready: $PY"
