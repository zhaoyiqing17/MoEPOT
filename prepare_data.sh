#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="${ENV_DIR:-/data/nestor_env/envs/nestor}"
PY="${ENV_DIR}/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "missing environment: $ENV_DIR; run bash $ROOT/setup_env.sh first" >&2
  exit 2
fi

cd "$ROOT"
export PATH="$ENV_DIR/bin:$PATH"
exec "$PY" scripts/prepare_assets.py "$@"
