#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="${ENV_DIR:-/data/nestor_env/envs/nestor}"
DATA_ROOT="${MOEPOT_DATA_ROOT:-/data/nestor_moepot_ns}"
QUEUE_DIR="$DATA_ROOT/runs/queue"
mkdir -p "$QUEUE_DIR"
echo "$$" > "$QUEUE_DIR/pid"

finish_status() {
  local state="$1"
  local error="${2:-}"
  "$ENV_DIR/bin/python" - "$QUEUE_DIR/status.json" "$state" "$error" <<'PY'
import json, sys, time
path, state, error = sys.argv[1:]
payload = {"state": state, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
if error:
    payload["error"] = error
open(path, "w").write(json.dumps(payload, indent=2))
PY
}

finish_status running
trap 'finish_status failed "interrupted or command failed"; exit 1' ERR INT TERM

for dataset in ns1e-5 ns1e-3; do
  echo "===== $dataset FT200 ====="
  bash "$ROOT/run_moepot.sh" --dataset "$dataset" --epochs 200
  state="$("$ENV_DIR/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["state"])' "$DATA_ROOT/runs/$dataset/ft200/status.json")"
  if [[ "$state" != "finished" ]]; then
    echo "$dataset did not finish; refusing to start the next dataset" >&2
    exit 4
  fi
done

trap - ERR INT TERM
finish_status finished
echo "ALL_FT200_FINISHED"
