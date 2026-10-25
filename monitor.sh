#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${MOEPOT_DATA_ROOT:-/data/nestor_moepot_ns}"
STATUS_FILE="$DATA_ROOT/runs/queue/status.json"
PID_FILE="$DATA_ROOT/runs/queue/pid"

show_status() {
  echo "time: $(date -Is)"
  if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "process: RUNNING (pid $(cat "$PID_FILE"))"
  else
    echo "process: STOPPED"
  fi
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader || true
  if [[ -f "$STATUS_FILE" ]]; then
    cat "$STATUS_FILE"
  fi
  for dataset in ns1e-5 ns1e-3; do
    log="$DATA_ROOT/runs/$dataset/ft200/training.log"
    if [[ -f "$log" ]]; then
      echo "--- $dataset"
      tail -n 5 "$log"
    fi
  done
}

if [[ "${1:-}" == "--follow" ]]; then
  while true; do
    clear
    show_status
    if [[ -f "$PID_FILE" ]] && ! kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
      break
    fi
    sleep 60
  done
else
  show_status
fi
