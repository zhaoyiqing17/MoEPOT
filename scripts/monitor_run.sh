#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <run-dir> [--follow]" >&2
  exit 2
fi

RUN_DIR="$1"
FOLLOW="${2:-}"
PID_FILE="$RUN_DIR/pid"

show_status() {
  echo "time: $(date -Is)"
  if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "process: RUNNING (pid $(cat "$PID_FILE"))"
  else
    echo "process: STOPPED"
  fi
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader || true
  if [[ -f "$RUN_DIR/zero_shot.log" ]]; then
    echo "--- zero_shot.log"
    tail -n 20 "$RUN_DIR/zero_shot.log"
  fi
  if [[ -f "$RUN_DIR/logs.txt" ]]; then
    echo "--- logs.txt"
    tail -n 20 "$RUN_DIR/logs.txt"
  fi
  if [[ -f "$RUN_DIR/nohup.out" ]]; then
    echo "--- nohup.out"
    tail -n 20 "$RUN_DIR/nohup.out"
  fi
}

if [[ "$FOLLOW" == "--follow" ]]; then
  while true; do
    clear
    show_status
    if [[ -f "$PID_FILE" ]] && ! kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
      echo "run has stopped"
      break
    fi
    sleep 60
  done
else
  show_status
fi
