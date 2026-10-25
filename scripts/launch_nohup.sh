#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 zero | $0 finetune <ns1e-5|ns1e-3> <smoke|full>" >&2
  exit 2
fi

ROOT="${MOEPOT_ROOT:-/code/MoEPOT}"
DATA_ROOT="${MOEPOT_DATA_ROOT:-/data/nestor_moepot_ns}"
ACTION="$1"
shift

case "$ACTION" in
  zero)
    SCRIPT="$ROOT/scripts/run_zero_shot.sh"
    ARGS=()
    RUN_DIR="$DATA_ROOT/runs/zero_shot"
    ;;
  finetune)
    if [[ $# -ne 2 ]]; then
      echo "usage: $0 finetune <ns1e-5|ns1e-3> <smoke|full>" >&2
      exit 2
    fi
    SCRIPT="$ROOT/scripts/run_finetune.sh"
    ARGS=("$1" "$2")
    RUN_DIR="$DATA_ROOT/runs/finetune/$1/$2"
    ;;
  *)
    echo "unknown action: $ACTION" >&2
    exit 2
    ;;
esac

mkdir -p "$RUN_DIR"
LOG="$RUN_DIR/nohup.out"
PID_FILE="$RUN_DIR/pid"
nohup bash "$SCRIPT" "${ARGS[@]}" >"$LOG" 2>&1 &
echo $! >"$PID_FILE"
echo "started pid=$(cat "$PID_FILE")"
echo "nohup log: $LOG"
if [[ "$ACTION" == "finetune" ]]; then
  echo "training log: $RUN_DIR/logs.txt"
else
  echo "evaluation log: $RUN_DIR/zero_shot.log"
fi
