#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="${ENV_DIR:-/data/nestor_env/envs/nestor}"
DATA_ROOT="${MOEPOT_DATA_ROOT:-/data/nestor_moepot_ns}"
CHECKPOINT="${MOEPOT_CHECKPOINT:-/model/nestor_moepot_tiny/model_moepot_tiny.pth}"
DATASET="all"
EPOCHS=200
SMOKE_EPOCHS=0
CHECK_ONLY=0
EVAL_ONLY=0
RESUME_TRAINING=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check-only) CHECK_ONLY=1; shift ;;
    --eval-only) EVAL_ONLY=1; shift ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --smoke-epochs) SMOKE_EPOCHS="$2"; shift 2 ;;
    --resume-training) RESUME_TRAINING="$2"; shift 2 ;;
    -h|--help)
      echo "usage: $0 [--check-only] [--eval-only] [--dataset all|ns1e-5|ns1e-3] [--epochs 200] [--smoke-epochs 5] [--resume-training PATH]"
      exit 0
      ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

case "$DATASET" in
  all|ns1e-5|ns1e-3) ;;
  *) echo "--dataset must be all, ns1e-5 or ns1e-3" >&2; exit 2 ;;
esac

PY="$ENV_DIR/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "missing environment: $ENV_DIR; run bash $ROOT/setup_env.sh first" >&2
  exit 2
fi
cd "$ROOT"
export PATH="$ENV_DIR/bin:$PATH"
export PYTHONUNBUFFERED=1

if [[ "$CHECK_ONLY" == 1 ]]; then
  "$PY" scripts/check_env.py
  "$PY" check_repro.py --checkpoint "$CHECKPOINT"
  exit 0
fi

if [[ "$DATASET" == "all" && -n "$RESUME_TRAINING" ]]; then
  echo "--resume-training requires one explicit --dataset; resume each dataset separately" >&2
  exit 2
fi

if [[ "$EVAL_ONLY" == 1 ]]; then
  RUN_DIR="$DATA_ROOT/runs/zero_shot"
  mkdir -p "$RUN_DIR"
  "$PY" evaluate.py \
    --model MoEPOT --dataset ns2d \
    --resume_path "$CHECKPOINT" \
    --train_paths ns2d_fno_1e-5 \
    --test_paths ns2d_fno_1e-5 ns2d_fno_1e-3 \
    --ntrain_list 1000 --data_weights 1 \
    --res 128 --modes 32 --patch_size 8 \
    --width 512 --n_layers 4 --n_blocks 4 --mlp_ratio 1 --out_layer_dim 32 \
    --batch_size 20 --epochs 200 --lr 0.001 --warmup_epochs 40 \
    --noise_scale 0.0 --T_in 10 --T_ar 1 --T_bundle 1 \
    --n_channels 4 --n_class 6 --normalize 0 --gpu 0 \
    --metrics_json "$RUN_DIR/pretrained_eval.json" \
    2>&1 | tee "$RUN_DIR/zero_shot.log"
  exit 0
fi

run_one() {
  local tag="$1"
  local dataset_id mode run_dir expected_rows
  case "$tag" in
    ns1e-5) dataset_id="ns2d_fno_1e-5" ;;
    ns1e-3) dataset_id="ns2d_fno_1e-3" ;;
  esac
  if [[ "$SMOKE_EPOCHS" -gt 0 ]]; then
    mode="smoke"
    expected_rows="$SMOKE_EPOCHS"
  else
    mode="ft200"
    expected_rows="$EPOCHS"
  fi
  run_dir="$DATA_ROOT/runs/$tag/$mode"
  mkdir -p "$run_dir"
  if [[ -e "$run_dir/status.json" && -z "$RESUME_TRAINING" ]]; then
    echo "run directory already contains a previous run: $run_dir" >&2
    echo "resume with --resume-training $run_dir/latest.pth or use a clean working dataset" >&2
    return 3
  fi
  echo "starting dataset=$tag mode=$mode epochs=$EPOCHS smoke_epochs=$SMOKE_EPOCHS"
  "$PY" finetune.py \
    --model MoEPOT --dataset ns2d \
    --resume_path "$CHECKPOINT" \
    --resume_training "$RESUME_TRAINING" \
    --train_paths "$dataset_id" --test_paths "$dataset_id" \
    --ntrain_list 1000 --data_weights 1 --cls_id 0 \
    --res 128 --modes 32 --patch_size 8 \
    --width 512 --n_layers 4 --n_blocks 4 --mlp_ratio 1 --out_layer_dim 32 \
    --batch_size 20 --epochs "$EPOCHS" --lr 0.001 --lr_method cycle \
    --warmup_epochs 40 --noise_scale 0.0 --T_in 10 --T_ar 1 --T_bundle 1 \
    --n_channels 4 --n_class 6 --normalize 0 --beta1 0.9 --beta2 0.9 \
    --grad_clip 10000.0 --use_writer \
    --smoke_epochs "$SMOKE_EPOCHS" \
    --comment "${tag}_${mode}" --log_path "$run_dir" --gpu 0 \
    2>&1 | tee "$run_dir/training.log"
  "$PY" scripts/validate_run.py --run-dir "$run_dir" --expected-rows "$expected_rows"
}

if [[ "$DATASET" == "all" ]]; then
  run_one ns1e-5
  run_one ns1e-3
else
  run_one "$DATASET"
fi
