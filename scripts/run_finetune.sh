#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <ns1e-5|ns1e-3> <smoke|full>" >&2
  exit 2
fi

DATASET_TAG="$1"
MODE="$2"
case "$DATASET_TAG" in
  ns1e-5)
    DATASET_ID="ns2d_fno_1e-5"
    ;;
  ns1e-3)
    DATASET_ID="ns2d_fno_1e-3"
    ;;
  *)
    echo "unknown dataset: $DATASET_TAG" >&2
    exit 2
    ;;
esac
case "$MODE" in
  smoke|full) ;;
  *)
    echo "mode must be smoke or full" >&2
    exit 2
    ;;
esac

ROOT="${MOEPOT_ROOT:-/code/MoEPOT}"
DATA_ROOT="${MOEPOT_DATA_ROOT:-/data/nestor_moepot_ns}"
MODEL_PATH="${MOEPOT_CHECKPOINT:-/model/nestor_moepot_tiny/model_moepot_tiny.pth}"
ENV_DIR="${ENV_DIR:-/data/nestor_env/envs/nestor}"
GPU="${GPU:-0}"
OUT_DIR="$DATA_ROOT/runs/finetune/$DATASET_TAG/$MODE"
SMOKE_ARGS=()
if [[ "$MODE" == "smoke" ]]; then
  SMOKE_ARGS=(--smoke_epochs 5)
fi

mkdir -p "$OUT_DIR"
cd "$ROOT"

test -f "$MODEL_PATH" || {
  echo "missing checkpoint: $MODEL_PATH" >&2
  exit 2
}
test -f "data/large/${DATASET_ID/ns2d_fno_/ns2d_}_train.hdf5" || {
  echo "missing registered training HDF5 for $DATASET_ID" >&2
  exit 2
}

export PATH="$ENV_DIR/bin:$PATH"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"

echo "finetune start: dataset=$DATASET_TAG mode=$MODE time=$(date -Is)"
nvidia-smi

python finetune.py \
  --model MoEPOT \
  --dataset ns2d \
  --resume_path "$MODEL_PATH" \
  --train_paths "$DATASET_ID" \
  --test_paths "$DATASET_ID" \
  --ntrain_list 1000 \
  --data_weights 1 \
  --cls_id 0 \
  --res 128 \
  --modes 32 \
  --patch_size 8 \
  --width 512 \
  --n_layers 4 \
  --n_blocks 4 \
  --mlp_ratio 1 \
  --out_layer_dim 32 \
  --batch_size 20 \
  --epochs 200 \
  --lr 0.001 \
  --lr_method cycle \
  --warmup_epochs 40 \
  --noise_scale 0.0 \
  --T_in 10 \
  --T_ar 1 \
  --T_bundle 1 \
  --n_channels 4 \
  --n_class 6 \
  --normalize 0 \
  --beta1 0.9 \
  --beta2 0.9 \
  --grad_clip 10000.0 \
  --use_writer \
  --comment "${DATASET_TAG}_${MODE}" \
  --log_path "$OUT_DIR" \
  --gpu "$GPU" \
  "${SMOKE_ARGS[@]}"

test -f "$OUT_DIR/model.pth" || {
  echo "training finished without model.pth: $OUT_DIR" >&2
  exit 3
}
echo "finetune finish: dataset=$DATASET_TAG mode=$MODE time=$(date -Is)"
