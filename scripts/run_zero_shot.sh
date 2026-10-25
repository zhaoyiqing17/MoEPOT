#!/usr/bin/env bash
set -euo pipefail

ROOT="${MOEPOT_ROOT:-/code/MoEPOT}"
DATA_ROOT="${MOEPOT_DATA_ROOT:-/data/nestor_moepot_ns}"
MODEL_PATH="${MOEPOT_CHECKPOINT:-/model/nestor_moepot_tiny/model_moepot_tiny.pth}"
ENV_DIR="${ENV_DIR:-/data/nestor_env/envs/nestor}"
GPU="${GPU:-0}"
OUT_DIR="$DATA_ROOT/runs/zero_shot"
LOG="$OUT_DIR/zero_shot.log"

mkdir -p "$OUT_DIR"
cd "$ROOT"

test -f "$MODEL_PATH" || {
  echo "missing checkpoint: $MODEL_PATH" >&2
  exit 2
}
test -f data/large/ns2d_1e-5_test.hdf5 || {
  echo "missing registered NS 1e-5 test HDF5" >&2
  exit 2
}
test -f data/large/ns2d_1e-3_test.hdf5 || {
  echo "missing registered NS 1e-3 test HDF5" >&2
  exit 2
}

export PATH="$ENV_DIR/bin:$PATH"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"

echo "zero-shot start: $(date -Is)" | tee "$LOG"
nvidia-smi | tee -a "$LOG"

python evaluate.py \
  --model MoEPOT \
  --dataset ns2d \
  --resume_path "$MODEL_PATH" \
  --train_paths ns2d_fno_1e-5 \
  --test_paths ns2d_fno_1e-5 ns2d_fno_1e-3 \
  --ntrain_list 1000 \
  --data_weights 1 \
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
  --warmup_epochs 40 \
  --noise_scale 0.0 \
  --T_in 10 \
  --T_ar 1 \
  --T_bundle 1 \
  --n_channels 4 \
  --n_class 6 \
  --normalize 0 \
  --gpu "$GPU" 2>&1 | tee -a "$LOG"

echo "zero-shot finish: $(date -Is)" | tee -a "$LOG"
