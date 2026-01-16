#!/bin/bash

set -euo pipefail

ROOT="${HOME}/fmri-fm"
cd $ROOT

# export all env variables
set -a
source .env
set +a

EXP_NAME="augmentation"
EXP_DIR="experiments/${EXP_NAME}"
OUT_DIR="${EXP_DIR}/output"

name="debug"
base_config="${EXP_DIR}/pretrain.yaml"

overrides="tr_scale=0.8 crop_scale=0.5 crop_aspect=0.8 gray_jitter=0.4 gauss_sigma=0.5"
overrides="${overrides} epochs=10 warmup_epochs=10 wandb=false debug=true plot_period=1"

uv run torchrun --standalone --nproc_per_node=1 \
    src/flat_mae/main_pretrain.py \
    --cfg-path "${base_config}" \
    --overrides \
    name="${name}" \
    output_dir="${OUT_DIR}" \
    ${overrides}
