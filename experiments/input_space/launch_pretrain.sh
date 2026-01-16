#!/usr/bin/env bash
#SBATCH --job-name=input_space
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=infinite
#SBATCH --partition=main
#SBATCH --nodelist=n-2,n-3,n-4
#SBATCH --account=training
#SBATCH --array=2

set -euo pipefail

ROOT="${HOME}/fmri-fm"
cd $ROOT

# export all env variables
set -a
source .env
set +a

EXP_NAME="input_space"
EXP_DIR="experiments/${EXP_NAME}"
OUT_DIR="${EXP_DIR}/output"

spaces=(
    schaefer400
    flat
    mni_cortex
)
space=${spaces[SLURM_ARRAY_TASK_ID]}

config="${EXP_DIR}/pretrain.yaml"

name="${EXP_NAME}/${space}/pretrain"
notes="input space ablation (input_space=${space})"
overrides="input_space=${space}"

# name="${EXP_NAME}/${space}_default_init/pretrain"
# notes="input space ablation (default head init) (input_space=${space})"
# overrides="input_space=${space} model_kwargs.head_init_scale=null"

# name="${EXP_NAME}/${space}_mds/pretrain"
# notes="input space ablation (default head init, mask drop scale) (input_space=${space})"
# overrides="input_space=${space} model_kwargs.head_init_scale=null model_kwargs.mask_drop_scale=true"

uv run python \
    src/flat_mae/main_pretrain.py \
    --cfg-path "${config}" \
    --overrides \
    $overrides \
    name="${name}" \
    notes="${notes}" \
    output_dir="${OUT_DIR}"
