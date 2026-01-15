#!/usr/bin/env bash
#SBATCH --job-name=masking
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=infinite
#SBATCH --partition=main
#SBATCH --output=slurms/slurm-%A_%a.out
# #SBATCH --nodelist=n-4
#SBATCH --account=training
#SBATCH --array=0-7

set -euo pipefail

ROOT="${HOME}/fmri-fm"
cd $ROOT

# export all env variables
set -a
source .env
set +a

EXP_NAME="masking"
EXP_DIR="experiments/${EXP_NAME}"
OUT_DIR="${EXP_DIR}/output"

configs=(
    "uniform_mr0.5|mask_ratio=0.5"
    "uniform_mr0.75|mask_ratio=0.75"
    "uniform_mr0.9|mask_ratio=0.9"
    "uniform_mr0.95|mask_ratio=0.95"
    "tube_mr0.5|mask_ratio=0.5 masking=tube"
    "tube_mr0.75|mask_ratio=0.75 masking=tube"
    "tube_mr0.9|mask_ratio=0.9 masking=tube"
    "tube_mr0.95|mask_ratio=0.95 masking=tube"
)

config=${configs[SLURM_ARRAY_TASK_ID]}
name=$(echo $config | cut -d '|' -f 1)
overrides=$(echo $config | cut -d '|' -f 2)

base_config="${EXP_DIR}/pretrain.yaml"
fullname="${EXP_NAME}/${name}/pretrain"
notes="masking ablations $name (${overrides})"

# add small delay between jobs
# bit of hack to try to get wandb to assign different colors
sleep $(( SLURM_ARRAY_TASK_ID * 10 ))

uv run torchrun --standalone --nproc_per_node=1 \
    src/flat_mae/main_pretrain.py \
    --cfg-path "${base_config}" \
    --overrides \
    $overrides \
    name="${fullname}" \
    notes="${notes}" \
    output_dir="${OUT_DIR}"
