#!/usr/bin/env bash
#SBATCH --job-name=patch_size
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=infinite
#SBATCH --partition=main
#SBATCH --output=slurms/slurm-%A_%a.out
#SBATCH --nodelist=n-1,n-2,n-3,n-4
#SBATCH --account=training
#SBATCH --array=0-2

set -euo pipefail

# ROOT="${HOME}/fmri-fm"
ROOT="/data/connor/fmri-fm"
cd $ROOT

# export all env variables
set -a
source .env
set +a

EXP_NAME="patch_size"
EXP_DIR="experiments/${EXP_NAME}"
OUT_DIR="${EXP_DIR}/output"

configs=(
    "patch8|patch_size=8 mask_patch_size=16"
    "patch8_mps8|patch_size=8"
    "patch8_mps8_2|patch_size=8 checkpoint_period=5"
)

config=${configs[SLURM_ARRAY_TASK_ID]}
name=$(echo $config | cut -d '|' -f 1)
overrides=$(echo $config | cut -d '|' -f 2)

base_config="${EXP_DIR}/pretrain.yaml"
fullname="${EXP_NAME}/${name}/pretrain"
notes="patch_size ablations $name (${overrides})"

# add small delay between jobs
# bit of hack to try to get wandb to assign different colors
# sleep $(( SLURM_ARRAY_TASK_ID * 10 ))

# for debugging
# overrides="${overrides} debug=true wandb=false"

uv run torchrun --standalone --nproc_per_node=1 \
    src/flat_mae/main_pretrain.py \
    --cfg-path "${base_config}" \
    --overrides \
    $overrides \
    name="${fullname}" \
    notes="${notes}" \
    output_dir="${OUT_DIR}"
