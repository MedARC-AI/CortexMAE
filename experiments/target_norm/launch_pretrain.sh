#!/usr/bin/env bash
#SBATCH --job-name=target_norm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=infinite
#SBATCH --partition=main
#SBATCH --output=slurms/slurm-%A_%a.out
#SBATCH --nodelist=n-3
#SBATCH --account=training
#SBATCH --array=0,1,2

set -euo pipefail

ROOT="${HOME}/fmri-fm"
cd $ROOT

# export all env variables
set -a
source .env
set +a

EXP_NAME="target_norm"
EXP_DIR="experiments/${EXP_NAME}"
OUT_DIR="${EXP_DIR}/output"

configs=(
    "patch|model_kwargs.target_norm=patch"
    "pca_nc2|model_kwargs.target_norm=pca model_kwargs.pca_norm_nc=2"
    "pca_nc8|model_kwargs.target_norm=pca model_kwargs.pca_norm_nc=8"
)

config=${configs[SLURM_ARRAY_TASK_ID]}
name=$(echo $config | cut -d '|' -f 1)
overrides=$(echo $config | cut -d '|' -f 2)

base_config="${EXP_DIR}/pretrain.yaml"
fullname="${EXP_NAME}/${name}/pretrain"
notes="target_norm ablations $name (${overrides})"

# add small delay between jobs
# bit of hack to try to get wandb to assign different colors
sleep $(( SLURM_ARRAY_TASK_ID * 10 ))

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
