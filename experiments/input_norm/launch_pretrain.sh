#!/usr/bin/env bash
#SBATCH --job-name=input_norm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=infinite
#SBATCH --partition=main
#SBATCH --output=slurms/slurm-%A_%a.out
#SBATCH --nodelist=n-1,n-2,n-3,n-4
#SBATCH --account=training
# #SBATCH --array=0-4
#SBATCH --array=5-6

set -euo pipefail

ROOT="${HOME}/fmri-fm"
cd $ROOT

# export all env variables
set -a
source .env
set +a

EXP_NAME="input_norm"
EXP_DIR="experiments/${EXP_NAME}"
OUT_DIR="${EXP_DIR}/output"

configs=(
    "coord_frame|no_coord_normalize=false normalize=frame"
    "coord_global|no_coord_normalize=false normalize=global"
    "coord_none|no_coord_normalize=false normalize=null"
    "nocoord_frame|no_coord_normalize=true normalize=frame"
    "nocoord_global|no_coord_normalize=true normalize=global"
    "nocoord_frame_mni|no_coord_normalize=true normalize=frame input_space=mni_cortex"
    "nocoord_global_mni|no_coord_normalize=true normalize=global input_space=mni_cortex"
)

config=${configs[SLURM_ARRAY_TASK_ID]}
name=$(echo $config | cut -d '|' -f 1)
overrides=$(echo $config | cut -d '|' -f 2)

base_config="${EXP_DIR}/pretrain.yaml"
fullname="${EXP_NAME}/${name}/pretrain"
notes="input_norm ablations $name (${overrides})"

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
