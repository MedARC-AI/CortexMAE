#!/usr/bin/env bash
#SBATCH --job-name=model_scaling
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=infinite
#SBATCH --partition=main
#SBATCH --output=slurms/slurm-%A_%a.out
#SBATCH --nodelist=n-2,n-3,n-4
#SBATCH --account=training
# #SBATCH --array=0-8
#SBATCH --array=9-12

set -euo pipefail

# ROOT="${HOME}/fmri-fm"
ROOT="/data/connor/fmri-fm"
cd $ROOT

# export all env variables
set -a
source .env
set +a

EXP_NAME="model_scaling"
EXP_DIR="experiments/${EXP_NAME}"
OUT_DIR="${EXP_DIR}/output"

# for "wds" models we apply the weight decay scaling rule from nanochat
# weight_decay = 0.05 * (12 / depth) ** 2
# seems counterintuitive to use smaller weight decay for bigger models, but let's see
# https://github.com/karpathy/nanochat/blob/63bb5831e27ec4ad5f7493412cf16f3aa2a35877/scripts/base_train.py#L152
configs=(
    "d3|model=mae_vit_d3"
    "d6|model=mae_vit_d6"
    "d9|model=mae_vit_d9"
    "d15|model=mae_vit_d15"
    "d3_wds|model=mae_vit_d3 weight_decay=0.8"
    "d6_wds|model=mae_vit_d6 weight_decay=0.2"
    "d9_wds|model=mae_vit_d9 weight_decay=0.088"
    "d15_wds|model=mae_vit_d15 weight_decay=0.032"
    "d12|model=mae_vit_d12"
    "d3_2|model=mae_vit_d3 seed=4129"
    "d6_2|model=mae_vit_d6 seed=4129"
    "d9_2|model=mae_vit_d9 seed=4129"
    "d15_2|model=mae_vit_d15 seed=4129"
)

config=${configs[SLURM_ARRAY_TASK_ID]}
name=$(echo $config | cut -d '|' -f 1)
overrides=$(echo $config | cut -d '|' -f 2)

base_config="${EXP_DIR}/pretrain.yaml"
fullname="${EXP_NAME}/${name}/pretrain"
notes="model scaling experiment $name (${overrides})"

# add small delay between jobs
# bit of hack to try to get wandb to assign different colors
sleep $(( SLURM_ARRAY_TASK_ID * 10 ))

# for debugging
# overrides="${overrides} debug=true wandb=false"

uv run --no-sync torchrun --standalone --nproc_per_node=1 \
    src/flat_mae/main_pretrain.py \
    --cfg-path "${base_config}" \
    --overrides \
    $overrides \
    name="${fullname}" \
    notes="${notes}" \
    output_dir="${OUT_DIR}"
