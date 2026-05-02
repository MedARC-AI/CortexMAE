#!/usr/bin/env bash
#SBATCH --job-name=rest_v_task
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=infinite
#SBATCH --partition=main
#SBATCH --output=slurms/slurm-%A_%a.out
#SBATCH --account=training
#SBATCH --qos=low
#SBATCH --array=0-7
#SBATCH --requeue

set -euo pipefail

ROOT="/data/connor/fmri-fm"
cd $ROOT

# export all env variables
set -a
source .env
set +a

EXP_NAME="rest_v_task"
EXP_DIR="experiments/${EXP_NAME}"
OUT_DIR="${EXP_DIR}/output"

configs=(
    "rest_ep50_1|epochs=50 seed=6101|rest"
    "rest_ep50_2|epochs=50 seed=6102|rest"
    "task_ep50_1|epochs=50 seed=6101|task"
    "task_ep50_2|epochs=50 seed=6102|task"
    "rest_ep100_1|epochs=100 seed=6101|rest"
    "rest_ep100_2|epochs=100 seed=6102|rest"
    "task_ep100_1|epochs=100 seed=6101|task"
    "task_ep100_2|epochs=100 seed=6102|task"
)

config=${configs[SLURM_ARRAY_TASK_ID]}
name=$(echo $config | cut -d '|' -f 1)
overrides=$(echo $config | cut -d '|' -f 2)
subset=$(echo $config | cut -d '|' -f 3)

base_config="${EXP_DIR}/pretrain_${subset}.yaml"
fullname="${EXP_NAME}/${name}/pretrain"
notes="rest_v_task ablation $name (${overrides})"

# add small delay between jobs
# bit of hack to try to get wandb to assign different colors
sleep $(( SLURM_ARRAY_TASK_ID * 5 ))

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
