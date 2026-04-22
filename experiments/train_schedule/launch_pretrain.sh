#!/usr/bin/env bash
#SBATCH --job-name=train_schedule
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=infinite
#SBATCH --partition=main
#SBATCH --output=slurms/slurm-%A_%a.out
#SBATCH --account=training
#SBATCH --qos=low
#SBATCH --array=8-9
#SBATCH --requeue

set -euo pipefail

ROOT="/data/connor/fmri-fm"
cd $ROOT

# export all env variables
set -a
source .env
set +a

EXP_NAME="train_schedule"
EXP_DIR="experiments/${EXP_NAME}"
OUT_DIR="${EXP_DIR}/output"

configs=(
    "ep100_1|epochs=100 seed=6101"
    "ep100_2|epochs=100 seed=6102"
    "ep50_1|epochs=50 seed=6101"
    "ep50_2|epochs=50 seed=6102"
    "ep25_1|epochs=25 seed=6101"
    "ep25_2|epochs=25 seed=6102"
    "ep200_1|epochs=200 seed=6101"
    "ep200_2|epochs=200 seed=6102"
    "ep10_1|epochs=10 seed=6101"
    "ep10_2|epochs=10 seed=6102"
)

config=${configs[SLURM_ARRAY_TASK_ID]}
name=$(echo $config | cut -d '|' -f 1)
overrides=$(echo $config | cut -d '|' -f 2)

base_config="${EXP_DIR}/pretrain.yaml"
fullname="${EXP_NAME}/${name}/pretrain"
notes="train schedule ablation $name (${overrides})"

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
