#!/usr/bin/env bash
#SBATCH --job-name=input_space
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=infinite
#SBATCH --partition=main
#SBATCH --output=slurms/slurm-%A_%a.out
#SBATCH --nodelist=n-1,n-2
#SBATCH --account=training
#SBATCH --array=0-11

set -euo pipefail

ROOT="${HOME}/fmri-fm"
cd $ROOT

# export all env variables
set -a
source .env
set +a

EXP_NAME="input_space_v2"
EXP_DIR="experiments/${EXP_NAME}"
OUT_DIR="${EXP_DIR}/output"

configs=(
    "schaefer400_lr3e-4_1|input_space=schaefer400 base_lr=3e-4 seed=5401"
    "schaefer400_lr3e-4_2|input_space=schaefer400 base_lr=3e-4 seed=5402"
    "schaefer400_lr1e-3_1|input_space=schaefer400 base_lr=1e-3 seed=5401"
    "schaefer400_lr1e-3_2|input_space=schaefer400 base_lr=1e-3 seed=5402"
    "schaefer400_lr3e-3_1|input_space=schaefer400 base_lr=3e-3 seed=5401"
    "schaefer400_lr3e-3_2|input_space=schaefer400 base_lr=3e-3 seed=5402"
    "mni_cortex_lr3e-4_1|input_space=mni_cortex base_lr=3e-4 seed=5401"
    "mni_cortex_lr3e-4_2|input_space=mni_cortex base_lr=3e-4 seed=5402"
    "mni_cortex_lr1e-3_1|input_space=mni_cortex base_lr=1e-3 seed=5401"
    "mni_cortex_lr1e-3_2|input_space=mni_cortex base_lr=1e-3 seed=5402"
    "mni_cortex_lr3e-3_1|input_space=mni_cortex base_lr=3e-3 seed=5401"
    "mni_cortex_lr3e-3_2|input_space=mni_cortex base_lr=3e-3 seed=5402"
)

config=${configs[SLURM_ARRAY_TASK_ID]}
name=$(echo $config | cut -d '|' -f 1)
overrides=$(echo $config | cut -d '|' -f 2)

base_config="${EXP_DIR}/pretrain.yaml"
fullname="${EXP_NAME}/${name}/pretrain"
notes="input_space ablation v2 $name (${overrides})"

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
