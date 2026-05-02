#!/usr/bin/env bash
#SBATCH --job-name=flops
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=infinite
#SBATCH --partition=main
#SBATCH --output=slurms/slurm-%A_%a.out
#SBATCH --account=training
#SBATCH --array=0-17

set -euo pipefail

export OMP_NUM_THREADS=8

ROOT="/data/connor/fmri-fm"
cd $ROOT

# export all env variables
set -a
source .env.medarc.r2
set +a

EXP_NAME="flops"
EXP_DIR="experiments/${EXP_NAME}"

configs=(
    flat_lr1e-3_1
    mni_cortex_lr1e-3_1
    schaefer400_lr3e-4_1
    flat_lr1e-3_2
    mni_cortex_lr1e-3_2
    schaefer400_lr3e-4_2
    flat_lr1e-3_3
    mni_cortex_lr1e-3_3
    schaefer400_lr3e-4_3
)

datasets=(
    aabc_sex/4
    hcpya_task21/32
)

num_datasets=${#datasets[@]}
configid=$(( $SLURM_ARRAY_TASK_ID / $num_datasets ))
datasetid=$(( $SLURM_ARRAY_TASK_ID % $num_datasets ))

config=${configs[configid]}
key=$(echo $config | cut -d / -f 1)
space=$(echo $key | sed 's/\(.*\)_lr.*/\1/')
model="${space}_mae"
ckpt_path="experiments/input_space_v3/output/input_space_v3/${key}/pretrain/checkpoint-last.pth"

dsconfig=${datasets[datasetid]}
dataset=$(echo $dsconfig | cut -d / -f 1)
bs=$(echo $dsconfig | cut -d / -f 2)
overrides="model_kwargs.ckpt_path=${ckpt_path} batch_size=${bs}"

result="${EXP_DIR}/results/flops__${key}__${dataset}.txt"

uv run --no-sync python -m fmri_fm_eval.main_flops \
    $model \
    $dataset \
    --overrides \
    $overrides \
    | tee $result
