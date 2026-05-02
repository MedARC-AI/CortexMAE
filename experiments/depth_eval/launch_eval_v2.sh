#!/usr/bin/env bash
#SBATCH --job-name=depth_eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=infinite
#SBATCH --partition=main
#SBATCH --output=slurms/slurm-%A_%a.out
#SBATCH --account=training
#SBATCH --array=0-13%2

set -euo pipefail

export OMP_NUM_THREADS=8

ROOT="/data/connor/fmri-fm"
cd $ROOT

# export all env variables
set -a
source .env
set +a

EXP_NAME="depth_eval"
EXP_DIR="experiments/${EXP_NAME}"
OUT_DIR="${EXP_DIR}/output"

configs=(
    "blocks12/patch/attn/null"
    "blocks10/patch/attn/10"
    "blocks8/patch/attn/8"
    "blocks6/patch/attn/6"
    "blocks4/patch/attn/4"
    "blocks2/patch/attn/2"
    "blocks0/patch/attn/0"
)

datasets=(
    hcpya_task21
    nsd_cococlip
)
batch_sizes=(
    64
    64
)

num_datasets=${#datasets[@]}
configid=$(( $SLURM_ARRAY_TASK_ID / $num_datasets ))
datasetid=$(( $SLURM_ARRAY_TASK_ID % $num_datasets ))

config=${configs[configid]}
key=$(echo $config | cut -d / -f 1)
repr=$(echo $config | cut -d / -f 2)
clf=$(echo $config | cut -d / -f 3)
blocks=$(echo $config | cut -d / -f 4)

model="flat_mae"
ckpt_path="experiments/input_space_v3/output/input_space_v3/flat_lr1e-3_1/pretrain/checkpoint-last.pth"

if [[ ! -f $ckpt_path ]]; then
    echo "checkpoint ${ckpt_path} doesn't exist; not running"
    exit
fi

dataset=${datasets[datasetid]}
bs=${batch_sizes[datasetid]}
overrides="model_kwargs.ckpt_path=${ckpt_path} model_kwargs.keep_blocks=${blocks} batch_size=${bs} accum_iter=2"

notes="depth eval ablation $key; eval v2 (${dataset} ${repr} ${clf})"

name="${EXP_NAME}/${key}/eval_v2/${dataset}__${repr}__${clf}"
result="${OUT_DIR}/${name}/eval_table.csv"
if [[ -f $result ]]; then
    echo "result $result exists; skipping"
    exit
fi

# add small delay between jobs
# sleep $(( SLURM_ARRAY_TASK_ID * 10 ))

uv run --no-sync python -m fmri_fm_eval.main_probe \
    $model \
    $repr \
    $clf \
    $dataset \
    --overrides \
    output_root="${OUT_DIR}" \
    name="${name}" \
    notes="${notes}" \
    $overrides
