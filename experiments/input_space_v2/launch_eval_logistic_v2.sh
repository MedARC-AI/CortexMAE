#!/usr/bin/env bash
#SBATCH --job-name=input_space
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=infinite
#SBATCH --partition=main
#SBATCH --output=slurms/slurm-%A_%a.out
# #SBATCH --nodelist=n-1,n-4
#SBATCH --nodelist=n-1,n-2,n-3,n-4
#SBATCH --account=training
# #SBATCH --array=0-167
#SBATCH --array=168-335

set -euo pipefail

export OMP_NUM_THREADS=8

# ROOT="${HOME}/fmri-fm"
ROOT="/data/connor/fmri-fm"
cd $ROOT

# export all env variables
set -a
source .env
set +a

EXP_NAME="input_space_v2"
EXP_DIR="experiments/${EXP_NAME}"
OUT_DIR="${EXP_DIR}/output"

configs=(
    flat_lr1e-3_1/cls
    flat_lr1e-3_2/cls
    flat_lr1e-3_3/cls
    flat_lr1e-3_4/cls
    flat_lr1e-3_5/cls
    mni_cortex_lr1e-3_1/cls
    mni_cortex_lr1e-3_2/cls
    mni_cortex_lr1e-3_3/cls
    mni_cortex_lr1e-3_4/cls
    mni_cortex_lr1e-3_5/cls
    schaefer400_lr3e-4_1/cls
    schaefer400_lr3e-4_2/cls
    schaefer400_lr3e-4_3/cls
    schaefer400_lr3e-4_4/cls
    schaefer400_lr3e-4_5/cls
    flat_lr1e-3_6/cls
    flat_lr1e-3_7/cls
    flat_lr1e-3_8/cls
    mni_cortex_lr1e-3_6/cls
    mni_cortex_lr1e-3_7/cls
    mni_cortex_lr1e-3_8/cls
    schaefer400_lr3e-4_6/cls
    schaefer400_lr3e-4_7/cls
    schaefer400_lr3e-4_8/cls
    flat_lr1e-3_1/patch
    flat_lr1e-3_2/patch
    flat_lr1e-3_3/patch
    flat_lr1e-3_4/patch
    flat_lr1e-3_5/patch
    mni_cortex_lr1e-3_1/patch
    mni_cortex_lr1e-3_2/patch
    mni_cortex_lr1e-3_3/patch
    mni_cortex_lr1e-3_4/patch
    mni_cortex_lr1e-3_5/patch
    schaefer400_lr3e-4_1/patch
    schaefer400_lr3e-4_2/patch
    schaefer400_lr3e-4_3/patch
    schaefer400_lr3e-4_4/patch
    schaefer400_lr3e-4_5/patch
    flat_lr1e-3_6/patch
    flat_lr1e-3_7/patch
    flat_lr1e-3_8/patch
    mni_cortex_lr1e-3_6/patch
    mni_cortex_lr1e-3_7/patch
    mni_cortex_lr1e-3_8/patch
    schaefer400_lr3e-4_6/patch
    schaefer400_lr3e-4_7/patch
    schaefer400_lr3e-4_8/patch
)

datasets=(
    abide_dx
    adhd200_dx
    adni_ad_vs_cn
    ppmi_dx
    aabc_age
    aabc_sex
    hcpya_rest1lr_gender
)

# 24 models x 7 datasets

num_datasets=${#datasets[@]}
configid=$(( $SLURM_ARRAY_TASK_ID / $num_datasets ))
datasetid=$(( $SLURM_ARRAY_TASK_ID % $num_datasets ))

config=${configs[configid]}
key=$(echo $config | cut -d / -f 1)
space=$(echo $key | sed 's/\(.*\)_lr.*/\1/')
repr=$(echo $config | cut -d / -f 2)
clf="logistic"

model="${space}_mae"
ckpt_path="${OUT_DIR}/${EXP_NAME}/${key}/pretrain/checkpoint-last.pth"
if [[ ! -f $ckpt_path ]]; then
    echo "checkpoint ${ckpt_path} doesn't exist; not running"
    exit
fi

dataset=${datasets[datasetid]}
overrides="model_kwargs.ckpt_path=${ckpt_path} batch_size=2"

notes="input_space ablation v2 $key; eval v2 (${dataset} ${repr} ${clf})"

name="${EXP_NAME}/${key}/eval_v2/${dataset}__${repr}__${clf}"
result="${OUT_DIR}/${name}/eval_table.csv"
if [[ -f $result ]]; then
    echo "result $result exists; skipping"
    exit
fi

uv run --no-sync python -W ignore -m fmri_fm_eval.main_logistic \
    $model \
    $repr \
    $dataset \
    --overrides \
    output_root="${OUT_DIR}" \
    name="${name}" \
    notes="${notes}" \
    $overrides
