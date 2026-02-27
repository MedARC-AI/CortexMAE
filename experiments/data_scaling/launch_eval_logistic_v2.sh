#!/usr/bin/env bash
#SBATCH --job-name=data_scaling
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=infinite
#SBATCH --partition=main
#SBATCH --output=slurms/slurm-%A_%a.out
# #SBATCH --nodelist=n-1,n-4
#SBATCH --nodelist=n-1,n-2,n-3,n-4
#SBATCH --account=training
# #SBATCH --array=0-69
#SBATCH --array=70-139

set -euo pipefail

export OMP_NUM_THREADS=8

# ROOT="${HOME}/fmri-fm"
ROOT="/data/connor/fmri-fm"
cd $ROOT

# export all env variables
set -a
source .env
set +a

EXP_NAME="data_scaling"
EXP_DIR="experiments/${EXP_NAME}"
OUT_DIR="${EXP_DIR}/output"

configs=(
    n100_1/cls
    n200_1/cls
    n400_1/cls
    n800_1/cls
    n1600_1/cls
    n100_2/cls
    n200_2/cls
    n400_2/cls
    n800_2/cls
    n1600_2/cls
    n100_1/patch
    n200_1/patch
    n400_1/patch
    n800_1/patch
    n1600_1/patch
    n100_2/patch
    n200_2/patch
    n400_2/patch
    n800_2/patch
    n1600_2/patch
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

# 10 models x 7 datasets

num_datasets=${#datasets[@]}
configid=$(( $SLURM_ARRAY_TASK_ID / $num_datasets ))
datasetid=$(( $SLURM_ARRAY_TASK_ID % $num_datasets ))

config=${configs[configid]}
key=$(echo $config | cut -d / -f 1)
repr=$(echo $config | cut -d / -f 2)
clf="logistic"

model="flat_mae"
# nb using best checkpoint not last, since the small data models are overfit
ckpt_path="${OUT_DIR}/${EXP_NAME}/${key}/pretrain/checkpoint-best.pth"
if [[ ! -f $ckpt_path ]]; then
    echo "checkpoint ${ckpt_path} doesn't exist; not running"
    exit
fi

dataset=${datasets[datasetid]}
overrides="model_kwargs.ckpt_path=${ckpt_path} batch_size=2"

notes="data scaling experiment $key; eval v2 (${dataset} ${repr} ${clf})"

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
