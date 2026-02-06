#!/usr/bin/env bash
#SBATCH --job-name=input_space
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=infinite
#SBATCH --partition=main
#SBATCH --output=slurms/slurm-%A_%a.out
#SBATCH --nodelist=n-1,n-2,n-4
#SBATCH --account=training
# #SBATCH --array=0-23
# #SBATCH --array=24-39
#SBATCH --array=40-55
#SBATCH --dependency=afterany:2937

set -euo pipefail

export OMP_NUM_THREADS=8

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
    schaefer400_lr3e-4_1/patch/attn
    schaefer400_lr3e-4_2/patch/attn
    schaefer400_lr1e-3_1/patch/attn
    schaefer400_lr1e-3_2/patch/attn
    schaefer400_lr3e-3_1/patch/attn
    schaefer400_lr3e-3_2/patch/attn
    schaefer400_lr1e-4_1/patch/attn
    schaefer400_lr1e-4_2/patch/attn
    schaefer400_lr3e-5_1/patch/attn
    schaefer400_lr3e-5_2/patch/attn
    mni_cortex_lr3e-4_1/patch/attn
    mni_cortex_lr3e-4_2/patch/attn
    mni_cortex_lr1e-3_1/patch/attn
    mni_cortex_lr1e-3_2/patch/attn
)

datasets=(
    aabc_age
    hcpya_rest1lr_gender
    hcpya_task21
    nsd_cococlip
)
batch_sizes=(
    2
    2
    64
    64
)

num_datasets=${#datasets[@]}
configid=$(( $SLURM_ARRAY_TASK_ID / $num_datasets ))
datasetid=$(( $SLURM_ARRAY_TASK_ID % $num_datasets ))

config=${configs[configid]}
key=$(echo $config | cut -d / -f 1)
space=$(echo $key | sed 's/\(.*\)_lr.*/\1/')
repr=$(echo $config | cut -d / -f 2)
clf=$(echo $config | cut -d / -f 3)

model="${space}_mae"
ckpt_path="${OUT_DIR}/${EXP_NAME}/${key}/pretrain/checkpoint-last.pth"
if [[ ! -f $ckpt_path ]]; then
    echo "checkpoint ${ckpt_path} doesn't exist; not running"
    exit
fi

dataset=${datasets[datasetid]}
bs=${batch_sizes[datasetid]}
overrides="model_kwargs.ckpt_path=${ckpt_path} epochs=4 batch_size=${bs} accum_iter=2 lr=0.001 num_workers=8 wandb=false"

notes="input_space ablation v2 $key; eval (${dataset} ${repr} ${clf})"

name="${EXP_NAME}/${key}/eval/${dataset}__${repr}__${clf}"
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
