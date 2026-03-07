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
# #SBATCH --array=0-13
#SBATCH --array=10-13

set -euo pipefail

export OMP_NUM_THREADS=8

# ROOT="${HOME}/fmri-fm"
ROOT="/data/connor/fmri-fm"
cd $ROOT

# export all env variables
set -a
source .env
set +a

EXP_NAME="input_norm"
EXP_DIR="experiments/${EXP_NAME}"
OUT_DIR="${EXP_DIR}/output"

configs=(
    coord_frame/patch/attn
    coord_global/patch/attn
    coord_none/patch/attn
    nocoord_frame/patch/attn
    nocoord_global/patch/attn
    nocoord_frame_mni/patch/attn
    nocoord_global_mni/patch/attn
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

if [[ $key == *_mni ]]; then
    model="mni_cortex_mae"
else
    model="flat_mae"
fi
ckpt_path="${OUT_DIR}/${EXP_NAME}/${key}/pretrain/checkpoint-last.pth"
if [[ ! -f $ckpt_path ]]; then
    echo "checkpoint ${ckpt_path} doesn't exist; not running"
    exit
fi

dataset=${datasets[datasetid]}
bs=${batch_sizes[datasetid]}
overrides="model_kwargs.ckpt_path=${ckpt_path} batch_size=${bs} accum_iter=2"

notes="input_norm ablation $key; eval v2 (${dataset} ${repr} ${clf})"

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
