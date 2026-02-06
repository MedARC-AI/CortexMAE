#!/usr/bin/env bash
#SBATCH --job-name=data_scaling
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=infinite
#SBATCH --partition=main
#SBATCH --output=slurms/slurm-%A_%a.out
#SBATCH --nodelist=n-1,n-2,n-4
#SBATCH --account=training
#SBATCH --array=0-39

set -euo pipefail

export OMP_NUM_THREADS=8

ROOT="${HOME}/fmri-fm"
cd $ROOT

# export all env variables
set -a
source .env
set +a

EXP_NAME="data_scaling"
EXP_DIR="experiments/${EXP_NAME}"
OUT_DIR="${EXP_DIR}/output"

configs=(
    n1600_1/4/patch/attn
    n1600_1/9/patch/attn
    n1600_1/14/patch/attn
    n1600_1/19/patch/attn
    n1600_1/24/patch/attn
    n1600_1/29/patch/attn
    n1600_1/34/patch/attn
    n1600_1/39/patch/attn
    n1600_1/44/patch/attn
    n1600_1/49/patch/attn
    n1600_1/54/patch/attn
    n1600_1/59/patch/attn
    n1600_1/64/patch/attn
    n1600_1/69/patch/attn
    n1600_1/74/patch/attn
    n1600_1/79/patch/attn
    n1600_1/84/patch/attn
    n1600_1/89/patch/attn
    n1600_1/94/patch/attn
    n1600_1/99/patch/attn
    n1600_2/4/patch/attn
    n1600_2/9/patch/attn
    n1600_2/14/patch/attn
    n1600_2/19/patch/attn
    n1600_2/24/patch/attn
    n1600_2/29/patch/attn
    n1600_2/34/patch/attn
    n1600_2/39/patch/attn
    n1600_2/44/patch/attn
    n1600_2/49/patch/attn
    n1600_2/54/patch/attn
    n1600_2/59/patch/attn
    n1600_2/64/patch/attn
    n1600_2/69/patch/attn
    n1600_2/74/patch/attn
    n1600_2/79/patch/attn
    n1600_2/84/patch/attn
    n1600_2/89/patch/attn
    n1600_2/94/patch/attn
    n1600_2/99/patch/attn
)

datasets=(
    aabc_age
)
batch_sizes=(
    2
)

num_datasets=${#datasets[@]}
configid=$(( $SLURM_ARRAY_TASK_ID / $num_datasets ))
datasetid=$(( $SLURM_ARRAY_TASK_ID % $num_datasets ))

config=${configs[configid]}
key=$(echo $config | cut -d / -f 1)
epoch=$(echo $config | cut -d / -f 2)
repr=$(echo $config | cut -d / -f 3)
clf=$(echo $config | cut -d / -f 4)

model="flat_mae"
epoch=$(printf '%05d' $epoch)
ckpt_path="${OUT_DIR}/${EXP_NAME}/${key}/pretrain/checkpoint-${epoch}.pth"
if [[ ! -f $ckpt_path ]]; then
    echo "checkpoint ${ckpt_path} doesn't exist; not running"
    exit
fi

dataset=${datasets[datasetid]}
bs=${batch_sizes[datasetid]}
overrides="model_kwargs.ckpt_path=${ckpt_path} epochs=4 batch_size=${bs} accum_iter=2 lr=0.001 num_workers=8 wandb=false"

notes="data scaling experiment $key; eval epoch=${epoch} (${dataset} ${repr} ${clf})"

name="${EXP_NAME}/${key}/eval/${dataset}__${repr}__${clf}__${epoch}"
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
