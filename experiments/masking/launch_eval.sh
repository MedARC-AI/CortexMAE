#!/usr/bin/env bash
#SBATCH --job-name=masking
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=infinite
#SBATCH --partition=main
#SBATCH --output=slurms/slurm-%A_%a.out
#SBATCH --nodelist=n-2,n-3,n-4
#SBATCH --account=training
#SBATCH --array=0-7

set -euo pipefail

export OMP_NUM_THREADS=8

ROOT="${HOME}/fmri-fm"
cd $ROOT

# export all env variables
set -a
source .env
set +a

EXP_NAME="masking"
EXP_DIR="experiments/${EXP_NAME}"
OUT_DIR="${EXP_DIR}/output"

configs=(
    uniform_mr0.5/patch/attn
    uniform_mr0.75/patch/attn
    uniform_mr0.9/patch/attn
    uniform_mr0.95/patch/attn
    tube_mr0.5/patch/attn
    tube_mr0.75/patch/attn
    tube_mr0.9/patch/attn
    tube_mr0.95/patch/attn
)
config=${configs[SLURM_ARRAY_TASK_ID]}
key=$(echo $config | cut -d / -f 1)
repr=$(echo $config | cut -d / -f 2)
clf=$(echo $config | cut -d / -f 3)

model="flat_mae"
ckpt_path="${OUT_DIR}/${EXP_NAME}/${key}/pretrain/checkpoint-last.pth"

datasets=(
    hcpya_rest1lr_gender
    aabc_sex
    hcpya_task21
    nsd_cococlip
)
batch_sizes=(
    2
    2
    64
    64
)

datasetids="0 1 2 3"

# add small delay between jobs
# sleep $(( SLURM_ARRAY_TASK_ID * 10 ))

for ii in $datasetids; do
    dataset=${datasets[ii]}
    bs=${batch_sizes[ii]}
    overrides="model_kwargs.ckpt_path=${ckpt_path} epochs=4 batch_size=${bs} accum_iter=2 lr=0.001 num_workers=8 wandb=false"

    name="${EXP_NAME}/${key}/eval/${dataset}__${repr}__${clf}"
    result="${OUT_DIR}/${name}/eval_table.csv"
    if [[ -f $result ]]; then
        echo "result $result exists; skipping"
        continue
    fi

    notes="masking ablations $key; eval (${dataset} ${repr} ${clf})"

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
done
