#!/usr/bin/env bash
#SBATCH --job-name=input_space
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=infinite
#SBATCH --partition=main
#SBATCH --output=slurms/slurm-%A_%a.out
# #SBATCH --nodelist=n-4
#SBATCH --account=training
#SBATCH --array=0,1,2

set -euo pipefail

export OMP_NUM_THREADS=8

ROOT="${HOME}/fmri-fm"
cd $ROOT

# export all env variables
set -a
source .env
set +a

EXP_NAME="input_space"
EXP_DIR="experiments/${EXP_NAME}"
OUT_DIR="${EXP_DIR}/output"

configs=(
    schaefer400/cls/linear
    schaefer400/patch/linear
    schaefer400/patch/attn
)
config=${configs[SLURM_ARRAY_TASK_ID]}
space=$(echo $config | cut -d / -f 1)
repr=$(echo $config | cut -d / -f 2)
clf=$(echo $config | cut -d / -f 3)

model="${space}_mae"
ckpt_path="${OUT_DIR}/${EXP_NAME}/${space}/pretrain/checkpoint-last.pth"

datasets=(
    hcpya_rest1lr_gender
    hcpya_task21
    nsd_cococlip
)
batch_sizes=(
    2
    64
    64
)

datasetids="0 1 2"
if [[ $space == "mni_cortex" ]]; then
    datasetids="0 1"
fi

for ii in $datasetids; do
    dataset=${datasets[ii]}
    bs=${batch_sizes[ii]}
    overrides="model_kwargs.ckpt_path=${ckpt_path} epochs=4 batch_size=${bs} accum_iter=2 lr=0.001 num_workers=8 wandb=false"

    name="${EXP_NAME}/${space}/eval/${dataset}__${repr}__${clf}"
    result="${OUT_DIR}/${name}/eval_table.csv"
    if [[ -f $result ]]; then
        echo "result $result exists; skipping"
        continue
    fi

    notes="input space ablation (input_space=${space}); eval (${dataset} ${repr} ${clf})"

    uv run python -m fmri_fm_eval.main_probe \
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
