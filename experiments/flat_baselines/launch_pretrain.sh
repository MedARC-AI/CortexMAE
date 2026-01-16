#!/usr/bin/env bash
#SBATCH --job-name=flat_baselines
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=infinite
#SBATCH --partition=main
#SBATCH --output=slurms/slurm-%A_%a.out
#SBATCH --nodelist=n-2,n-3,n-4
#SBATCH --account=training
#SBATCH --array=0-11

set -euo pipefail

ROOT="${HOME}/fmri-fm"
cd $ROOT

# export all env variables
set -a
source .env
set +a

EXP_NAME="flat_baselines"
EXP_DIR="experiments/${EXP_NAME}"
OUT_DIR="${EXP_DIR}/output"

configs=(
    "pt-2|t_patch_size=2 masking=tube"
    "pt-2_bf16|t_patch_size=2 masking=tube amp_dtype=bfloat16"
    "pt-2_simarch|t_patch_size=2 masking=tube model_kwargs.no_decode_pos=true model_kwargs.no_embed_class=true"
    "pt-2_zinit|t_patch_size=2 masking=tube model_kwargs.head_init_scale=0.0"
    "pt-2_mds|t_patch_size=2 masking=tube model_kwargs.mask_drop_scale=true"
    "pt-2_noclip|t_patch_size=2 masking=tube clip_vmax=null"
    "pt-2_nonorm|t_patch_size=2 masking=tube normalize=null"
    "pt-2_lr5e-4|t_patch_size=2 masking=tube base_lr=5e-4"
    "pt-2_lr2e-3|t_patch_size=2 masking=tube base_lr=2e-3"
    "pt-2_bs64|t_patch_size=2 masking=tube batch_size=64"
    "pt-2_clip0.2|t_patch_size=2 masking=tube clip_grad=0.2"
    "pt-2_buf2k|t_patch_size=2 masking=tube datasets.hcp-train.buffer_size=2000"
)

config=${configs[SLURM_ARRAY_TASK_ID]}
name=$(echo $config | cut -d '|' -f 1)
overrides=$(echo $config | cut -d '|' -f 2)
# allow resuming with config change (I added plot_period)
overrides="${overrides} unsafe_resume=true"

fullname="${EXP_NAME}/${name}/pretrain"
notes="misc flat baseline ablations $name (${overrides})"

# add small delay between jobs
# bit of hack to try to get wandb to assign different colors
# sleep $(( SLURM_ARRAY_TASK_ID * 10 ))

uv run torchrun --standalone --nproc_per_node=1 \
    src/flat_mae/main_pretrain.py \
    --overrides \
    $overrides \
    name="${fullname}" \
    notes="${notes}" \
    output_dir="${OUT_DIR}"
