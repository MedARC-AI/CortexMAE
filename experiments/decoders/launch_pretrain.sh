#!/usr/bin/env bash
#SBATCH --job-name=decoders
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

ROOT="${HOME}/fmri-fm"
cd $ROOT

# export all env variables
set -a
source .env
set +a

EXP_NAME="decoders"
EXP_DIR="experiments/${EXP_NAME}"
OUT_DIR="${EXP_DIR}/output"

configs=(
    "attn_reg1|model_kwargs.decoding=attn model_kwargs.reg_tokens=1"
    "cross_reg1|model_kwargs.decoding=cross model_kwargs.reg_tokens=1"
    "crossreg_reg1|model_kwargs.decoding=crossreg model_kwargs.reg_tokens=1"
    "crossreg_reg4|model_kwargs.decoding=crossreg model_kwargs.reg_tokens=4"
    "crossreg_reg16|model_kwargs.decoding=crossreg model_kwargs.reg_tokens=16"
    "attn_reg1_pep4|model_kwargs.decoding=attn model_kwargs.reg_tokens=1 model_kwargs.pred_edge_pad=4"
    "cross_reg1_pep4|model_kwargs.decoding=cross model_kwargs.reg_tokens=1 model_kwargs.pred_edge_pad=4"
    "crossreg_reg4_pep4|model_kwargs.decoding=crossreg model_kwargs.reg_tokens=4 model_kwargs.pred_edge_pad=4"
)

config=${configs[SLURM_ARRAY_TASK_ID]}
name=$(echo $config | cut -d '|' -f 1)
overrides=$(echo $config | cut -d '|' -f 2)

base_config="${EXP_DIR}/pretrain.yaml"
fullname="${EXP_NAME}/${name}/pretrain"
notes="decoder ablations $name (${overrides})"

# add small delay between jobs
# bit of hack to try to get wandb to assign different colors
sleep $(( SLURM_ARRAY_TASK_ID * 10 ))

uv run torchrun --standalone --nproc_per_node=1 \
    src/flat_mae/main_pretrain.py \
    --cfg-path "${base_config}" \
    --overrides \
    $overrides \
    name="${fullname}" \
    notes="${notes}" \
    output_dir="${OUT_DIR}"
