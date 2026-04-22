#!/usr/bin/env bash
#SBATCH --job-name=decoder_depth
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=infinite
#SBATCH --partition=main
#SBATCH --output=slurms/slurm-%A_%a.out
#SBATCH --account=training
#SBATCH --qos=low
# #SBATCH --array=0-4
#SBATCH --array=4
#SBATCH --requeue

set -euo pipefail

ROOT="/data/connor/fmri-fm"
cd $ROOT

# export all env variables
set -a
source .env
set +a

EXP_NAME="decoder_depth"
EXP_DIR="experiments/${EXP_NAME}"
OUT_DIR="${EXP_DIR}/output"

# default decoder embed dim = 512, heads = 16 from MAE
# standard ViT-B embed dim = 768, heads = 12
configs=(
    "d4_c512_h16_1|model_kwargs.decoder_depth=4 model_kwargs.decoder_embed_dim=512 model_kwargs.decoder_num_heads=16 seed=2751"
    "d2_c512_h16_1|model_kwargs.decoder_depth=2 model_kwargs.decoder_embed_dim=512 model_kwargs.decoder_num_heads=16 seed=2751"
    "d8_c512_h16_1|model_kwargs.decoder_depth=8 model_kwargs.decoder_embed_dim=512 model_kwargs.decoder_num_heads=16 seed=2751"
    "d12_c512_h16_1|model_kwargs.decoder_depth=12 model_kwargs.decoder_embed_dim=512 model_kwargs.decoder_num_heads=16 seed=2751"
    "d4_c768_h12_1|model_kwargs.decoder_depth=4 model_kwargs.decoder_embed_dim=768 model_kwargs.decoder_num_heads=12 seed=2751"
)

config=${configs[SLURM_ARRAY_TASK_ID]}
name=$(echo $config | cut -d '|' -f 1)
overrides=$(echo $config | cut -d '|' -f 2)

base_config="${EXP_DIR}/pretrain.yaml"
fullname="${EXP_NAME}/${name}/pretrain"
notes="decoder_depth ablation $name (${overrides})"

# add small delay between jobs
# bit of hack to try to get wandb to assign different colors
sleep $(( SLURM_ARRAY_TASK_ID * 5 ))

# for debugging
# overrides="${overrides} debug=true wandb=false"

uv run torchrun --standalone --nproc_per_node=1 \
    src/flat_mae/main_pretrain.py \
    --cfg-path "${base_config}" \
    --overrides \
    $overrides \
    name="${fullname}" \
    notes="${notes}" \
    output_dir="${OUT_DIR}"
