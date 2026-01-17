#!/usr/bin/env bash
#SBATCH --job-name=data_scaling
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=infinite
#SBATCH --partition=main
#SBATCH --output=slurms/slurm-%A_%a.out
#SBATCH --nodelist=n-3,n-4
#SBATCH --account=training
#SBATCH --array=0-9

set -euo pipefail

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
    "n100_1|0|99|seed=1644"
    "n200_1|0|199|seed=1644"
    "n400_1|0|399|seed=1644"
    "n800_1|0|799|seed=1644"
    "n1600_1|0|1599|seed=1644"
    "n100_2|800|899|seed=3472"
    "n200_2|800|999|seed=3472"
    "n400_2|800|1199|seed=3472"
    "n800_2|800|1599|seed=3472"
    "n1600_2|0|1599|seed=3472"
)

config=${configs[SLURM_ARRAY_TASK_ID]}
name=$(echo $config | cut -d '|' -f 1)
start=$(echo $config | cut -d '|' -f 2 | xargs printf '%05d')
stop=$(echo $config | cut -d '|' -f 3 | xargs printf '%05d')
overrides=$(echo $config | cut -d '|' -f 4)

base_config="${EXP_DIR}/pretrain.yaml"
fullname="${EXP_NAME}/${name}/pretrain"
notes="data scaling experiment $name (${overrides})"

base_url="/data/fmri-datasets/pretrain/hcpya-all.flat.wds"
url="${base_url}/hcpya-all-flat-{${start}..${stop}}.tar"

# add small delay between jobs
# bit of hack to try to get wandb to assign different colors
sleep $(( SLURM_ARRAY_TASK_ID * 10 ))

# for debugging
# overrides="${overrides} debug=true wandb=false"

uv run torchrun --standalone --nproc_per_node=1 \
    src/flat_mae/main_pretrain.py \
    --cfg-path "${base_config}" \
    --overrides \
    $overrides \
    datasets.hcp-train.url=\"${url}\" \
    name="${fullname}" \
    notes="${notes}" \
    output_dir="${OUT_DIR}"
