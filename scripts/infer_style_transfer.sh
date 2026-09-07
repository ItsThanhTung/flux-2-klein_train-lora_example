#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

EXP_DIR=exps/train_lora_style_transfer_unified_9b_r64
DATA_ROOT=/vast/users/zhenyu.xie/tungdt/code/klein/data/unified_style_data
ckpt_idx=(500 1000 1500 2000 2500 3000 3500 4000 4500 5000)

for ckpt in "${ckpt_idx[@]}"; do
  ckpt_path="${EXP_DIR}/checkpoint-${ckpt}"
  if [[ ! -d "${ckpt_path}" ]]; then
    echo "skip missing ${ckpt_path}"
    continue
  fi
  python inference_style_transfer.py \
    --config "${EXP_DIR}/training_config.yaml" \
    --lora_path "${ckpt_path}" \
    --content "${DATA_ROOT}/sample_refs/content_00.jpg" \
    --style "${DATA_ROOT}/sample_refs/style_00.jpg" \
    --output "exps_outputs_style_9b/train-${ckpt}_sample00.png" \
    --seed 2025
done
