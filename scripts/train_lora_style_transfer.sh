#!/usr/bin/env bash
set -euo pipefail
ROOT=/vast/users/zhenyu.xie/tungdt/code/klein
EX="$ROOT/code/flux-2-klein_train-lora_example"
source "$ROOT/env/bin/activate"
if [ -f /vast/users/zhenyu.xie/tungdt/slurm_scripts/activate_amd_gpu.sh ]; then
  # shellcheck disable=SC1091
  source /vast/users/zhenyu.xie/tungdt/slurm_scripts/activate_amd_gpu.sh || true
fi
cd "$EX"
mkdir -p train_logs

export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTHONUNBUFFERED=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1800}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:True}"

# total_batch = train_batch_size(1) * num_processes(4) * grad_accum(8) = 32
accelerate launch \
  --config_file "$ROOT/configs/accelerate_4gpu.yaml" \
  --num_processes 4 \
  --mixed_precision bf16 \
  train_lora_base.py \
  --config configs/train_lora_style_transfer_9b.yaml \
  --output_dir exps/train_lora_style_transfer_unified_9b_r64 \
  2>&1 | tee -a train_logs/train_lora_style_transfer_unified_9b_r64.log
