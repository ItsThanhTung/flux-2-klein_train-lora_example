#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

mkdir -p train_logs

PYTORCH_ALLOC_CONF=expandable_segments:True accelerate launch \
    --num_processes 1 \
    distill_dmd2_lora_removal.py \
    --config configs/distill_dmd2_lora_removal_stripe.yaml \
    "$@" \
    2>&1 | tee train_logs/distill_dmd2_lora_removal_stripe.log
