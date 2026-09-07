#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python inference_distilled.py \
    --config configs/train_lora_removal_stripe_260130_r64.yaml \
    --lora_path exps/distill_dmd2_lora_removal_stripe_4step \
    --lora_strength 1.0 \
    --image_root validation_data/test_imgs \
    --mask_root validation_data/test_mask \
    --output_root exps_outputs_dmd2/removal_4step \
    --seed 2025 \
    "$@"
