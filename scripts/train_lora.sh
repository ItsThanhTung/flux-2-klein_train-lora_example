mkdir -p train_logs

accelerate launch train_lora_base.py --config configs/train_lora_removal_stripe_260130_r64.yaml \
    --output_dir exps/train_lora_removal_stripe_multi-data_9b_r64 \
    > train_logs/train_lora_removal_stripe_multi-data_9b_r64.log 2>&1
