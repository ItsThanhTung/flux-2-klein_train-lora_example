ckpt_idx=( 1000 2000)
EXP_DIR=exps/train_lora_removal_stripe_multi-data_9b_r64
for ckpt_idx in "${ckpt_idx[@]}"; do

    python inference_distilled.py \
        --config ${EXP_DIR}/training_config.yaml \
        --lora_path ${EXP_DIR}/checkpoint-${ckpt_idx} \
        --lora_strength 1.0 \
        --image_root validation_data/test_imgs \
        --mask_root validation_data/test_mask \
        --output_root exps_outputs_9b_r64/train-${ckpt_idx}_distilled \
        --seed 2025
done
