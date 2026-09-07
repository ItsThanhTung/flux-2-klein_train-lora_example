# Style transfer LoRA example (Diffusers / Accelerate)
#
# Object removal:
#   scripts/install.sh
#   scripts/train_lora.sh
#   scripts/infer_base.sh / infer_distilled.sh
#
# Style transfer (content + style_ref → edited), Klein 9B:
#   configs/train_lora_style_transfer_9b.yaml
#   scripts/train_lora_style_transfer.sh
#   scripts/infer_style_transfer.sh
#
# Recommend workflow:
# - Build config files and datasets (`datasets/*.py`)
# - Train with accelerate
