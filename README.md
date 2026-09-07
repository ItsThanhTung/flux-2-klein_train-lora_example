# FLUX.2 Klein LoRA training examples

Train LoRAs on [FLUX.2-klein](https://huggingface.co/black-forest-labs) with Diffusers + Accelerate + PEFT. This repo covers:

1. **Object removal** — stripe-mask conditioned erase LoRA (9B)
2. **Style transfer** — content + style reference → stylized output (9B)
3. **DMD2 distillation** — four-step student LoRA for object removal (see [README_DISTILL.md](README_DISTILL.md))

## Setup

Python 3.10 recommended.

```bash
bash scripts/install.sh
# or: pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
#     pip install -r requirements.txt
```

Configure Accelerate for your GPU count (`accelerate config`, or pass `--config_file` / `--num_processes`).

Edit dataset paths in the YAML configs before training — several defaults point at local cluster paths.

## Object removal LoRA

Stripe-mask conditioning on `black-forest-labs/FLUX.2-klein-base-9B`.

| | Path |
|---|---|
| Config | `configs/train_lora_removal_stripe_260130_r64.yaml` |
| Train | `scripts/train_lora.sh` |
| Infer (base) | `scripts/infer_base.sh` |
| Infer (distilled pipe) | `scripts/infer_distilled.sh` |
| Data download | `scripts/download_data.sh` |

```bash
bash scripts/download_data.sh
bash validation_data/download.sh   # optional validation images/masks

bash scripts/train_lora.sh
# logs → train_logs/ ; checkpoints → exps/train_lora_removal_stripe_multi-data_9b_r64/

bash scripts/infer_base.sh
```

Datasets are registered in `datasets/` (`ObjectEffectStripeDataset`, etc.) and selected via `train_datasets` in the config.

## Style transfer LoRA

Two controls: **content** image + **style** reference → stylized target. Prompt default:

> apply the artistic style of the second image to the first image, keep the subject and composition of the first image

| | Unified pairs (`pairs.jsonl`) | OmniConsistency (`pairs.json`) |
|---|---|---|
| Config | `configs/train_lora_style_transfer_9b.yaml` | `configs/train_lora_style_transfer_ominidataset_9b.yaml` |
| Train | `scripts/train_lora_style_transfer.sh` | `scripts/train_lora_style_transfer_ominidataset.sh` |
| Infer | `scripts/infer_style_transfer.sh` | same script; point `--lora_path` / data roots |

```bash
# update data_root / pairs paths in the YAML, then:
bash scripts/train_lora_style_transfer.sh
# or
bash scripts/train_lora_style_transfer_ominidataset.sh

python inference_style_transfer.py \
  --config exps/<run>/training_config.yaml \
  --lora_path exps/<run>/checkpoint-1000 \
  --content /path/to/content.jpg \
  --style /path/to/style.jpg \
  --output out.png \
  --seed 2025
```

`StyleTransferDataset` accepts several pair schemas (`content_image`/`style_image`/`stylized_image`, `image`/`ref_style`/`edited_image`, or `content`/`style`/`target`).

## Distillation (object removal)

Four-step DMD2 distillation of the removal LoRA:

```bash
bash scripts/distill_dmd2_lora_removal_stripe.sh
```

Full details: [README_DISTILL.md](README_DISTILL.md).

## Layout

```text
train_lora_base.py          # shared LoRA trainer
inference_*.py              # base / distilled / style-transfer inference
datasets/                   # dataset classes used by configs
configs/                    # OmegaConf training YAMLs
scripts/                    # install, train, infer launchers
exps/                       # checkpoints (gitignored)
train_logs/                 # train logs (gitignored)
```

## Notes

- Base model: `FLUX.2-klein-base-9B` (or 4B for distill teacher/student as in the distill README).
- Typical LoRA rank: **64**; bf16 + gradient checkpointing; `offload: true` helps on ~64GB GPUs for style transfer.
- Validation grids (style transfer) and W&B logging are controlled by `validation_*` / `report_to` in the config.
- Override any config field on the CLI: `accelerate launch train_lora_base.py --config ... --learning_rate 1e-4`.
