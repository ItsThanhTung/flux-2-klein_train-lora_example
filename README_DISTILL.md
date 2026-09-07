# Four-step object-removal LoRA distillation

This root-level workflow ports the DMD2 trainer from
`/Volumes/ESSD/workspace/flux2-klein-train-lora-distill-dmd` and specializes it
for the striped object-removal LoRA trained by this repository. It is separate
from `distill_example` and does not change any existing training or inference
script.

## Inputs and output

The frozen teacher is `black-forest-labs/FLUX.2-klein-base-4B` plus the final
LoRA produced by `scripts/train_lora.sh`:

```text
exps/train_lora_removal_stripe_multi-data_260130_r64/
```

The student is `black-forest-labs/FLUX.2-klein-4B`. The final four-step LoRA is
written to:

```text
exps/distill_dmd2_lora_removal_stripe_4step/
```

Both paths can be overridden on the command line. A checkpoint directory or a
direct LoRA weights file is accepted.

## Setup

Install the normal project environment first:

```bash
bash scripts/install.sh
```

Core DMD2 and teacher-latent L1/L2 need no additional packages. To enable LPIPS
or the SenseFlow GAN, install the optional stack:

```bash
pip install -r requirements_distill.txt
```

Download the existing training and validation data before running a dry run:

```bash
bash scripts/download_data.sh
bash validation_data/download.sh
```

## Validate data and configuration

`--dry-run` loads one real batch and exits before downloading either model:

```bash
python distill_dmd2_lora_removal.py \
  --config configs/distill_dmd2_lora_removal_stripe.yaml \
  --dry-run
```

The expected shapes at the default batch size are:

```text
target batch: (1, 3, 1024, 1024)
condition batch: (1, 1, 3, 1024, 1024)
```

## Train

The supplied launcher targets one 80 GB-or-larger GPU and accepts any trainer
override after the script name:

```bash
bash scripts/distill_dmd2_lora_removal_stripe.sh
```

For example, warm-start from a distilled checkpoint while restarting optimizer
state and step counting:

```bash
bash scripts/distill_dmd2_lora_removal_stripe.sh \
  --student-lora-init-path exps/distill_dmd2_lora_removal_stripe_4step/checkpoint-300 \
  --output-dir exps/distill_dmd2_lora_removal_stripe_4step_resume
```

## L1 and L2 teacher-latent losses

The initial config keeps both losses disabled:

```yaml
teacher_latent_l1_weight: 0.0
teacher_latent_l2_weight: 0.0
teacher_latent_sample_steps: 30
```

They can be enabled independently. Teacher sampling is skipped entirely while
both weights are zero:

```bash
bash scripts/distill_dmd2_lora_removal_stripe.sh \
  --teacher-latent-l1-weight 1.0 \
  --teacher-latent-l2-weight 0.25
```
or set in the config file:

```yaml
teacher_latent_l1_weight: 1.0
teacher_latent_l2_weight: 0.25
```

The student objective adds weighted mean absolute error for L1 and weighted
mean squared error for L2 between complete student and teacher latent samples.

For some tasks, enable l1 or l2 loss to help the student learn better. You need to experience which config gives better results.

## Optional LPIPS and SenseFlow GAN

The full source loss stack is retained but disabled by default. Set
`lpips_weight` or `gan_weight` above zero only after installing
`requirements_distill.txt`. These paths require additional GPU memory.

## Four-step inference

After training, run:

```bash
bash scripts/infer_dmd2_lora_removal_stripe.sh
```

The distilled output is a standard Diffusers LoRA containing
`pytorch_lora_weights.safetensors`, so it can also be passed directly to
`Flux2KleinPipeline.load_lora_weights`.
