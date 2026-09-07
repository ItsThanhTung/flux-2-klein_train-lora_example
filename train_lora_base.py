import gc
import sys
import signal
import threading
import urllib.request
from omegaconf import OmegaConf

import argparse
import copy
import itertools
import json
import logging
import math
import os
import random
import shutil
import warnings
from contextlib import nullcontext
from pathlib import Path
from omegaconf import OmegaConf

import numpy as np
import torch
import transformers
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from PIL import Image
from tqdm.auto import tqdm

from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM

import diffusers
from diffusers import (
    AutoencoderKLFlux2,
    FlowMatchEulerDiscreteScheduler,
    Flux2Transformer2DModel,
    Flux2KleinPipeline
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    _collate_lora_metadata,
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    free_memory,
    offload_models
)
from diffusers.utils import (
    check_min_version,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
)
from diffusers.utils.import_utils import is_torch_npu_available
from diffusers.utils.torch_utils import is_compiled_module

from datasets.prob_picking_dataset import ProbPickingDataset
from datasets.object_effect_dataset import ObjectEffectDataset, create_mask_overlayed_image
from datasets.object_effect_stripe_dataset import ObjectEffectStripeDataset, create_mask_overlayed_image_stripes
from datasets.style_transfer_dataset import StyleTransferDataset

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.36.0.dev0")

logger = get_logger(__name__)

if is_torch_npu_available():
    torch.npu.config.allow_internal_format = False


def resize_target_area(image, target_area=1024*1024):
    width, height = image.size
    if width * height > target_area:
        scale = (target_area / (width * height)) ** 0.5
        new_width = int(width * scale) // 32 * 32
        new_height = int(height * scale) // 32 * 32
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return image

def log_validation(
    pipeline,
    args,
    accelerator,
    pipeline_args,
    step,
    torch_dtype,
):
    logger.info("Running validation...")
    pipeline = pipeline.to(accelerator.device, dtype=torch_dtype)
    pipeline.set_progress_bar_config(disable=True)

    # change system prompt template
    if args.prompt_template_encode is not None:
        pipeline.prompt_template_encode = args.prompt_template_encode

    autocast_ctx = torch.autocast(accelerator.device.type)
    
    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed is not None else None

    if 'train_datasets' in args:
        dataset_config = args.train_datasets[0]
    elif 'train_dataset' in args:
        dataset_config = args.train_dataset
    else:
        raise ValueError(f"train_dataset or train_datasets must be provided")

    # load validation images
    image_logs = []
    for case in args.validation_data.cases:
        # Style transfer: content + style → edited
        if dataset_config.type == "StyleTransferDataset" or (
            hasattr(case, "content_root") and case.content_root is not None
        ):
            content_root = case.content_root
            style_root = case.style_root
            num_samples = case.num_samples
            content_files = sorted(
                f for f in os.listdir(content_root)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
            )
            content_files = random.sample(content_files, min(num_samples, len(content_files)))

            for content_file in content_files:
                content_img = Image.open(os.path.join(content_root, content_file)).convert("RGB")
                style_img = Image.open(os.path.join(style_root, content_file)).convert("RGB")

                content_input = resize_target_area(content_img)
                style_input = resize_target_area(style_img)
                # match style spatial size to content for the pipeline call
                style_input = style_input.resize(content_input.size, Image.Resampling.LANCZOS)

                with autocast_ctx, torch.no_grad():
                    output = pipeline(
                        image=[content_input, style_input],
                        prompt_embeds=pipeline_args["prompt_embeds"],
                        negative_prompt_embeds=pipeline_args["negative_prompt_embeds"],
                        height=content_input.height,
                        width=content_input.width,
                        generator=generator,
                        num_inference_steps=20,
                        guidance_scale=4.0,
                    ).images[0]

                output = output.resize(content_img.size, Image.Resampling.LANCZOS)
                style_vis = style_img.resize(content_img.size, Image.Resampling.LANCZOS)
                image_grid = Image.new("RGB", (3 * content_img.width, content_img.height))
                image_grid.paste(content_img, (0, 0))
                image_grid.paste(style_vis, (content_img.width, 0))
                image_grid.paste(output, (2 * content_img.width, 0))
                image_logs.append(image_grid)
            continue

        # Object-removal: image + mask overlay
        image_root = case.image_root
        mask_root = case.mask_root
        num_samples = case.num_samples

        image_files = sorted(os.listdir(image_root))

        # random pick samples
        image_files = random.sample(image_files, num_samples)

        for image_file in image_files:
            if image_file == '.DS_Store':
                continue
            mask_file = image_file.split('.')[0] + '.png'

            val_image = Image.open(os.path.join(image_root, image_file)).convert("RGB")
            val_mask = Image.open(os.path.join(mask_root, mask_file))

            if val_mask.mode != 'L':
                val_mask = np.array(val_mask)[..., -1]
            else:
                val_mask = np.array(val_mask)

            if dataset_config.type == "ObjectEffectDataset":
                val_mask = val_mask > 0
                mask_color = dataset_config.kwargs.mask_color
                opacity = dataset_config.kwargs.mask_overlay_opacity
                val_mask_overlayed_image = create_mask_overlayed_image(np.array(val_image), val_mask, mask_color=mask_color, opacity=opacity)
            elif dataset_config.type == "ObjectEffectStripeDataset":
                val_mask = (val_mask > 0).astype(np.uint8)
                opacity = dataset_config.kwargs.mask_overlay_opacity
                stripe_width = dataset_config.kwargs.stripe_width
                val_mask_overlayed_image = create_mask_overlayed_image_stripes(np.array(val_image), val_mask, opacity=opacity, stripe_width=stripe_width)
            else:
                raise ValueError(f"Unsupported dataset type for validation: {dataset_config.type}")

            val_mask_overlayed_image = Image.fromarray(val_mask_overlayed_image)

            # resize first
            val_mask_overlayed_image_input_for_model = resize_target_area(val_mask_overlayed_image)

            with autocast_ctx, torch.no_grad():
                output = pipeline(
                    # prompt=args.model_prompt,
                    image=val_mask_overlayed_image_input_for_model,
                    # precomputed prompt embeds and negative prompt embeds
                    prompt_embeds=pipeline_args["prompt_embeds"],
                    negative_prompt_embeds=pipeline_args["negative_prompt_embeds"],
                    # pass height and width
                    height=val_mask_overlayed_image_input_for_model.height,
                    width=val_mask_overlayed_image_input_for_model.width,
                    #
                    generator=generator,
                    num_inference_steps=50,
                    guidance_scale=4.0,
                ).images[0]
            
            output = output.resize((val_image.width, val_image.height))

            # concat and save
            image_grid = Image.new('RGB', (2 * val_image.width, val_image.height))
            image_grid.paste(val_mask_overlayed_image, (0, 0))
            image_grid.paste(output, (val_image.width, 0))

            image_logs.append(image_grid)

    for tracker in accelerator.trackers:
        phase_name = "validation"
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in image_logs])
            tracker.writer.add_images(phase_name, np_images, step, dataformats="NHWC")
        if tracker.name == "wandb":
            tracker.log(
                {
                    phase_name: [
                        wandb.Image(image, caption=f"Val index {i}") for i, image in enumerate(image_logs)
                    ]
                }
            )

    del pipeline
    free_memory()


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the config file.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_lora_path",
        type=str,
        default=None,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="hidream-dreambooth-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=512,
        help="Maximum sequence length to use with the Qwen2.5 VL as text encoder (Not used at this moment).",
    )
    ############################
    # LoRA training parameters
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=None,
        help="LoRA alpha to be used for additional scaling. If not provided, the rank will be used.",
    )
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="Dropout probability for LoRA layers")
    parser.add_argument(
        "--lora_layers",
        type=str,
        default=None,
        help=(
            'The transformer modules to apply LoRA training on. Please specify the layers in a comma separated. E.g. - "to_k,to_q,to_v" will result in lora training of attention layers only'
        ),
    )
    ############################
    # Dataset parameters
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    ############################
    # Training parameters
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--offload",
        action="store_true",
        default=False,
        help="Whether to offload the model to CPU.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")

    ############################
    # Validation parameters
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=1000,
        help="Number of steps between each validation.",
    )

    ############################
    # Qwen model params
    parser.add_argument(
        "--prompt_template_encode",
        type=str,
        default=None,
        help="System prompt template for encoding images. If not provided, the default template will be used.",
    )
    parser.add_argument(
        "--model_prompt",
        type=str,
        help="Model prompt to use for the model.",
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help=('We default to the "none" weighting scheme for uniform sampling and uniform loss'),
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    ############################
    # Optimizer parameters
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help=('The optimizer type to use. Choose between ["AdamW", "prodigy"]'),
    )

    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )

    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="coefficients for computing the Prodigy stepsize using running averages. If set to None, "
        "uses the value of square root of beta2. Ignored if optimizer is adamW",
    )
    parser.add_argument("--prodigy_decouple", type=bool, default=True, help="Use AdamW style decoupled weight decay")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04, help="Weight decay to use for unet params")

    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )

    parser.add_argument(
        "--prodigy_use_bias_correction",
        type=bool,
        default=True,
        help="Turn on Adam's bias correction. True by default. Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        type=bool,
        default=True,
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage. True by default. "
        "Ignored if optimizer is adamW",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    ############################

    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="flux2-klein-4b_lora_object-effect-removal",
        help="The name of the tracker to use.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def calculate_dimensions(target_area, ratio):
    width = math.sqrt(target_area * ratio)
    height = width / ratio

    width = round(width / 32) * 32
    height = round(height / 32) * 32

    return width, height, None



def main():
    args = parse_args()
    # use omegaconf to manage configurations
    if args.config is not None:
        config = OmegaConf.load(args.config)
        for k, v in config.items():
            args.__dict__[k] = v

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        # save config to folder
        to_save_config = OmegaConf.create(vars(args))
        OmegaConf.save(config=to_save_config, f=os.path.join(args.output_dir, "training_config.yaml"))

    # For mixed precision training we cast all non-trainable weights (vae, text_encoder and transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    to_kwargs = {"dtype": weight_dtype, "device": accelerator.device} if not args.offload else {"dtype": weight_dtype}
    if args.offload:
        logger.info("Using OFFLOAD")

    # Load scheduler and models
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="scheduler", 
        revision=args.revision,
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    # load vae
    vae = AutoencoderKLFlux2.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )
    vae.to(**to_kwargs)

    latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(accelerator.device)
    latents_bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(
        accelerator.device
    )

    # load transformer
    transformer = Flux2Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )
    # we never offload the transformer to CPU
    transformer.to(accelerator.device, dtype=weight_dtype)

    # load text encoder
    text_encoder = Qwen3ForCausalLM.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder.to(**to_kwargs)
    # Load the tokenizers
    tokenizer = Qwen2TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
    # form text encoding pipeline
    text_encoding_pipeline = Flux2KleinPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=None,
        transformer=None,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=None,
        revision=args.revision,
    )

    # We only train the additional adapter LoRA layers
    text_encoder.requires_grad_(False)
    transformer.requires_grad_(False)
    vae.requires_grad_(False)

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    if args.lora_layers is not None:
        target_modules = [layer.strip() for layer in args.lora_layers.split(",")]
    else:
        # Shared attn/MLP targets + per-block to_out for all single transformer blocks
        # (4B has 20 blocks, 9B has 24 — detect from the loaded model)
        target_modules = [
            "to_q",
            "to_k",
            "to_v",
            "to_out.0",
            "add_q_proj",
            "add_k_proj",
            "add_v_proj",
            "to_add_out",
            "linear_in",
            "linear_out",
            "to_qkv_mlp_proj",
        ]
        num_single_blocks = len(transformer.single_transformer_blocks)
        target_modules.extend(
            f"single_transformer_blocks.{i}.attn.to_out" for i in range(num_single_blocks)
        )
        logger.info(f"LoRA target modules: {len(target_modules)} names ({num_single_blocks} single blocks)")

    # now we will add new LoRA weights the transformer layers
    transformer_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank if args.lora_alpha is None else args.lora_alpha,
        lora_dropout=args.lora_dropout,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    transformer.add_adapter(transformer_lora_config)

    # Use to load pretrained LoRa if has
    def load_pretrained_lora(transformer_, input_dir):
        lora_state_dict = Flux2KleinPipeline.lora_state_dict(input_dir)

        transformer_state_dict = {
            f"{k.replace('transformer.', '')}": v for k, v in lora_state_dict.items() if k.startswith("transformer.")
        }
        transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
        incompatible_keys = set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="default")
        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )

    if args.pretrained_lora_path is not None:
        logger.info(f"Loading pretrained LoRa from {args.pretrained_lora_path}")
        load_pretrained_lora(transformer, args.pretrained_lora_path)
        logger.info(f"Pretrained LoRa loaded successfully")


    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            transformer_lora_layers_to_save = None
            modules_to_save = {}

            for model in models:
                if isinstance(unwrap_model(model), type(unwrap_model(transformer))):
                    model = unwrap_model(model)
                    transformer_lora_layers_to_save = get_peft_model_state_dict(model)
                    modules_to_save["transformer"] = model
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                if weights:
                    weights.pop()

            Flux2KleinPipeline.save_lora_weights(
                output_dir,
                transformer_lora_layers=transformer_lora_layers_to_save,
                **_collate_lora_metadata(modules_to_save),
            )

    def load_model_hook(models, input_dir):
        transformer_ = None

        if not accelerator.distributed_type == DistributedType.DEEPSPEED:
            while len(models) > 0:
                model = models.pop()

                if isinstance(unwrap_model(model), type(unwrap_model(transformer))):
                    model = unwrap_model(model)
                    transformer_ = model
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")
        else:
            transformer_ = Flux2KleinPipeline.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="transformer"
            )
            transformer_.add_adapter(transformer_lora_config)

        lora_state_dict = Flux2KleinPipeline.lora_state_dict(input_dir)

        transformer_state_dict = {
            f"{k.replace('transformer.', '')}": v for k, v in lora_state_dict.items() if k.startswith("transformer.")
        }
        transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
        incompatible_keys = set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="default")
        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )

        # Make sure the trainable params are in float32. This is again needed since the base models
        # are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
        if args.mixed_precision == "fp16":
            models = [transformer_]
            # only upcast trainable parameters (LoRA) into fp32
            cast_training_params(models)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        models = [transformer]
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(models, dtype=torch.float32)

    transformer_lora_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))

    # Optimization parameters
    transformer_parameters_with_lr = {"params": transformer_lora_parameters, "lr": args.learning_rate}
    params_to_optimize = [transformer_parameters_with_lr]

    # Optimizer creation
    if not (args.optimizer.lower() == "prodigy" or args.optimizer.lower() == "adamw"):
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}.Supported optimizers include [adamW, prodigy]."
            "Defaulting to adamW"
        )
        args.optimizer = "adamw"

    if args.use_8bit_adam and not args.optimizer.lower() == "adamw":
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    if args.optimizer.lower() == "adamw":
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    if args.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

        if args.learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )

    # Dataset and DataLoaders creation:
    if 'train_datasets' in args:
        subset_train_datasets = []
        subset_train_probs = []
        for train_dataset_config in args.train_datasets:
            subset_train_dataset = eval(train_dataset_config.type)(**train_dataset_config.kwargs)
            subset_train_datasets.append(subset_train_dataset)
            subset_train_probs.append(train_dataset_config.sample_prob)
        train_dataset = ProbPickingDataset(datasets=subset_train_datasets, probs=subset_train_probs)
    elif 'train_dataset' in args:
        train_dataset = eval(args.train_dataset.type)(**args.train_dataset.kwargs)
    else:
        raise ValueError(f"train_dataset or train_datasets must be provided")

    def collate_fn(examples):
        batch = {}
        batch["target_image"] = torch.stack([example["target_image"] for example in examples])
        batch["start_image"] = torch.stack([example["start_image"] for example in examples])
        batch["control_images"] = [example["control_images"] for example in examples]
        if "style_image" in examples[0] and examples[0]["style_image"] is not None:
            batch["style_image"] = torch.stack([example["style_image"] for example in examples])
        return batch

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
    )

    def compute_text_embeddings(prompt, text_encoding_pipeline):
        with torch.no_grad():
            prompt_embeds, text_ids = text_encoding_pipeline.encode_prompt(
                prompt=prompt, max_sequence_length=args.max_sequence_length
            )
        return prompt_embeds, text_ids

    #########################################################
    # Precompute text embeddings for the model prompt
    precomputed_prompt_embeds = None
    precomputed_text_ids = None
    with offload_models(text_encoding_pipeline, device=accelerator.device, offload=args.offload):
        precomputed_prompt_embeds, precomputed_text_ids = compute_text_embeddings(
            args.model_prompt, text_encoding_pipeline
        )

    # for validattion
    validation_kwargs = {}
    with offload_models(text_encoding_pipeline, device=accelerator.device, offload=args.offload):
        validation_kwargs["prompt_embeds"] = precomputed_prompt_embeds
        validation_kwargs["negative_prompt_embeds"], _text_ids = compute_text_embeddings(
            "", text_encoding_pipeline
        )


    # release text encoder now
    del text_encoder
    del tokenizer
    del text_encoding_pipeline
    free_memory()
    torch.cuda.empty_cache()

    # TODO: precompute VAE latents? (only for small dataset)

    #########################################################

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_project_name = args.tracker_project_name
        accelerator.init_trackers(tracker_project_name, config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Save checkpoint on receiving termination signals
    _checkpoint_save_in_progress = False  # Prevent double-saving
    
    def save_checkpoint(signum, frame):
        nonlocal _checkpoint_save_in_progress
        
        # Prevent re-entry if already saving
        if _checkpoint_save_in_progress:
            print('Checkpoint save already in progress, ignoring signal', flush=True)
            return
        _checkpoint_save_in_progress = True
        
        signal_names = {signal.SIGTERM: 'SIGTERM', signal.SIGINT: 'SIGINT', signal.SIGHUP: 'SIGHUP'}
        signal_name = signal_names.get(signum, f'Signal {signum}')
        
        # Use flush=True to ensure logs appear before potential termination
        print(f'\n=== Received {signal_name} - Attempting to save checkpoint ===', flush=True)
        
        # Log to file as well (stdout might be lost on spot termination)
        log_file = os.path.join(args.output_dir, 'emergency_checkpoint.log')
        try:
            with open(log_file, 'a') as f:
                import datetime
                f.write(f'{datetime.datetime.now()}: Received {signal_name}, global_step={global_step}\n')
        except:
            pass

        folder_name = f'checkpoint-{global_step}'
        checkpoint_path = os.path.join(args.output_dir, folder_name)
        
        accelerator.wait_for_everyone()
        try:
            if accelerator.is_main_process:
                print(f'Saving emergency checkpoint to: {checkpoint_path}', flush=True)
                # Create directory first
                os.makedirs(checkpoint_path, exist_ok=True)

                # Save full state if time permits
                try:
                    accelerator.save_state(checkpoint_path)
                    print(f'Full checkpoint saved to: {checkpoint_path}', flush=True)
                except Exception as e:
                    print(f'Warning: Full state save failed (LoRA weights were saved): {e}', flush=True)
                
                # Log success
                try:
                    with open(log_file, 'a') as f:
                        f.write(f'{datetime.datetime.now()}: Checkpoint saved successfully to {checkpoint_path}\n')
                except:
                    pass
            
        except Exception as e:
            print(f'ERROR saving checkpoint: {e}', flush=True)
            import traceback
            traceback.print_exc()
            try:
                with open(log_file, 'a') as f:
                    f.write(f'{datetime.datetime.now()}: ERROR: {e}\n')
            except:
                pass
        finally:
            # Ensure we exit cleanly
            print('=== Exiting gracefully ===', flush=True)
            sys.stdout.flush()
            sys.stderr.flush()
            sys.exit(0)

    signal.signal(signal.SIGTERM, save_checkpoint)
    signal.signal(signal.SIGINT, save_checkpoint)
    signal.signal(signal.SIGHUP, save_checkpoint)  # Some cloud providers send SIGHUP

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        train_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            models_to_accumulate = [transformer]

            target_images = batch["target_image"]
            start_images = batch["start_image"]
            style_images = batch.get("style_image", None)

            prompt_embeds = precomputed_prompt_embeds.repeat(target_images.shape[0], 1, 1)
            text_ids = precomputed_text_ids.repeat(target_images.shape[0], 1, 1)
            
            with accelerator.accumulate(models_to_accumulate):

                with torch.no_grad():
                    with offload_models(vae, device=accelerator.device, offload=args.offload):
                        # encode target image using VAE (it's model input)
                        pixel_values = target_images.to(dtype=weight_dtype).to(accelerator.device)
                        model_input = vae.encode(pixel_values).latent_dist.mode()

                        # encode control image(s): content (+ optional style)
                        pixel_start_values = start_images.to(dtype=weight_dtype).to(accelerator.device)
                        content_latents = vae.encode(pixel_start_values).latent_dist.mode()

                        style_latents = None
                        if style_images is not None:
                            pixel_style_values = style_images.to(dtype=weight_dtype).to(accelerator.device)
                            style_latents = vae.encode(pixel_style_values).latent_dist.mode()

                    # normalize model input
                    model_input = Flux2KleinPipeline._patchify_latents(model_input)
                    model_input = (model_input - latents_bn_mean) / latents_bn_std

                    content_latents = Flux2KleinPipeline._patchify_latents(content_latents)
                    content_latents = (content_latents - latents_bn_mean) / latents_bn_std

                    model_input_ids = Flux2KleinPipeline._prepare_latent_ids(model_input).to(device=model_input.device)

                    if style_latents is not None:
                        style_latents = Flux2KleinPipeline._patchify_latents(style_latents)
                        style_latents = (style_latents - latents_bn_mean) / latents_bn_std

                        # Per-sample content(T=10) + style(T=20); same pattern for every batch row
                        cond_ids_list = []
                        for i in range(content_latents.shape[0]):
                            ids_i = Flux2KleinPipeline._prepare_image_ids(
                                [content_latents[i].unsqueeze(0), style_latents[i].unsqueeze(0)]
                            )
                            cond_ids_list.append(ids_i.squeeze(0))
                        cond_model_input_ids = torch.stack(cond_ids_list, dim=0).to(device=content_latents.device)

                        packed_content = Flux2KleinPipeline._pack_latents(content_latents)
                        packed_style = Flux2KleinPipeline._pack_latents(style_latents)
                        packed_cond_model_input = torch.cat([packed_content, packed_style], dim=1)
                    else:
                        # Single-control path (object removal)
                        cond_model_input = content_latents
                        cond_model_input_list = [
                            cond_model_input[i].unsqueeze(0) for i in range(cond_model_input.shape[0])
                        ]
                        cond_model_input_ids = Flux2KleinPipeline._prepare_image_ids(cond_model_input_list).to(
                            device=cond_model_input.device
                        )
                        cond_model_input_ids = cond_model_input_ids.view(
                            cond_model_input.shape[0], -1, model_input_ids.shape[-1]
                        )
                        packed_cond_model_input = Flux2KleinPipeline._pack_latents(cond_model_input)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]

                # Sample a random timestep for each image
                # for weighting schemes where we sample timesteps non-uniformly
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)

                # Add noise according to flow matching.
                # zt = (1 - texp) * x + texp * z1
                sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

                # [B, C, H, W] -> [B, H*W, C]
                # concatenate the model inputs with the cond inputs
                packed_noisy_model_input = Flux2KleinPipeline._pack_latents(noisy_model_input)
                orig_input_shape = packed_noisy_model_input.shape
                orig_input_ids_shape = model_input_ids.shape

                # concatenate the model inputs with the cond inputs
                packed_noisy_model_input = torch.cat([packed_noisy_model_input, packed_cond_model_input], dim=1)
                model_input_ids = torch.cat([model_input_ids, cond_model_input_ids], dim=1)

                # handle guidance (unwrap: DDP has no .config)
                if unwrap_model(transformer).config.guidance_embeds:
                    guidance = torch.full([1], args.guidance_scale, device=accelerator.device)
                    guidance = guidance.expand(model_input.shape[0])
                else:
                    guidance = None

                # Predict the noise residual
                model_pred = transformer(
                    hidden_states=packed_noisy_model_input,  # (B, image_seq_len, C)
                    timestep=timesteps / 1000,
                    guidance=guidance,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,  # B, text_seq_len, 4
                    img_ids=model_input_ids,  # B, image_seq_len, 4
                    return_dict=False,
                )[0]
                # pruning the condition information
                model_pred = model_pred[:, : orig_input_shape[1], :]
                model_input_ids = model_input_ids[:, : orig_input_ids_shape[1], :]

                model_pred = Flux2KleinPipeline._unpack_latents_with_ids(model_pred, model_input_ids)

                # these weighting schemes use a uniform timestep sampling
                # and instead post-weight the loss
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)

                # flow matching loss
                target = noise - model_input

                # Compute regular loss.
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(target_images.shape[0])).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(transformer_lora_parameters, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                accelerator.log({"train_loss": train_loss, "lr": lr_scheduler.get_last_lr()[0]}, step=global_step)
                global_step += 1
                train_loss = 0.0

                # Checkpoint: all ranks must enter save_state together under DDP
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process and args.checkpoints_total_limit is not None:
                        checkpoints = os.listdir(args.output_dir)
                        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                        if len(checkpoints) >= args.checkpoints_total_limit:
                            num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                            removing_checkpoints = checkpoints[0:num_to_remove]

                            logger.info(
                                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                            )
                            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                            for removing_checkpoint in removing_checkpoints:
                                removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                shutil.rmtree(removing_checkpoint)

                    accelerator.wait_for_everyone()
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")
                    accelerator.wait_for_everyone()

                # Validation only on main rank; other ranks must wait or NCCL desyncs
                if global_step % args.validation_steps == 0:
                    if accelerator.is_main_process:
                        pipeline = Flux2KleinPipeline.from_pretrained(
                            args.pretrained_model_name_or_path,
                            tokenizer=None,
                            text_encoder=None,
                            transformer=accelerator.unwrap_model(transformer),
                            vae=vae,
                            revision=args.revision,
                            variant=args.variant,
                            torch_dtype=weight_dtype,
                            )
                        log_validation(
                            pipeline=pipeline,
                            args=args,
                            accelerator=accelerator,
                            step=global_step,
                            torch_dtype=weight_dtype,
                            pipeline_args=validation_kwargs,
                        )
                        del pipeline
                        free_memory()
                    accelerator.wait_for_everyone()

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break


    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        modules_to_save = {}
        transformer = unwrap_model(transformer)
        transformer_lora_layers = get_peft_model_state_dict(transformer)
        modules_to_save["transformer"] = transformer

        Flux2KleinPipeline.save_lora_weights(
            save_directory=args.output_dir,
            transformer_lora_layers=transformer_lora_layers,
            **_collate_lora_metadata(modules_to_save),
        )

        pipeline = Flux2KleinPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            tokenizer=None,
            text_encoder=None,
            transformer=accelerator.unwrap_model(transformer),
            vae=vae,
            revision=args.revision,
            variant=args.variant,
            torch_dtype=weight_dtype,
        )
        log_validation(
            pipeline=pipeline,
            args=args,
            accelerator=accelerator,
            step=global_step,
            torch_dtype=weight_dtype,
            pipeline_args=validation_kwargs,
        )
        del pipeline
        free_memory()

    accelerator.end_training()


if __name__ == "__main__":
    main()