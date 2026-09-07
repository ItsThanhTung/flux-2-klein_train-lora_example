from __future__ import annotations

import argparse
import contextlib
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from diffusers import Flux2KleinPipeline
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params, compute_loss_weighting_for_sd3
from omegaconf import OmegaConf
from peft.utils import get_peft_model_state_dict
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from datasets.object_effect_stripe_dataset import (
    ObjectEffectStripeDataset,
    create_mask_overlayed_image_stripes,
)
from datasets.prob_picking_dataset import ProbPickingDataset
from klein_dmd2_utils import (
    adapter_parameters,
    add_lora_adapter,
    decode_images,
    default_lora_targets,
    ema_update_adapter,
    encode_batch,
    encode_negative_prompt,
    expand_to_ndim,
    get_sigmas,
    prepare_klein_training_noise_scheduler,
    prepare_student_training_sample,
    predict_student_x0_from_noisy,
    sample_fake_timesteps,
    sample_final_latents,
    set_adapter,
    set_discriminator_heads_grad,
    teacher_latent_losses,
    transformer_cfg_score,
    transformer_score,
    unwrap_module,
)


DEFAULT_CONFIG = "configs/distill_dmd2_lora_removal_stripe.yaml"
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

DEFAULTS: dict[str, Any] = {
    "project_root": ".",
    "teacher_model": "black-forest-labs/FLUX.2-klein-base-9B",
    "student_model": "black-forest-labs/FLUX.2-klein-9B",
    "cache_dir": "cache_huggingface",
    "teacher_lora_path": "exps/train_lora_removal_stripe_multi-data_260130_r64",
    "student_lora_init_path": None,
    "output_dir": "exps/distill_dmd2_lora_removal_stripe_4step",
    "report_to": "wandb",
    "tracker_project_name": "flux2_klein_dmd2_lora_object_removal",
    "resolution": 1024,
    "train_batch_size": 1,
    "dataloader_num_workers": 8,
    "max_train_steps": 400,
    "gradient_accumulation_steps": 8,
    "learning_rate": 5e-5,
    "fake_learning_rate": 5e-5,
    "lr_scheduler": "constant_with_warmup",
    "lr_warmup_steps": 50,
    "rank": 64,
    "lora_alpha": 64,
    "lora_dropout": 0.0,
    "lora_layers": None,
    "student_num_inference_steps": 4,
    "dmd2_weight": 1.0,
    "teacher_latent_l1_weight": 0.0,
    "teacher_latent_l2_weight": 0.0,
    "teacher_latent_sample_steps": 30,
    "lpips_weight": 0.0,
    "gan_weight": 0.0,
    "discriminator_learning_rate": 5e-5,
    "no_gan_disc_pretrain": True,
    "fake_score_weight": 1.0,
    "fake_initialize": "student",
    "fake_num_updates": 3,
    "fake_critic_pretrain_steps": 200,
    "fake_time_sampler": "uniform_high",
    "fake_time_min": 600,
    "fake_time_max": 1000,
    "ida_lambda": 0.97,
    "teacher_guidance_scale": 5.0,
    "weighting_scheme": "none",
    "logit_mean": 0.0,
    "logit_std": 1.0,
    "mode_scale": 1.29,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_weight_decay": 1e-4,
    "adam_epsilon": 1e-8,
    "max_grad_norm": 1.0,
    "mixed_precision": "bf16",
    "seed": 2026,
    "checkpointing_steps": 50,
    "checkpointing_steps_after_fake_critic_pretrain": 25,
    "gradient_checkpointing": True,
    "allow_tf32": True,
    "dry_run": False,
    "max_samples": 0,
    "repeats": 1,
    "model_prompt": None,
    "prompt_file": None,
    "validation_steps": 50,
    "validation_num_inference_steps": 4,
    "validation_guidance_scale": 1.0,
    "validation_data": {"cases": []},
    "train_datasets": [],
}


def is_empty(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip().lower() in {"", "none", "null"})


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--teacher-model", default=None)
    parser.add_argument("--student-model", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--teacher-lora-path", default=None)
    parser.add_argument("--student-lora-init-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--report-to", default=None)
    parser.add_argument("--tracker-project-name", default=None)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--dataloader-num-workers", type=int, default=None)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--fake-learning-rate", type=float, default=None)
    parser.add_argument("--lr-scheduler", default=None)
    parser.add_argument("--lr-warmup-steps", type=int, default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--lora-dropout", type=float, default=None)
    parser.add_argument("--lora-layers", default=None)
    parser.add_argument("--student-num-inference-steps", type=int, default=None)
    parser.add_argument("--dmd2-weight", type=float, default=None)
    parser.add_argument("--teacher-latent-l1-weight", type=float, default=None)
    parser.add_argument("--teacher-latent-l2-weight", type=float, default=None)
    parser.add_argument("--teacher-latent-sample-steps", type=int, default=None)
    parser.add_argument("--lpips-weight", type=float, default=None)
    parser.add_argument("--gan-weight", type=float, default=None)
    parser.add_argument("--discriminator-learning-rate", type=float, default=None)
    parser.add_argument("--fake-score-weight", type=float, default=None)
    parser.add_argument("--fake-initialize", choices=["student", "teacher"], default=None)
    parser.add_argument("--fake-num-updates", type=int, default=None)
    parser.add_argument("--fake-critic-pretrain-steps", type=int, default=None)
    parser.add_argument("--fake-time-sampler", choices=["infer", "uniform_high", "logit_normal"], default=None)
    parser.add_argument("--fake-time-min", type=int, default=None)
    parser.add_argument("--fake-time-max", type=int, default=None)
    parser.add_argument("--ida-lambda", type=float, default=None)
    parser.add_argument("--teacher-guidance-scale", type=float, default=None)
    parser.add_argument(
        "--weighting-scheme",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        default=None,
    )
    parser.add_argument("--logit-mean", type=float, default=None)
    parser.add_argument("--logit-std", type=float, default=None)
    parser.add_argument("--mode-scale", type=float, default=None)
    parser.add_argument("--adam-beta1", type=float, default=None)
    parser.add_argument("--adam-beta2", type=float, default=None)
    parser.add_argument("--adam-weight-decay", type=float, default=None)
    parser.add_argument("--adam-epsilon", type=float, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--checkpointing-steps", type=int, default=None)
    parser.add_argument("--checkpointing-steps-after-fake-critic-pretrain", type=int, default=None)
    parser.add_argument("--model-prompt", default=None)
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("--validation-steps", type=int, default=None)
    parser.add_argument("--validation-num-inference-steps", type=int, default=None)
    parser.add_argument("--validation-guidance-scale", type=float, default=None)
    parser.add_argument("--no-gan-disc-pretrain", dest="no_gan_disc_pretrain", action="store_true", default=None)
    parser.add_argument("--gan-disc-pretrain", dest="no_gan_disc_pretrain", action="store_false")
    parser.add_argument("--gradient-checkpointing", dest="gradient_checkpointing", action="store_true", default=None)
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.add_argument("--allow-tf32", dest="allow_tf32", action="store_true", default=None)
    parser.add_argument("--no-allow-tf32", dest="allow_tf32", action="store_false")
    parser.add_argument("--dry-run", action="store_true", default=None)


def parse_args(input_args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distill a FLUX.2 Klein object-removal LoRA with DMD2.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    add_common_args(parser)
    cli_args = parser.parse_args(input_args)

    merged = dict(DEFAULTS)
    if cli_args.config:
        config_path = Path(cli_args.config).expanduser()
        config_data = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
        if config_data:
            merged = deep_merge(merged, config_data)
    overrides = {
        key: value
        for key, value in vars(cli_args).items()
        if key != "config" and value is not None
    }
    merged = deep_merge(merged, overrides)
    merged["config"] = cli_args.config
    merged["lora_alpha"] = merged["lora_alpha"] or merged["rank"]
    for key in ("teacher_lora_path", "student_lora_init_path", "cache_dir", "prompt_file"):
        if is_empty(merged[key]):
            merged[key] = None
    if is_empty(merged["report_to"]) or str(merged["report_to"]).lower() == "none":
        merged["report_to"] = None
    return argparse.Namespace(**merged)


def resolve_path(path: str | Path, project_root: Path) -> Path:
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def resolve_optional_path(path: str | Path | None, project_root: Path) -> Path | None:
    return None if is_empty(path) else resolve_path(path, project_root)


def read_prompt(args: argparse.Namespace, project_root: Path) -> str:
    if not is_empty(args.model_prompt):
        return str(args.model_prompt).strip()
    prompt_path = resolve_optional_path(args.prompt_file, project_root)
    if prompt_path is not None and prompt_path.is_file():
        prompt = " ".join(prompt_path.read_text(encoding="utf-8").split())
        if prompt:
            return prompt
    raise ValueError("A non-empty model_prompt or readable prompt_file is required.")


class DMDContractAdapter(Dataset):
    """Map the existing paired dataset output to the DMD2 batch contract."""

    def __init__(self, dataset: ObjectEffectStripeDataset, prompt: str, label: str):
        self.dataset = dataset
        self.prompt = prompt
        self.label = label

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.dataset[index]
        source = self.dataset.data[index]
        target_path = str(source.get("gt_path", f"{self.label}-{index}"))
        return {
            "pixel_values": item["target_image"],
            "cond_pixel_values": item["start_image"].unsqueeze(0),
            "prompt": self.prompt,
            "name": f"{self.label}:{target_path}",
        }


def build_dataset(args: argparse.Namespace, project_root: Path, prompt: str) -> Dataset:
    configs = list(args.train_datasets or [])
    if not configs:
        raise ValueError("train_datasets must contain at least one dataset definition.")
    datasets = []
    probabilities = []
    for index, config in enumerate(configs):
        dataset_type = str(config.get("type", ""))
        if dataset_type != "ObjectEffectStripeDataset":
            raise ValueError(f"Unsupported DMD2 dataset type: {dataset_type!r}")
        kwargs = dict(config.get("kwargs", {}))
        kwargs["data_root"] = str(resolve_path(kwargs["data_root"], project_root))
        kwargs["json_file_path"] = str(resolve_path(kwargs["json_file_path"], project_root))
        kwargs.setdefault("out_size", args.resolution)
        dataset = ObjectEffectStripeDataset(**kwargs)
        datasets.append(DMDContractAdapter(dataset, prompt, f"dataset-{index}"))
        probabilities.append(float(config.get("sample_prob", 0.0)))
    if any(probability < 0 for probability in probabilities):
        raise ValueError("Dataset sample probabilities cannot be negative.")
    if not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"Dataset sample probabilities must sum to 1.0; got {sum(probabilities):.8f}.")
    base_length = sum(len(dataset) for dataset in datasets)
    if args.max_samples and args.max_samples > 0:
        base_length = min(base_length, int(args.max_samples))
    length = max(1, base_length * max(1, int(args.repeats)))
    return ProbPickingDataset(datasets=datasets, probs=probabilities, length=length)


def collate_fn(examples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "pixel_values": torch.stack([example["pixel_values"] for example in examples]).float(),
        "cond_pixel_values": torch.stack([example["cond_pixel_values"] for example in examples]).float(),
        "prompts": [example["prompt"] for example in examples],
        "names": [example["name"] for example in examples],
    }


def list_images(root: Path) -> list[Path]:
    return sorted(path for path in root.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES)


def matching_image(root: Path, stem: str) -> Path:
    for suffix in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
        candidate = root / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    return root / f"{stem}.png"


def resolve_validation_cases(args: argparse.Namespace, project_root: Path) -> list[dict[str, Any]]:
    if not args.validation_steps or args.validation_steps <= 0:
        return []
    validation_data = args.validation_data or {}
    resolved = []
    for case in validation_data.get("cases", []):
        image_root = resolve_path(case["image_root"], project_root)
        mask_root = resolve_path(case["mask_root"], project_root)
        if not image_root.is_dir():
            raise FileNotFoundError(f"Validation image_root does not exist: {image_root}")
        if not mask_root.is_dir():
            raise FileNotFoundError(f"Validation mask_root does not exist: {mask_root}")
        image_paths = list_images(image_root)
        if case.get("num_samples") is not None:
            image_paths = image_paths[: int(case["num_samples"])]
        if not image_paths:
            raise ValueError(f"No validation images found in {image_root}")
        for image_path in image_paths:
            mask_path = matching_image(mask_root, image_path.stem)
            if not mask_path.is_file():
                raise FileNotFoundError(f"Validation mask does not exist: {mask_path}")
        resolved.append({"image_paths": image_paths, "mask_root": mask_root})
    return resolved


def resize_target_area(image: Image.Image, target_area: int = 1024 * 1024) -> Image.Image:
    width, height = image.size
    scale = min(1.0, math.sqrt(target_area / (width * height)))
    new_width = max(32, int(width * scale) // 32 * 32)
    new_height = max(32, int(height * scale) // 32 * 32)
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def validation_autocast(device_type: str, weight_dtype: torch.dtype):
    if device_type == "cuda" and weight_dtype in {torch.float16, torch.bfloat16}:
        return torch.autocast("cuda", dtype=weight_dtype)
    return contextlib.nullcontext()


@torch.no_grad()
def run_validation(
    student_pipe,
    args: argparse.Namespace,
    accelerator: Accelerator,
    output_dir: Path,
    validation_cases: list[dict[str, Any]],
    prompt: str,
    weight_dtype: torch.dtype,
    global_step: int,
) -> None:
    if not validation_cases:
        return
    first_dataset = args.train_datasets[0]
    overlay_kwargs = first_dataset.get("kwargs", {})
    opacity = float(overlay_kwargs.get("mask_overlay_opacity", 0.5))
    stripe_width = int(overlay_kwargs.get("stripe_width", 12))
    validation_dir = output_dir / "validation" / f"step-{global_step}"
    validation_dir.mkdir(parents=True, exist_ok=True)

    wrapped_transformer = student_pipe.transformer
    transformer = accelerator.unwrap_model(wrapped_transformer)
    was_training = transformer.training
    student_pipe.transformer = transformer
    set_adapter(student_pipe.transformer, "student")
    student_pipe.transformer.eval()
    student_pipe.set_progress_bar_config(disable=True)
    generator = (
        torch.Generator(device=accelerator.device).manual_seed(int(args.seed))
        if args.seed is not None
        else None
    )
    image_logs = []
    try:
        for case_index, case in enumerate(validation_cases):
            for image_index, image_path in enumerate(case["image_paths"]):
                source = Image.open(image_path).convert("RGB")
                mask_path = matching_image(case["mask_root"], image_path.stem)
                mask_image = Image.open(mask_path)
                mask_array = np.array(mask_image.convert("L")) if mask_image.mode == "L" else np.array(mask_image)[..., -1]
                mask_array = (mask_array > 0).astype(np.uint8)
                overlay = create_mask_overlayed_image_stripes(
                    np.array(source), mask_array, opacity=opacity, stripe_width=stripe_width
                )
                condition = resize_target_area(Image.fromarray(overlay))
                with validation_autocast(accelerator.device.type, weight_dtype):
                    output = student_pipe(
                        prompt=prompt,
                        image=condition,
                        generator=generator,
                        num_inference_steps=args.validation_num_inference_steps,
                        guidance_scale=args.validation_guidance_scale,
                    ).images[0]
                output = output.resize(source.size, Image.Resampling.LANCZOS)
                display_condition = Image.fromarray(overlay).resize(source.size)
                grid = Image.new("RGB", (source.width * 2, source.height))
                grid.paste(display_condition, (0, 0))
                grid.paste(output, (source.width, 0))
                save_path = validation_dir / f"{case_index:02d}_{image_index:02d}_{image_path.stem}.png"
                grid.save(save_path)
                image_logs.append((grid, save_path))
    finally:
        if was_training:
            student_pipe.transformer.train()
        student_pipe.transformer = wrapped_transformer

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            tracker.writer.add_images(
                "validation", np.stack([np.asarray(image) for image, _ in image_logs]), global_step, dataformats="NHWC"
            )
        elif tracker.name == "wandb":
            import wandb

            tracker.log(
                {"validation": [wandb.Image(str(path), caption=path.name) for _, path in image_logs]},
                step=global_step,
            )


def pipeline_from_pretrained(model_id: str, weight_dtype: torch.dtype, cache_dir: str | None):
    kwargs: dict[str, Any] = {"torch_dtype": weight_dtype}
    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir
    return Flux2KleinPipeline.from_pretrained(model_id, **kwargs)


def validate_args(args: argparse.Namespace, project_root: Path) -> tuple[Path, Path | None]:
    teacher_path = resolve_optional_path(args.teacher_lora_path, project_root)
    student_path = resolve_optional_path(args.student_lora_init_path, project_root)
    if teacher_path is None:
        raise ValueError("teacher_lora_path is required for task-specific distillation.")
    if not teacher_path.exists():
        raise FileNotFoundError(f"teacher_lora_path does not exist: {teacher_path}")
    if student_path is None:
        student_path = teacher_path
    if not student_path.exists():
        raise FileNotFoundError(f"student_lora_init_path does not exist: {student_path}")
    if args.fake_initialize == "teacher" and teacher_path is None:
        raise ValueError("fake_initialize=teacher requires teacher_lora_path.")
    if args.student_num_inference_steps < 1:
        raise ValueError("student_num_inference_steps must be positive.")
    if args.teacher_latent_sample_steps < 1:
        raise ValueError("teacher_latent_sample_steps must be positive.")
    loss_weights = (
        args.dmd2_weight,
        args.teacher_latent_l1_weight,
        args.teacher_latent_l2_weight,
        args.lpips_weight,
        args.gan_weight,
    )
    if any(weight < 0 for weight in (*loss_weights, args.fake_score_weight)):
        raise ValueError("Loss weights cannot be negative.")
    if not any(weight > 0 for weight in loss_weights):
        raise ValueError("At least one student loss must have a positive weight.")
    if args.fake_critic_pretrain_steps > 0 and args.fake_score_weight <= 0:
        raise ValueError("fake_critic_pretrain_steps requires fake_score_weight > 0.")
    return teacher_path, student_path


def tracker_config(args: argparse.Namespace) -> dict[str, Any]:
    return OmegaConf.to_container(OmegaConf.create(vars(args)), resolve=True)


def save_student_lora(student_pipe, accelerator: Accelerator, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    transformer = accelerator.unwrap_model(student_pipe.transformer)
    set_adapter(transformer, "student")
    state = get_peft_model_state_dict(transformer, adapter_name="student")
    Flux2KleinPipeline.save_lora_weights(str(destination), transformer_lora_layers=state)


def main(args: argparse.Namespace) -> None:
    project_root = resolve_path(args.project_root, REPO_ROOT) if args.project_root != "." else REPO_ROOT
    output_dir = resolve_path(args.output_dir, project_root)
    cache_path = resolve_optional_path(args.cache_dir, project_root)
    cache_dir = str(cache_path) if cache_path is not None else None
    teacher_lora_path, student_lora_init_path = validate_args(args, project_root)
    prompt = read_prompt(args, project_root)
    validation_cases = resolve_validation_cases(args, project_root)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=None if args.mixed_precision == "no" else args.mixed_precision,
        log_with=args.report_to,
        project_config=ProjectConfiguration(project_dir=str(output_dir), logging_dir=str(output_dir / "logs")),
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    if args.seed is not None:
        set_seed(args.seed)
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    dataset = build_dataset(args, project_root, prompt)
    dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
        drop_last=True,
    )
    if args.dry_run:
        batch = next(iter(dataloader))
        print(f"Loaded {len(dataset)} weighted object-removal examples")
        print(f"target batch: {tuple(batch['pixel_values'].shape)}")
        print(f"condition batch: {tuple(batch['cond_pixel_values'].shape)}")
        print(f"first file: {batch['names'][0]}")
        print(f"prompt: {batch['prompts'][0]}")
        print(f"teacher_model={args.teacher_model}")
        print(f"student_model={args.student_model}")
        print(f"teacher_lora={teacher_lora_path}")
        print(f"student_lora_init={student_lora_init_path}")
        print(f"student_steps={args.student_num_inference_steps}")
        print(f"teacher_latent_l1_weight={args.teacher_latent_l1_weight}")
        print(f"teacher_latent_l2_weight={args.teacher_latent_l2_weight}")
        print(f"gan_weight={args.gan_weight}")
        print(f"validation_images={sum(len(case['image_paths']) for case in validation_cases)}")
        return

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    with accelerator.main_process_first():
        student_pipe = pipeline_from_pretrained(args.student_model, weight_dtype, cache_dir)
        teacher_pipe = pipeline_from_pretrained(args.teacher_model, weight_dtype, cache_dir)
        fake_pipe = None
        teacher_pipe.load_lora_weights(str(teacher_lora_path), adapter_name="teacher_removal")
        teacher_pipe.set_adapters(["teacher_removal"], adapter_weights=[1.0])
        student_pipe.load_lora_weights(str(student_lora_init_path), adapter_name="student")
        if args.fake_initialize == "student":
            student_pipe.load_lora_weights(str(student_lora_init_path), adapter_name="fake_score")
        else:
            fake_pipe = pipeline_from_pretrained(args.teacher_model, weight_dtype, cache_dir)
            fake_pipe.load_lora_weights(str(teacher_lora_path), adapter_name="fake_score")

    student_pipe.to(accelerator.device)
    teacher_pipe.to(accelerator.device)
    if fake_pipe is not None:
        fake_pipe.to(accelerator.device)
    for module in (teacher_pipe.transformer, teacher_pipe.vae, teacher_pipe.text_encoder):
        module.requires_grad_(False)
        module.eval()
    for module in (student_pipe.transformer, student_pipe.vae, student_pipe.text_encoder):
        module.requires_grad_(False)
    if fake_pipe is not None:
        for module in (fake_pipe.transformer, fake_pipe.vae, fake_pipe.text_encoder):
            module.requires_grad_(False)
        fake_pipe.vae.eval()
        fake_pipe.text_encoder.eval()

    if args.gradient_checkpointing:
        student_pipe.transformer.enable_gradient_checkpointing()
        if fake_pipe is not None:
            fake_pipe.transformer.enable_gradient_checkpointing()

    targets = [item.strip() for item in args.lora_layers.split(",")] if args.lora_layers else default_lora_targets()
    if "student" not in getattr(student_pipe.transformer, "peft_config", {}):
        add_lora_adapter(student_pipe.transformer, "student", args.rank, args.lora_alpha, args.lora_dropout, targets)
    fake_transformer = fake_pipe.transformer if fake_pipe is not None else student_pipe.transformer
    if "fake_score" not in getattr(fake_transformer, "peft_config", {}):
        add_lora_adapter(fake_transformer, "fake_score", args.rank, args.lora_alpha, args.lora_dropout, targets)
    if accelerator.mixed_precision == "fp16":
        cast_modules = [student_pipe.transformer]
        if fake_pipe is not None:
            cast_modules.append(fake_pipe.transformer)
        cast_training_params(cast_modules, dtype=torch.float32)

    student_parameters = adapter_parameters(student_pipe.transformer, "student")
    fake_parameters = adapter_parameters(fake_transformer, "fake_score")
    if not student_parameters or not fake_parameters:
        raise RuntimeError("Student and fake_score adapters must both expose trainable LoRA parameters.")
    optimizer_student = torch.optim.AdamW(
        student_parameters,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    optimizer_fake = torch.optim.AdamW(
        fake_parameters,
        lr=args.fake_learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    lpips_loss_fn = None
    if args.lpips_weight > 0:
        try:
            import lpips
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("Install requirements_distill.txt or set lpips_weight: 0.") from exc
        lpips_loss_fn = lpips.LPIPS(net="alex").to(accelerator.device).eval().requires_grad_(False)

    discriminator = None
    clip_model = None
    gan_loss_fn = None
    optimizer_discriminator = None
    if args.gan_weight > 0:
        try:
            from senseflow.models.clip import CLIP
            from senseflow.models.vfmgan import GANLoss, ProjectedDiscriminatorPlus
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("Install requirements_distill.txt or set gan_weight: 0.") from exc
        discriminator = ProjectedDiscriminatorPlus(
            c_dim=768,
            dino_name="vit_large_patch14_dinov2.lvd142m",
            hooks=[2, 5, 8, 10, 14, 19, 23],
            crop_plan="none",
            p_crop=0.5,
            use_checkpoint=False,
            fix_res_dino=False,
            useatt=False,
            ret_cls=False,
            conv2d=True,
            downsample=3,
            diffaug=True,
            dino_pretrain=True,
        ).to(accelerator.device)
        discriminator.train()
        clip_model = CLIP().to(accelerator.device).eval().requires_grad_(False)
        gan_loss_fn = GANLoss(gan_type="hinge", real_label_val=1.0, fake_label_val=0.0, loss_weight=1.0)
        optimizer_discriminator = torch.optim.AdamW(
            discriminator.heads.parameters(),
            lr=args.discriminator_learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    update_steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    train_epochs = math.ceil(args.max_train_steps / update_steps_per_epoch)
    scheduler_student = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer_student,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )
    scheduler_fake = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer_fake,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps * max(1, args.fake_num_updates),
    )
    scheduler_discriminator = None
    if optimizer_discriminator is not None:
        scheduler_discriminator = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer_discriminator,
            num_warmup_steps=args.lr_warmup_steps,
            num_training_steps=args.max_train_steps,
        )

    if fake_pipe is None:
        (
            prepared_transformer,
            optimizer_student,
            optimizer_fake,
            dataloader,
            scheduler_student,
            scheduler_fake,
        ) = accelerator.prepare(
            student_pipe.transformer,
            optimizer_student,
            optimizer_fake,
            dataloader,
            scheduler_student,
            scheduler_fake,
        )
        student_pipe.transformer = prepared_transformer
    else:
        (
            prepared_transformer,
            prepared_fake_transformer,
            optimizer_student,
            optimizer_fake,
            dataloader,
            scheduler_student,
            scheduler_fake,
        ) = accelerator.prepare(
            student_pipe.transformer,
            fake_pipe.transformer,
            optimizer_student,
            optimizer_fake,
            dataloader,
            scheduler_student,
            scheduler_fake,
        )
        student_pipe.transformer = prepared_transformer
        fake_pipe.transformer = prepared_fake_transformer
    if discriminator is not None:
        discriminator, optimizer_discriminator, scheduler_discriminator = accelerator.prepare(
            discriminator, optimizer_discriminator, scheduler_discriminator
        )

    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(OmegaConf.create(vars(args)), str(output_dir / "distillation_config.yaml"))
        if args.report_to:
            accelerator.init_trackers(args.tracker_project_name, config=tracker_config(args))

    noise_scheduler = None
    progress = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process, desc="Steps")
    global_step = 0
    fake_accumulated: list[dict[str, Any]] = []
    gan_accumulated: list[dict[str, Any]] = []

    for _epoch in range(train_epochs):
        student_pipe.transformer.train()
        if fake_pipe is not None:
            fake_pipe.transformer.train()
        for batch in dataloader:
            fake_transformer = fake_pipe.transformer if fake_pipe is not None else student_pipe.transformer
            accumulation_models = (
                (student_pipe.transformer,)
                if fake_pipe is None
                else (student_pipe.transformer, fake_pipe.transformer)
            )
            with accelerator.accumulate(*accumulation_models):
                prompt_embeds, text_ids, target_latents, condition_latents = encode_batch(
                    student_pipe, batch, accelerator, weight_dtype
                )
                batch_size = target_latents.shape[0]
                fake_pretrain = global_step < args.fake_critic_pretrain_steps
                if noise_scheduler is None:
                    noise_scheduler = prepare_klein_training_noise_scheduler(
                        teacher_pipe.scheduler,
                        target_latents,
                        args.student_num_inference_steps,
                        accelerator.device,
                    )

                set_adapter(student_pipe.transformer, "student")
                input_noise = torch.randn_like(target_latents)
                student_context = torch.no_grad() if fake_pretrain else contextlib.nullcontext()
                with student_context:
                    noisy_input, generator_timesteps, generator_sigmas = prepare_student_training_sample(
                        student_pipe.transformer,
                        input_noise,
                        condition_latents,
                        prompt_embeds,
                        text_ids,
                        noise_scheduler,
                        args.student_num_inference_steps,
                        weight_dtype,
                    )
                    student_x0 = predict_student_x0_from_noisy(
                        student_pipe.transformer,
                        noisy_input,
                        condition_latents,
                        prompt_embeds,
                        text_ids,
                        generator_timesteps,
                        generator_sigmas,
                        weight_dtype,
                    )

                zero = torch.zeros((), device=accelerator.device)
                dmd2_loss = zero
                latent_l1_loss = zero
                latent_l2_loss = zero
                fake_loss = zero
                lpips_loss = zero
                gan_loss = zero
                discriminator_real_loss = zero
                discriminator_fake_loss = zero

                t1_sigmas = expand_to_ndim(generator_sigmas.to(student_x0.device, student_x0.dtype), student_x0.ndim)
                t1_noise = torch.randn_like(student_x0)
                noisy_student_t1 = ((1.0 - t1_sigmas) * student_x0 + t1_sigmas * t1_noise).to(weight_dtype)

                needs_teacher_target = args.teacher_latent_l1_weight > 0 or args.teacher_latent_l2_weight > 0
                if not fake_pretrain and (args.dmd2_weight > 0 or needs_teacher_target):
                    with torch.no_grad():
                        (
                            teacher_prompt_embeds,
                            teacher_text_ids,
                            _teacher_targets,
                            teacher_condition_latents,
                        ) = encode_batch(teacher_pipe, batch, accelerator, weight_dtype)
                        teacher_negative_embeds, teacher_negative_ids = encode_negative_prompt(
                            teacher_pipe, batch_size, accelerator, weight_dtype
                        )
                        if args.dmd2_weight > 0:
                            teacher_prediction = transformer_cfg_score(
                                teacher_pipe.transformer,
                                noisy_student_t1.detach(),
                                teacher_condition_latents,
                                teacher_prompt_embeds,
                                teacher_text_ids,
                                teacher_negative_embeds,
                                teacher_negative_ids,
                                generator_timesteps,
                                args.teacher_guidance_scale,
                                weight_dtype,
                            )
                            set_adapter(fake_transformer, "fake_score")
                            fake_prediction = transformer_score(
                                fake_transformer,
                                noisy_student_t1.detach(),
                                condition_latents,
                                prompt_embeds,
                                text_ids,
                                generator_timesteps,
                                weight_dtype,
                            )
                        if needs_teacher_target:
                            teacher_final_latents = sample_final_latents(
                                teacher_pipe.transformer,
                                input_noise.detach(),
                                teacher_condition_latents,
                                teacher_prompt_embeds,
                                teacher_text_ids,
                                teacher_pipe.scheduler,
                                args.teacher_latent_sample_steps,
                                weight_dtype,
                                guidance_scale=args.teacher_guidance_scale,
                                negative_prompt_embeds=teacher_negative_embeds,
                                negative_text_ids=teacher_negative_ids,
                            ).detach()

                if not fake_pretrain and args.dmd2_weight > 0:
                    predicted_real_x0 = noisy_student_t1.float() - t1_sigmas.float() * teacher_prediction.float()
                    predicted_fake_x0 = noisy_student_t1.float() - t1_sigmas.float() * fake_prediction.float()
                    real_residual = student_x0.float() - predicted_real_x0
                    fake_residual = student_x0.float() - predicted_fake_x0
                    reduce_dims = tuple(range(1, real_residual.ndim))
                    denominator = real_residual.abs().mean(dim=reduce_dims, keepdim=True).clamp_min(1e-6)
                    score_gradient = torch.nan_to_num((real_residual - fake_residual) / denominator)
                    dmd2_target = (student_x0 - score_gradient).detach()
                    dmd2_loss = 0.5 * F.mse_loss(student_x0.float(), dmd2_target.float())

                generator_loss = args.dmd2_weight * dmd2_loss
                if not fake_pretrain and needs_teacher_target:
                    set_adapter(student_pipe.transformer, "student")
                    student_final_latents = sample_final_latents(
                        student_pipe.transformer,
                        input_noise,
                        condition_latents,
                        prompt_embeds,
                        text_ids,
                        noise_scheduler,
                        args.student_num_inference_steps,
                        weight_dtype,
                    )
                    latent_total, latent_l1_loss, latent_l2_loss = teacher_latent_losses(
                        student_final_latents,
                        teacher_final_latents,
                        args.teacher_latent_l1_weight,
                        args.teacher_latent_l2_weight,
                    )
                    generator_loss = generator_loss + latent_total

                student_image = None
                if not fake_pretrain and lpips_loss_fn is not None:
                    student_image = decode_images(student_pipe.vae, student_x0, weight_dtype)
                    target_pixels = batch["pixel_values"].to(accelerator.device, dtype=weight_dtype)
                    lpips_loss = lpips_loss_fn(student_image.float(), target_pixels.float()).mean()
                    generator_loss = generator_loss + args.lpips_weight * lpips_loss

                train_gan = discriminator is not None and not (fake_pretrain and args.no_gan_disc_pretrain)
                if train_gan:
                    target_pixels = batch["pixel_values"].to(accelerator.device, dtype=weight_dtype)
                    with torch.no_grad():
                        clip_condition = clip_model.encode_text(batch["prompts"])
                    time_weight = ((1.0 - generator_sigmas.float()) ** 2).flatten()
                    if time_weight.numel() == 1:
                        time_weight = time_weight.expand(batch_size)
                    gan_accumulated.append(
                        {
                            "student_x0": student_x0.detach(),
                            "real_pixels": target_pixels.detach(),
                            "reference_pixels": target_pixels.detach(),
                            "clip_condition": clip_condition.detach(),
                            "time_weight": time_weight.detach(),
                        }
                    )
                    if not fake_pretrain:
                        set_discriminator_heads_grad(discriminator, False)
                        if student_image is None:
                            student_image = decode_images(student_pipe.vae, student_x0, weight_dtype)
                        prediction = discriminator(student_image.float(), clip_condition, target_pixels.float())
                        per_sample_loss = gan_loss_fn(prediction, True, is_disc=False, keepdim=True)
                        gan_loss = (per_sample_loss * time_weight.to(per_sample_loss)).mean()
                        generator_loss = generator_loss + args.gan_weight * gan_loss
                        set_discriminator_heads_grad(discriminator, True)

                if not fake_pretrain:
                    set_adapter(student_pipe.transformer, "student")
                    accelerator.backward(generator_loss)
                if args.fake_score_weight > 0:
                    fake_accumulated.append(
                        {
                            "student_x0": student_x0.detach(),
                            "condition_latents": condition_latents.detach(),
                            "prompt_embeds": prompt_embeds.detach(),
                            "text_ids": text_ids.detach(),
                            "batch_size": batch_size,
                        }
                    )

                if not fake_pretrain:
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(student_parameters, args.max_grad_norm)
                    optimizer_student.step()
                    scheduler_student.step()
                    optimizer_student.zero_grad()

                if accelerator.sync_gradients:
                    if not fake_pretrain and fake_pipe is None:
                        ema_update_adapter(student_pipe.transformer, "student", "fake_score", args.ida_lambda)
                    fake_updates = 1 if fake_pretrain else max(1, args.fake_num_updates)
                    for _ in range(fake_updates):
                        if args.fake_score_weight <= 0:
                            break
                        optimizer_fake.zero_grad()
                        for batch_index, saved in enumerate(fake_accumulated):
                            saved_x0 = saved["student_x0"]
                            fake_timesteps = sample_fake_timesteps(
                                args, noise_scheduler, saved["batch_size"], accelerator.device
                            )
                            fake_sigmas = get_sigmas(
                                noise_scheduler,
                                fake_timesteps,
                                saved_x0.ndim,
                                saved_x0.dtype,
                                accelerator.device,
                            )
                            noise = torch.randn_like(saved_x0)
                            noisy_fake = ((1.0 - fake_sigmas) * saved_x0 + fake_sigmas * noise).to(weight_dtype)
                            set_adapter(fake_transformer, "fake_score")
                            sync_context = (
                                contextlib.nullcontext()
                                if batch_index == len(fake_accumulated) - 1
                                else accelerator.no_sync(fake_transformer)
                            )
                            with sync_context:
                                prediction = transformer_score(
                                    fake_transformer,
                                    noisy_fake.detach(),
                                    saved["condition_latents"],
                                    saved["prompt_embeds"],
                                    saved["text_ids"],
                                    fake_timesteps,
                                    weight_dtype,
                                )
                                target = noise.detach() - saved_x0
                                weighting_scheme = (
                                    "logit_normal" if args.fake_time_sampler == "logit_normal" else args.weighting_scheme
                                )
                                weighting = compute_loss_weighting_for_sd3(
                                    weighting_scheme=weighting_scheme, sigmas=fake_sigmas
                                )
                                fake_loss = torch.mean(
                                    (weighting.float() * (prediction.float() - target.float()) ** 2).reshape(
                                        saved["batch_size"], -1
                                    ),
                                    dim=1,
                                ).mean()
                                accelerator.backward(args.fake_score_weight * fake_loss)
                        accelerator.clip_grad_norm_(fake_parameters, args.max_grad_norm)
                        optimizer_fake.step()
                        scheduler_fake.step()
                    optimizer_fake.zero_grad()
                    fake_accumulated.clear()

                    if discriminator is not None and gan_accumulated:
                        optimizer_discriminator.zero_grad()
                        for batch_index, saved in enumerate(gan_accumulated):
                            sync_context = (
                                contextlib.nullcontext()
                                if batch_index == len(gan_accumulated) - 1
                                else accelerator.no_sync(discriminator)
                            )
                            with sync_context:
                                with torch.no_grad():
                                    fake_pixels = decode_images(student_pipe.vae, saved["student_x0"], weight_dtype)
                                real_prediction = discriminator(
                                    saved["real_pixels"].float(),
                                    saved["clip_condition"],
                                    saved["reference_pixels"].float(),
                                )
                                real_per_sample = gan_loss_fn(real_prediction, True, is_disc=True, keepdim=True)
                                discriminator_real_loss = (
                                    real_per_sample * saved["time_weight"].to(real_per_sample)
                                ).mean()
                                accelerator.backward(discriminator_real_loss)
                                fake_prediction = discriminator(
                                    fake_pixels.detach().float(),
                                    saved["clip_condition"],
                                    saved["reference_pixels"].float(),
                                )
                                fake_per_sample = gan_loss_fn(fake_prediction, False, is_disc=True, keepdim=True)
                                discriminator_fake_loss = (
                                    fake_per_sample * saved["time_weight"].to(fake_per_sample)
                                ).mean()
                                accelerator.backward(discriminator_fake_loss)
                        accelerator.clip_grad_norm_(unwrap_module(discriminator).heads.parameters(), args.max_grad_norm)
                        optimizer_discriminator.step()
                        scheduler_discriminator.step()
                        optimizer_discriminator.zero_grad()
                        gan_accumulated.clear()
                optimizer_student.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                progress.update(1)
                phase = "fake_pretrain" if fake_pretrain else "dmd2"
                progress.set_postfix(
                    phase=phase,
                    dmd2=f"{dmd2_loss.item():.4f}",
                    l1=f"{latent_l1_loss.item():.4f}",
                    l2=f"{latent_l2_loss.item():.4f}",
                    fake=f"{fake_loss.item():.4f}",
                )
                if args.report_to:
                    logs = {
                        "train/dmd2_loss": dmd2_loss.item(),
                        "train/teacher_latent_l1_loss": latent_l1_loss.item(),
                        "train/teacher_latent_l2_loss": latent_l2_loss.item(),
                        "train/fake_loss": fake_loss.item(),
                        "train/lpips_loss": lpips_loss.item(),
                        "train/gan_loss": gan_loss.item(),
                        "train/discriminator_real_loss": discriminator_real_loss.item(),
                        "train/discriminator_fake_loss": discriminator_fake_loss.item(),
                        "train/student_lr": scheduler_student.get_last_lr()[0],
                        "train/fake_lr": scheduler_fake.get_last_lr()[0],
                        "train/is_fake_pretrain": int(fake_pretrain),
                    }
                    if scheduler_discriminator is not None:
                        logs["train/discriminator_lr"] = scheduler_discriminator.get_last_lr()[0]
                    accelerator.log(logs, step=global_step)
                checkpoint_interval = (
                    args.checkpointing_steps
                    if global_step <= args.fake_critic_pretrain_steps
                    else args.checkpointing_steps_after_fake_critic_pretrain
                )
                if accelerator.is_main_process and checkpoint_interval and global_step % checkpoint_interval == 0:
                    save_student_lora(student_pipe, accelerator, output_dir / f"checkpoint-{global_step}")
                if validation_cases and args.validation_steps and global_step % args.validation_steps == 0:
                    if accelerator.is_main_process:
                        run_validation(
                            student_pipe,
                            args,
                            accelerator,
                            output_dir,
                            validation_cases,
                            prompt,
                            weight_dtype,
                            global_step,
                        )
                    accelerator.wait_for_everyone()
                if global_step >= args.max_train_steps:
                    break
        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_student_lora(student_pipe, accelerator, output_dir)
        print(f"Saved distilled object-removal student LoRA to: {output_dir}")
    if args.report_to:
        accelerator.end_training()


if __name__ == "__main__":
    main(parse_args())
