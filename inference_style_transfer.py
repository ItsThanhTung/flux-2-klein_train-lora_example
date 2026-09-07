#!/usr/bin/env python3
"""Inference for content+style → stylized edit LoRA (flux-2-klein_train-lora_example)."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from diffusers import Flux2KleinPipeline
from omegaconf import OmegaConf
from PIL import Image


def resize_target_area(image: Image.Image, target_area: int = 1024 * 1024) -> Image.Image:
    width, height = image.size
    if width * height > target_area:
        scale = (target_area / (width * height)) ** 0.5
        new_width = max(32, int(width * scale) // 32 * 32)
        new_height = max(32, int(height * scale) // 32 * 32)
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    new_width = max(32, width // 32 * 32)
    new_height = max(32, height // 32 * 32)
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--lora_path", type=str, required=True)
    ap.add_argument("--content", type=Path, required=True)
    ap.add_argument("--style", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--lora_strength", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    config = OmegaConf.load(args.config)
    model_id = getattr(
        config, "pretrained_model_name_or_path", "black-forest-labs/FLUX.2-klein-base-9B"
    )
    prompt = getattr(
        config,
        "model_prompt",
        "apply the artistic style of the second image to the first image, keep the subject and composition of the first image",
    )

    pipe = Flux2KleinPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16).to("cuda")
    pipe.load_lora_weights(args.lora_path, adapter_name="style")
    if args.lora_strength != 1.0:
        pipe.set_adapters(["style"], adapter_weights=[args.lora_strength])

    content = resize_target_area(Image.open(args.content).convert("RGB"))
    style = resize_target_area(Image.open(args.style).convert("RGB")).resize(
        content.size, Image.Resampling.LANCZOS
    )

    gen = torch.Generator("cuda").manual_seed(args.seed)
    out = pipe(
        prompt=prompt,
        image=[content, style],
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        height=content.height,
        width=content.width,
        generator=gen,
    ).images[0]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.save(args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
