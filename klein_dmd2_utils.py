from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import Flux2KleinPipeline
from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu, retrieve_timesteps
from diffusers.training_utils import compute_density_for_timestep_sampling
from peft import LoraConfig


def unwrap_module(module: nn.Module) -> nn.Module:
    return module.module if hasattr(module, "module") else module


def set_discriminator_heads_grad(discriminator: nn.Module, requires_grad: bool) -> None:
    for parameter in unwrap_module(discriminator).heads.parameters():
        parameter.requires_grad_(requires_grad)


def default_lora_targets(num_single_blocks: int = 24) -> list[str]:
    """LoRA targets for Klein; 4B uses 20 single blocks, 9B uses 24."""
    targets = [
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
    targets.extend(f"single_transformer_blocks.{index}.attn.to_out" for index in range(num_single_blocks))
    return targets


def add_lora_adapter(
    transformer,
    name: str,
    rank: int,
    alpha: int,
    dropout: float,
    target_modules: list[str],
) -> None:
    transformer.add_adapter(
        LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        ),
        adapter_name=name,
    )


def set_adapter(transformer, name: str) -> None:
    transformer = unwrap_module(transformer)
    if hasattr(transformer, "set_adapter"):
        transformer.set_adapter(name)
    else:
        transformer.set_adapters([name])


def adapter_parameters(transformer, adapter_name: str) -> list[torch.nn.Parameter]:
    suffix = f".{adapter_name}.weight"
    parameters = []
    for name, parameter in transformer.named_parameters():
        if (".lora_A." in name or ".lora_B." in name) and name.endswith(suffix):
            parameter.requires_grad_(True)
            parameters.append(parameter)
    return parameters


@torch.no_grad()
def ema_update_adapter(transformer, source_adapter: str, target_adapter: str, decay: float) -> None:
    if decay >= 1.0:
        return
    named_parameters = dict(transformer.named_parameters())
    source_suffix = f".{source_adapter}.weight"
    target_suffix = f".{target_adapter}.weight"
    for source_name, source_parameter in named_parameters.items():
        if not source_name.endswith(source_suffix):
            continue
        target_name = source_name[: -len(source_suffix)] + target_suffix
        target_parameter = named_parameters.get(target_name)
        if target_parameter is not None:
            target_parameter.data.mul_(decay).add_(source_parameter.data, alpha=1.0 - decay)


def get_sigmas(
    noise_scheduler,
    timesteps: torch.Tensor,
    n_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    sigmas = noise_scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(device)
    timesteps = timesteps.to(device)
    step_indices = [(schedule_timesteps == timestep).nonzero().item() for timestep in timesteps]
    sigma = sigmas[step_indices].flatten()
    while sigma.ndim < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def expand_to_ndim(value: torch.Tensor, n_dim: int) -> torch.Tensor:
    while value.ndim < n_dim:
        value = value.unsqueeze(-1)
    return value


def sample_training_timesteps(args, noise_scheduler, batch_size: int, device: torch.device) -> torch.Tensor:
    density = compute_density_for_timestep_sampling(
        weighting_scheme=args.weighting_scheme,
        batch_size=batch_size,
        logit_mean=args.logit_mean,
        logit_std=args.logit_std,
        mode_scale=args.mode_scale,
    )
    indices = (density * noise_scheduler.config.num_train_timesteps).long()
    return noise_scheduler.timesteps[indices].to(device=device)


def sample_fake_timesteps(args, noise_scheduler, batch_size: int, device: torch.device) -> torch.Tensor:
    if args.fake_time_sampler == "infer":
        return sample_training_timesteps(args, noise_scheduler, batch_size, device)
    if args.fake_time_sampler == "logit_normal":
        density = compute_density_for_timestep_sampling(
            weighting_scheme="logit_normal",
            batch_size=batch_size,
            logit_mean=args.logit_mean,
            logit_std=args.logit_std,
            mode_scale=args.mode_scale,
        )
        indices = (density * noise_scheduler.config.num_train_timesteps).long()
        return noise_scheduler.timesteps[indices].to(device=device)
    if args.fake_time_sampler == "uniform_high":
        schedule_timesteps = noise_scheduler.timesteps.to(device=device)
        low = int(min(args.fake_time_min, args.fake_time_max))
        high = int(max(args.fake_time_min, args.fake_time_max))
        sampled = torch.randint(low, high + 1, (batch_size,), device=device, dtype=schedule_timesteps.dtype)
        distances = (schedule_timesteps.float().unsqueeze(0) - sampled.float().unsqueeze(1)).abs()
        return schedule_timesteps[distances.argmin(dim=1)]
    raise ValueError(f"Unsupported fake_time_sampler: {args.fake_time_sampler}")


def encode_images(
    vae,
    images: torch.Tensor,
    latents_bn_mean: torch.Tensor,
    latents_bn_std: torch.Tensor,
) -> torch.Tensor:
    latents = vae.encode(images).latent_dist.mode()
    latents = Flux2KleinPipeline._patchify_latents(latents)
    return (latents - latents_bn_mean) / latents_bn_std


def decode_images(vae, latents: torch.Tensor, weight_dtype: torch.dtype) -> torch.Tensor:
    mean = vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
    std = torch.sqrt(
        vae.bn.running_var.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        + vae.config.batch_norm_eps
    )
    latents = Flux2KleinPipeline._unpatchify_latents(latents * std + mean)
    return vae.decode(latents.to(dtype=weight_dtype), return_dict=False)[0]


def prepare_condition_tokens(cond_latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, num_conditions, _channels, _height, _width = cond_latents.shape
    packed = []
    ids = []
    for condition_index in range(num_conditions):
        latents = cond_latents[:, condition_index]
        packed.append(Flux2KleinPipeline._pack_latents(latents))
        ids.append(Flux2KleinPipeline._prepare_image_ids([latents[:1]], scale=10 + condition_index * 10))
    condition_tokens = torch.cat(packed, dim=1)
    condition_ids = torch.cat(ids, dim=1).to(cond_latents.device).expand(batch_size, -1, -1)
    return condition_tokens, condition_ids


def build_model_input(
    target_latents: torch.Tensor,
    cond_latents: torch.Tensor,
) -> tuple[torch.Tensor, int, torch.Tensor, torch.Tensor]:
    target_ids = Flux2KleinPipeline._prepare_latent_ids(target_latents).to(target_latents.device)
    target_tokens = Flux2KleinPipeline._pack_latents(target_latents)
    condition_tokens, condition_ids = prepare_condition_tokens(cond_latents)
    hidden_states = torch.cat([target_tokens, condition_tokens], dim=1)
    image_ids = torch.cat([target_ids, condition_ids], dim=1)
    return hidden_states, target_tokens.shape[1], target_ids, image_ids


def transformer_score(
    transformer,
    latents: torch.Tensor,
    cond_latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    text_ids: torch.Tensor,
    timesteps: torch.Tensor,
    weight_dtype: torch.dtype,
) -> torch.Tensor:
    hidden_states, target_length, target_ids, image_ids = build_model_input(latents, cond_latents)
    unwrapped = unwrap_module(transformer)
    guidance = None
    if getattr(unwrapped.config, "guidance_embeds", False):
        guidance = torch.ones(latents.shape[0], device=latents.device, dtype=weight_dtype)
    prediction = transformer(
        hidden_states=hidden_states.to(dtype=weight_dtype),
        timestep=timesteps.to(dtype=weight_dtype) / 1000,
        guidance=guidance,
        encoder_hidden_states=prompt_embeds,
        txt_ids=text_ids,
        img_ids=image_ids,
        return_dict=False,
    )[0]
    prediction = prediction[:, :target_length]
    return Flux2KleinPipeline._unpack_latents_with_ids(prediction, target_ids)


def transformer_cfg_score(
    transformer,
    latents: torch.Tensor,
    cond_latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    text_ids: torch.Tensor,
    negative_prompt_embeds: torch.Tensor,
    negative_text_ids: torch.Tensor,
    timesteps: torch.Tensor,
    guidance_scale: float,
    weight_dtype: torch.dtype,
) -> torch.Tensor:
    conditional = transformer_score(
        transformer, latents, cond_latents, prompt_embeds, text_ids, timesteps, weight_dtype
    )
    if guidance_scale <= 1.0:
        return conditional
    unconditional = transformer_score(
        transformer,
        latents,
        cond_latents,
        negative_prompt_embeds,
        negative_text_ids,
        timesteps,
        weight_dtype,
    )
    return unconditional + guidance_scale * (conditional - unconditional)


def prepare_klein_inference_schedule(
    noise_scheduler,
    latents: torch.Tensor,
    num_inference_steps: int,
    device: torch.device,
):
    scheduler = type(noise_scheduler).from_config(noise_scheduler.config)
    image_sequence_length = Flux2KleinPipeline._pack_latents(latents[:1]).shape[1]
    mu = compute_empirical_mu(image_seq_len=image_sequence_length, num_steps=num_inference_steps)
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    if getattr(scheduler.config, "use_flow_sigmas", False):
        sigmas = None
    timesteps, _ = retrieve_timesteps(scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu)
    scheduler.set_begin_index(0)
    return timesteps, scheduler.sigmas.to(device=device, dtype=latents.dtype)


def prepare_klein_training_noise_scheduler(
    noise_scheduler,
    latents: torch.Tensor,
    num_steps_for_mu: int,
    device: torch.device,
):
    scheduler = type(noise_scheduler).from_config(noise_scheduler.config)
    num_train_timesteps = scheduler.config.num_train_timesteps
    image_sequence_length = Flux2KleinPipeline._pack_latents(latents[:1]).shape[1]
    mu = compute_empirical_mu(image_seq_len=image_sequence_length, num_steps=num_steps_for_mu)
    sigmas = np.linspace(1.0, 1 / num_train_timesteps, num_train_timesteps)
    if getattr(scheduler.config, "use_flow_sigmas", False):
        sigmas = None
    retrieve_timesteps(scheduler, num_train_timesteps, device, sigmas=sigmas, mu=mu)
    scheduler.set_begin_index(0)
    return scheduler


@torch.no_grad()
def encode_batch(pipe, batch: dict, accelerator, weight_dtype: torch.dtype):
    prompt_embeds, text_ids = pipe.encode_prompt(
        prompt=batch["prompts"],
        device=accelerator.device,
        max_sequence_length=512,
        text_encoder_out_layers=(9, 18, 27),
    )
    pixels = batch["pixel_values"].to(accelerator.device, dtype=pipe.vae.dtype)
    mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(accelerator.device, dtype=weight_dtype)
    std = torch.sqrt(
        pipe.vae.bn.running_var.view(1, -1, 1, 1).to(accelerator.device, dtype=weight_dtype)
        + pipe.vae.config.batch_norm_eps
    )
    target_latents = encode_images(pipe.vae, pixels, mean, std)
    batch_size, num_conditions, channels, height, width = batch["cond_pixel_values"].shape
    condition_pixels = batch["cond_pixel_values"].reshape(
        batch_size * num_conditions, channels, height, width
    )
    condition_pixels = condition_pixels.to(accelerator.device, dtype=pipe.vae.dtype)
    condition_latents = encode_images(pipe.vae, condition_pixels, mean, std)
    condition_latents = condition_latents.reshape(batch_size, num_conditions, *condition_latents.shape[1:])
    return prompt_embeds, text_ids, target_latents, condition_latents


@torch.no_grad()
def encode_negative_prompt(pipe, batch_size: int, accelerator, weight_dtype: torch.dtype):
    prompt_embeds, text_ids = pipe.encode_prompt(
        prompt=[""] * batch_size,
        device=accelerator.device,
        max_sequence_length=512,
        text_encoder_out_layers=(9, 18, 27),
    )
    return prompt_embeds.to(dtype=weight_dtype), text_ids


@torch.no_grad()
def prepare_student_training_sample(
    transformer,
    noise: torch.Tensor,
    cond_latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    text_ids: torch.Tensor,
    noise_scheduler,
    num_inference_steps: int,
    weight_dtype: torch.dtype,
):
    timesteps, sigmas = prepare_klein_inference_schedule(
        noise_scheduler, noise, num_inference_steps, noise.device
    )
    selected_step = torch.randint(0, num_inference_steps, (1,), device=noise.device).item()
    latents = noise
    for step_index in range(selected_step):
        timestep = timesteps[step_index].expand(latents.shape[0]).to(latents.device)
        prediction = transformer_score(
            transformer, latents, cond_latents, prompt_embeds, text_ids, timestep, weight_dtype
        )
        current_sigma = sigmas[step_index].to(dtype=latents.dtype)
        next_sigma = sigmas[step_index + 1].to(dtype=latents.dtype)
        predicted_x0 = latents.float() - current_sigma.float() * prediction.float()
        latents = (
            (1.0 - next_sigma.float()) * predicted_x0
            + next_sigma.float() * torch.randn_like(predicted_x0)
        ).to(dtype=latents.dtype)
    selected_timestep = timesteps[selected_step].expand(latents.shape[0]).to(latents.device)
    selected_sigma = sigmas[selected_step].to(device=latents.device, dtype=latents.dtype)
    return latents, selected_timestep, selected_sigma


def predict_student_x0_from_noisy(
    transformer,
    noisy_latents: torch.Tensor,
    cond_latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    text_ids: torch.Tensor,
    timesteps: torch.Tensor,
    sigmas: torch.Tensor,
    weight_dtype: torch.dtype,
) -> torch.Tensor:
    prediction = transformer_score(
        transformer,
        noisy_latents,
        cond_latents,
        prompt_embeds,
        text_ids,
        timesteps,
        weight_dtype,
    )
    sigmas = expand_to_ndim(sigmas, noisy_latents.ndim)
    return (noisy_latents.float() - sigmas.float() * prediction.float()).to(noisy_latents.dtype)


def sample_final_latents(
    transformer,
    initial_latents: torch.Tensor,
    cond_latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    text_ids: torch.Tensor,
    noise_scheduler,
    num_steps: int,
    weight_dtype: torch.dtype,
    guidance_scale: float = 1.0,
    negative_prompt_embeds: torch.Tensor | None = None,
    negative_text_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    timesteps, sigmas = prepare_klein_inference_schedule(
        noise_scheduler, initial_latents, int(num_steps), initial_latents.device
    )
    latents = initial_latents
    for step_index, timestep in enumerate(timesteps):
        timestep_batch = timestep.expand(latents.shape[0]).to(latents.device)
        if guidance_scale > 1 and negative_prompt_embeds is not None and negative_text_ids is not None:
            prediction = transformer_cfg_score(
                transformer,
                latents,
                cond_latents,
                prompt_embeds,
                text_ids,
                negative_prompt_embeds,
                negative_text_ids,
                timestep_batch,
                guidance_scale,
                weight_dtype,
            )
        else:
            prediction = transformer_score(
                transformer,
                latents,
                cond_latents,
                prompt_embeds,
                text_ids,
                timestep_batch,
                weight_dtype,
            )
        current_sigma = sigmas[step_index].to(latents.device, latents.dtype)
        next_sigma = (
            sigmas[step_index + 1].to(latents.device, latents.dtype)
            if step_index + 1 < sigmas.shape[0]
            else torch.zeros_like(current_sigma)
        )
        current_sigma = expand_to_ndim(current_sigma, latents.ndim)
        next_sigma = expand_to_ndim(next_sigma, latents.ndim)
        latents = (latents.float() + (next_sigma.float() - current_sigma.float()) * prediction.float()).to(
            latents.dtype
        )
    return latents


def teacher_latent_losses(
    student_latents: torch.Tensor,
    teacher_latents: torch.Tensor,
    l1_weight: float,
    l2_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return weighted total, raw L1, and raw L2 teacher-matching losses."""
    zero = student_latents.new_zeros(())
    l1_loss = F.l1_loss(student_latents.float(), teacher_latents.float()) if l1_weight > 0 else zero
    l2_loss = F.mse_loss(student_latents.float(), teacher_latents.float()) if l2_weight > 0 else zero
    total = l1_weight * l1_loss + l2_weight * l2_loss
    return total, l1_loss, l2_loss
