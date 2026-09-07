import math
import gc
import os
import random
import numpy as np
import argparse
from omegaconf import OmegaConf

import cv2
import torch
from PIL import Image
from diffusers import Flux2KleinPipeline

def free_memory():
    gc.collect()
    torch.cuda.empty_cache()

def create_mask_overlayed_image(image: np.ndarray, mask: np.ndarray, mask_color=np.ndarray, opacity=0.75) -> np.ndarray:
    '''
    Create a masked overlay image from the given image and mask.
    '''
    mask_color = mask_color[np.newaxis, np.newaxis, ...]
    masked_image = image.copy()
    masked_image[mask > 0] = mask_color * opacity + masked_image[mask > 0] * (1 - opacity)
    return masked_image


def create_mask_overlayed_image_stripes(image: np.ndarray, mask: np.ndarray, opacity=0.75, stripe_width=12):
    """
    Overlay masked region with large diagonal stripes (+opacity blend),
    which VLMs can reliably detect.

    Args:
        image: HxWxC RGB image
        mask:  HxW binary mask
        opacity: blending factor (0-1)
        stripe_width: width of each diagonal stripe in pixels
    """
    image = image.copy().astype(np.float32)
    H, W, C = image.shape

    # Create diagonal stripe pattern:
    xx, yy = np.meshgrid(np.arange(W), np.arange(H))
    # Random flip
    if random.random() < 0.5:
        xx = W - xx
    if random.random() < 0.5:
        yy = H - yy

    # Create stripes: (x + y) // stripe_width is even → white, odd → black
    pattern = (((xx + yy) // stripe_width) % 2) * 255  # 0 or 255
    pattern_rgb = np.stack([pattern] * 3, axis=-1).astype(np.float32)

    # Expand mask to 3 channels
    mask_3c = mask[:, :, None].astype(np.float32)

    # Blend inside mask only
    blended = image * (1 - mask_3c * opacity) + pattern_rgb * (mask_3c * opacity)

    # return blended.clip(0, 255).astype(np.uint8)

    # border: white and red
    # Create 2-color boundary using only erosion: white inside, black outside
    # Inner boundary (white) - erode mask to get inner edge
    inter_width = stripe_width * 2 + 1
    eroded_inner = cv2.erode(mask, np.ones((inter_width, inter_width), np.uint8), iterations=1)
    inner_boundary = mask - eroded_inner
    
    # Outer boundary (black) - erode more to get outer edge
    outer_width = stripe_width // 2 * 2 + 1
    eroded_outer = cv2.erode(mask, np.ones((outer_width, outer_width), np.uint8), iterations=1)
    outer_boundary = mask - eroded_outer
    
    # Create boundary colors
    inner_boundary_3c = inner_boundary[:, :, None]
    outer_boundary_3c = outer_boundary[:, :, None]
    inner_color = np.array([255, 255, 255], dtype=np.float32)   # white
    outer_color = np.array([255, 0, 0], dtype=np.float32)         # red
    
    # Apply boundaries over the blended image (outer first, then inner)
    final_image = blended * (1 - inner_boundary_3c * opacity) + inner_color * inner_boundary_3c * opacity
    final_image = final_image * (1 - outer_boundary_3c * opacity) + outer_color * outer_boundary_3c * opacity

    return final_image.clip(0, 255).astype(np.uint8)



def resize_target_area(image, target_area=1024*1024):
    width, height = image.size
    if width * height > target_area:
        scale = (target_area / (width * height)) ** 0.5
        new_width = int(width * scale) // 32 * 32
        new_height = int(height * scale) // 32 * 32
    else:
        new_width = width // 32 * 32
        new_height = height // 32 * 32
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--mask_root", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    config = OmegaConf.load(args.config)

    model_id = getattr(config, "pretrained_model_name_or_path", "black-forest-labs/FLUX.2-klein-base-9B")
    pipe = Flux2KleinPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    pipe.to("cuda")


    if args.lora_path is not None:
        pipe.load_lora_weights(args.lora_path, adapter_name="removal")

        if args.lora_strength != 1.0:
            pipe.set_adapters(["removal"], adapter_weights=[args.lora_strength])

        # TODO: verify speed between fuse / unfused
        pipe.fuse_lora("removal")

    # set up prompt
    model_prompt = config.model_prompt

    if args.seed is not None:
        seed = args.seed
    else:
        seed = random.randint(0, 1000000)

    generator = torch.Generator(device="cuda").manual_seed(seed)

    image_root = args.image_root
    mask_root = args.mask_root
    output_root = f'{args.output_root}_seed{seed}'
    os.makedirs(output_root, exist_ok=True)
    
    image_files = sorted(os.listdir(image_root))
    mask_files = sorted(os.listdir(mask_root))

    #########################################################
    # calculate prompt embedding in advanced
    prompt_embeds, _ = pipe.encode_prompt(model_prompt, max_sequence_length=512)
    negative_prompt_embeds, _ = pipe.encode_prompt("", max_sequence_length=512)

    # del text encoder
    del pipe.text_encoder
    del pipe.tokenizer
    free_memory()
    #########################################################

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
            
        val_mask = val_mask > 0

        # create masked overlay image
        if 'train_datasets' in config:
            dataset_config = config.train_datasets[0]
        elif 'train_dataset' in config:
            dataset_config = config.train_dataset
        else:
            raise ValueError(f"train_dataset or train_datasets must be provided")

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

        val_mask_overlayed_image = Image.fromarray(val_mask_overlayed_image)

        # # save for debug
        # overlayed_root_for_debug = f'overlayed_images_opacity{opacity}_for_debug'
        # os.makedirs(overlayed_root_for_debug, exist_ok=True)
        # val_mask_overlayed_image.save(os.path.join(overlayed_root_for_debug, image_file))

        # resize first
        val_mask_overlayed_image = resize_target_area(val_mask_overlayed_image)

        with torch.no_grad():
            output = pipe(
                # prompt=model_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                image=val_mask_overlayed_image,
                height=val_mask_overlayed_image.height,
                width=val_mask_overlayed_image.width,
                generator=generator,
                num_inference_steps=50,
                guidance_scale=4.0
            ).images[0]
        
        output = output.resize((val_image.width, val_image.height))

        output.save(os.path.join(output_root, image_file))

