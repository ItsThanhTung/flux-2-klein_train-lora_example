import traceback
import json
import os
import random
from PIL import Image
import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import albumentations as A

def create_mask_overlayed_image(image: np.ndarray, mask: np.ndarray, mask_color=np.ndarray, opacity=0.75) -> np.ndarray:
    '''
    Create a masked overlay image from the given image and mask.
    '''
    mask_color = mask_color[np.newaxis, np.newaxis, ...]
    masked_image = image.copy()
    masked_image[mask > 0] = mask_color * opacity + masked_image[mask > 0] * (1 - opacity)
    return masked_image

class ObjectEffectDataset(Dataset):
    '''
    RORD dataset (https://github.com/Forty-lock/RORD)

    Use solid color to overlay the objects that user want to remove.
    '''
    def __init__(self, 
                 data_root: str, 
                 json_file_path: str, 
                 out_size: int = 1024, 
                 mask_dilate_kernel = [1, 10],
                 mask_overlay_color = np.array([255, 0, 0]),
                 mask_overlay_opacity = 0.75
                 ):

        with open(json_file_path, 'r') as file:
            self.data = json.load(file)
        self.data_root = data_root
        
        self.transform = A.Compose([
            A.OpticalDistortion(),
            A.SmallestMaxSize(max_size=out_size),
            A.CropNonEmptyMaskIfExists(height=out_size, width=out_size),
            A.HorizontalFlip(),
        ], additional_targets={'gt': 'image', 'mask': 'mask', 'mask_full': 'mask'})

        self.mask_dilate_kernel = mask_dilate_kernel
        if not isinstance(mask_overlay_color, np.ndarray):
            mask_overlay_color = np.array(mask_overlay_color)
        self.mask_overlay_color = mask_overlay_color
        self.mask_overlay_opacity = mask_overlay_opacity

    def to_tensor(self, image: np.ndarray) -> torch.Tensor:
        image = image.transpose(2, 0, 1)
        tensor = torch.from_numpy(image).float() / 255.0
        tensor = (tensor - 0.5) / 0.5
        return tensor

    def to_mask_tensor(self, mask: np.ndarray) -> torch.Tensor:
        '''
        value in 0-1
        '''
        mask = mask.astype(np.float32)
        if mask.ndim == 2:
            mask = mask[None, ...]  # expand to (1, H, W)
        tensor = torch.from_numpy(mask).float()
        return tensor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        try:

            data = self.data[item]

            # Load image and its panoptic map
            img_path = os.path.join(self.data_root, data['image_path'])
            img = np.array(Image.open(img_path).convert('RGB'))

            gt_path = os.path.join(self.data_root, data['gt_path'])
            gt = np.array(Image.open(gt_path).convert('RGB'))

            # input mask
            mask_object_only_path = os.path.join(self.data_root, data['mask_object_only_path'])
            mask_object_only = np.array(Image.open(mask_object_only_path).convert('L'))
            mask_object_only[mask_object_only > 0] = 1

            # # mask to compute loss (NOT USED NOW)
            # mask_full_path = os.path.join(self.data_root, data['mask_full_path'])
            # mask_full = np.array(Image.open(mask_full_path).convert('L'))
            # mask_full[mask_full > 0] = 1

            # Apply augmentations to image and mask
            # transformed = self.transform(image=img, gt=gt, mask=mask_object_only, mask_full=mask_full)
            transformed = self.transform(image=img, gt=gt, mask=mask_object_only)
            img = transformed['image']
            mask = transformed['mask']
            gt = transformed['gt']
            # mask_full = transformed['mask_full']

            # Dilate mask
            dilate_kernel = random.randint(self.mask_dilate_kernel[0], self.mask_dilate_kernel[1])
            mask = cv2.dilate(mask, np.ones(dilate_kernel, np.uint8), iterations=1)
            # mask_full = cv2.dilate(mask_full, np.ones(dilate_kernel, np.uint8), iterations=1)

            mask_overlayed_image = create_mask_overlayed_image(img, mask, mask_color=self.mask_overlay_color, opacity=self.mask_overlay_opacity)
            mask_overlayed_image = Image.fromarray(mask_overlayed_image)
            
            # create tensor to return
            # we have 2 strategy: also use mask overlayed image as input or use original image as input
            # just test with mask overlayed image as input now
            # img_tensor = self.to_tensor(img)
            img_tensor = self.to_tensor(np.array(mask_overlayed_image))
            gt_tensor = self.to_tensor(gt)
            # mask_tensor = self.to_mask_tensor(mask)
            # mask_full_tensor = self.to_mask_tensor(mask_full)

            item = {
                'start_image': img_tensor,
                'target_image': gt_tensor,
                'control_images': [mask_overlayed_image],
                # 'mask': mask_tensor,
            }
            return item
        
        except Exception as e:
            raise Exception(f"Error in .__getitem__: {traceback.format_exc()}")
            # print(f"Error in .__getitem__: {traceback.format_exc()}")
            # print(f"Returning random item from dataset")
            # return self.__getitem__(random.randint(0, len(self) - 1))


if __name__ == "__main__":
    dataset = ObjectEffectDataset(
        data_root="/workspace/data/RORD",
        json_file_path="/workspace/data/RORD/train.json",
        out_size=1024,
        mask_dilate_kernel=[1, 10],
        mask_overlay_color = np.array([255, 0, 0]),
        mask_overlay_opacity = 0.5
    )
    
    item = dataset[0]

    def tensor2pil(tensor: torch.Tensor) -> Image.Image:
        tensor = tensor * 0.5 + 0.5
        tensor = tensor.permute(1, 2, 0).cpu().numpy()
        tensor = (tensor * 255).astype(np.uint8)
        return Image.fromarray(tensor)

    # save for debug
    os.makedirs("tmp", exist_ok=True)
    tensor2pil(item['start_image']).convert('RGB').save("tmp/start_image.png")
    tensor2pil(item['target_image']).convert('RGB').save("tmp/target_image.png")
    item['control_images'][0].convert('RGB').save("tmp/control_image.png")