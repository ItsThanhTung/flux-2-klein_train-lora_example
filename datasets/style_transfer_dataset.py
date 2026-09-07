import json
import os
import traceback
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


PROMPT_POOL = [
    "apply the artistic style of the second image to the first image, keep the subject and composition of the first image",
    "restyle the content of the first image using the visual style of the second image",
    "transfer the style from the style reference onto the content image while preserving the original subject",
    "redraw the first image in the style of the second image, preserve pose, layout, and identity",
    "change the first image to match the art style of the second image without changing the subject",
    "stylize the content image with the look of the style reference, keep composition intact",
    "paint the first image using the style of the second image",
    "convert the first photo into the artistic style shown in the second image",
]


def _resolve_triplet(row: dict, data_root: str) -> dict:
    """Normalize various pair schemas into content/style/target absolute paths."""
    if "content_image" in row and "style_image" in row and "stylized_image" in row:
        content, style, target = row["content_image"], row["style_image"], row["stylized_image"]
    elif "image" in row and "ref_style" in row and "edited_image" in row:
        content, style, target = row["image"], row["ref_style"], row["edited_image"]
    elif "content" in row and "style" in row and "target" in row:
        content, style, target = row["content"], row["style"], row["target"]
    else:
        raise KeyError(f"Unrecognized pair keys: {list(row.keys())}")

    def abspath(p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(data_root, p)

    return {
        "content": abspath(content),
        "style": abspath(style),
        "target": abspath(target),
        "prompt": row.get("prompt"),
        "style_name": row.get("style_name"),
    }


class StyleTransferDataset(Dataset):
    """Content + style reference → stylized edited image.

    Supports:
      - pairs.json / pairs.jsonl with keys:
          content_image, style_image, stylized_image  (ominidataset / omnistyle)
          OR image, ref_style, edited_image           (unified_style_data)
      - matching filenames under content_dir / style_dir / target_dir
    """

    def __init__(
        self,
        data_root: str,
        pairs_jsonl: str = None,
        pairs_json: str = None,
        content_dir: str = "images",
        style_dir: str = "style_refs",
        target_dir: str = "edited_images",
        out_size: int = 1024,
    ):
        self.data_root = data_root
        self.out_size = out_size
        self.pairs = []

        pairs_path = pairs_json or pairs_jsonl
        if pairs_path is not None:
            if not os.path.isabs(pairs_path):
                pairs_path = os.path.join(data_root, pairs_path)
            if pairs_path.endswith(".json") and not pairs_path.endswith(".jsonl"):
                with open(pairs_path, "r") as f:
                    rows = json.load(f)
                if isinstance(rows, dict) and "samples" in rows:
                    rows = rows["samples"]
                for row in rows:
                    self.pairs.append(_resolve_triplet(row, data_root))
            else:
                with open(pairs_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        self.pairs.append(_resolve_triplet(json.loads(line), data_root))
        else:
            content_root = os.path.join(data_root, content_dir)
            style_root = os.path.join(data_root, style_dir)
            target_root = os.path.join(data_root, target_dir)
            names = sorted(
                p.name
                for p in Path(target_root).iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
            )
            for name in names:
                self.pairs.append(
                    {
                        "content": os.path.join(content_root, name),
                        "style": os.path.join(style_root, name),
                        "target": os.path.join(target_root, name),
                    }
                )

        # Content + target share geometry; style is resized independently.
        self.pair_transform = A.Compose(
            [
                A.SmallestMaxSize(max_size=out_size),
                A.PadIfNeeded(min_height=out_size, min_width=out_size, border_mode=cv2.BORDER_REFLECT_101),
                A.RandomCrop(height=out_size, width=out_size),
                A.HorizontalFlip(p=0.5),
            ],
            additional_targets={"target": "image"},
        )
        self.style_transform = A.Compose(
            [
                A.SmallestMaxSize(max_size=out_size),
                A.PadIfNeeded(min_height=out_size, min_width=out_size, border_mode=cv2.BORDER_REFLECT_101),
                A.CenterCrop(height=out_size, width=out_size),
            ]
        )

    def to_tensor(self, image: np.ndarray) -> torch.Tensor:
        image = image.transpose(2, 0, 1)
        tensor = torch.from_numpy(image).float() / 255.0
        return (tensor - 0.5) / 0.5

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, item):
        try:
            pair = self.pairs[item]
            content = np.array(Image.open(pair["content"]).convert("RGB"))
            style = np.array(Image.open(pair["style"]).convert("RGB"))
            target = np.array(Image.open(pair["target"]).convert("RGB"))

            # Align target to content resolution before shared geometry augs
            if target.shape[:2] != content.shape[:2]:
                target = cv2.resize(
                    target,
                    (content.shape[1], content.shape[0]),
                    interpolation=cv2.INTER_AREA,
                )

            paired = self.pair_transform(image=content, target=target)
            content = paired["image"]
            target = paired["target"]
            style = self.style_transform(image=style)["image"]

            content_tensor = self.to_tensor(content)
            style_tensor = self.to_tensor(style)
            target_tensor = self.to_tensor(target)

            return {
                "start_image": content_tensor,
                "style_image": style_tensor,
                "target_image": target_tensor,
                "control_images": [
                    Image.fromarray(content),
                    Image.fromarray(style),
                ],
            }
        except Exception:
            raise Exception(f"Error in StyleTransferDataset.__getitem__: {traceback.format_exc()}")
