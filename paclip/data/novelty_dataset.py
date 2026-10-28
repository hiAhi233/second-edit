"""Manifest-backed PNG inputs with native GT and explicit ROI validity."""
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from paclip.data.datasets import _gray2clip


class NoveltyDataset(Dataset):
    def __init__(self, records, image_size=224):
        self.records = list(records)
        self.image_size = int(image_size)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        r = self.records[index]
        with Image.open(r["image_path"]) as im:
            native_size = im.size
            gray = np.asarray(im.convert("L").resize((self.image_size,) * 2, Image.Resampling.BILINEAR), dtype=np.float32) / 255
        if r.get("mask_path"):
            with Image.open(r["mask_path"]) as im:
                if im.size != native_size:
                    raise ValueError("Native GT size mismatch")
                mask = np.asarray(im.convert("L")) > 0
        else:
            if r["label"]:
                raise ValueError("Missing positive GT")
            mask = np.zeros(native_size[::-1], bool)
        if bool(mask.any()) != bool(r["label"]):
            raise ValueError("Image label and GT disagree")
        if r.get("roi_path"):
            with Image.open(r["roi_path"]) as im:
                roi = np.asarray(im.convert("L").resize((self.image_size,) * 2, Image.Resampling.NEAREST)) > 0
            policy = "provided_mask"
        else:
            roi = gray > 20 / 255
            policy = "intensity_gt20_not_anatomical_segmentation"
        valid = bool(r.get("roi_verified") is True and roi.sum() >= 3)
        return {**r, "gray": torch.from_numpy(gray.copy()), "image": _gray2clip(gray),
                "mask_high": torch.from_numpy(mask.astype(np.float32)),
                "roi": torch.from_numpy(roi.copy()), "roi_valid": valid, "roi_policy": policy}
