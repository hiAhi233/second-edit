"""Synthetic fixtures ONLY. Not MRI validation, not formal patient metadata."""
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F


def fixture_rows(root, counts=(6, 2, 2)):
    root = Path(root)
    rows = []
    rng = np.random.default_rng(12)
    for split, n in zip(("train", "validation", "test"), counts):
        for i in range(n):
            index = len(rows)
            positive = i % 2
            gray = rng.integers(70, 160, (40, 48), dtype=np.uint8)
            mask = np.zeros_like(gray)
            if positive:
                mask[12:20, 13:23] = 255
                gray[mask > 0] = 200
            image_path = root / f"image_{index}.png"
            mask_path = root / f"mask_{index}.png"
            Image.fromarray(gray).save(image_path); Image.fromarray(mask).save(mask_path)
            rows.append({"sample_id": f"fixture/{index}", "patient_id": f"fixture-patient-{index}",
                "image_path": str(image_path), "mask_path": str(mask_path), "label": positive,
                "dataset": "synthetic_test_fixture", "sequence": "fixture", "disease_labels": ["lesion"] if positive else ["normal"],
                "patient_verified": True, "sequence_verified": True, "disease_verified": True,
                "metadata_source": "generated test fixture, not clinical data", "roi_verified": True,
                "study_id": "fixture", "volume_id": str(index), "orientation_verified": False, "slice_z": None,
                "split": split})
    return rows


class TinyEncoder(nn.Module):
    """512-D deterministic visual substitute, exercises real production trainer."""
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(1, 1)
        self.visual_inlayer_bank = nn.Module()
        self.visual_inlayer_bank.active = "brain"
        self.visual_inlayer_bank.adapters = nn.ModuleDict({"brain": nn.Linear(1, 1)})
        with torch.no_grad():
            self.backbone.weight.fill_(1); self.backbone.bias.zero_()
            self.visual_inlayer_bank.adapters["brain"].weight.fill_(.1)
            self.visual_inlayer_bank.adapters["brain"].bias.zero_()
        self.register_buffer("channels", torch.linspace(-1, 1, 512))

    def encode_image(self, images, return_levels=False):
        values = F.adaptive_avg_pool2d(images.mean(1, keepdim=True), (14, 14)).flatten(2).transpose(1, 2)
        values = self.backbone(values) + self.visual_inlayer_bank.adapters["brain"](values)
        levels = torch.stack([F.normalize(torch.sin(values * (i + 1) + self.channels), dim=-1) for i in range(3)], dim=1)
        return levels[:, 0].mean(1), levels.mean(1), (14, 14), levels
