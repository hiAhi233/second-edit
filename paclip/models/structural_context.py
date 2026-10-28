"""逐层 ROI 邻域上下文：3x3 特征均值与 1-cosine 误差，正常源验证校准。

不训练 MLP；默认包含中心 patch 的 3x3 均值，``exclude_center=True`` 是单独消融。
上下文误差只用于异常分数，不被当作已校准病灶概率；与基础概率融合前先校准到
0–1 的 context_score。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class ContextConfig:
    kernel: int = 3
    exclude_center: bool = False
    scale_floor: float = 1e-4
    eps: float = 1e-6
    calibration_by: str = "sequence"

    def __post_init__(self):
        if self.kernel < 3 or self.kernel % 2 == 0:
            raise ValueError("Context kernel must be an odd integer >= 3")
        if self.scale_floor <= 0 or self.eps <= 0:
            raise ValueError("scale_floor and eps must be positive")
        if self.calibration_by not in ("none", "sequence"):
            raise ValueError("calibration_by must be 'none' or 'sequence'")


class StructuralContext:
    def __init__(self, config: Optional[ContextConfig] = None,
                 layer_ids=(5, 8, 11), layer_weights=(.25, .35, .40)):
        self.config = config or ContextConfig()
        self.layer_ids = list(layer_ids)
        self.layer_weights = torch.as_tensor(layer_weights, dtype=torch.float32)
        if len(self.layer_ids) != len(self.layer_weights) or (self.layer_weights < 0).any() or not self.layer_weights.sum() > 0:
            raise ValueError("Layer IDs and nonnegative weights must match")
        self.layer_weights = self.layer_weights / self.layer_weights.sum()
        self.calibration = None
        self.last_valid_counts = None
        self.last_group_fallback = 0

    def compute_errors(self, levels, roi, hw):
        """levels: (B,L,N,D); roi: (B,H,W) bool -> errors (B,L,H,W), valid (B,L,H,W)."""
        if levels.ndim != 4:
            raise ValueError("levels must have shape (B,L,N,D)")
        batch, layers, tokens, dim = levels.shape
        height, width = hw
        if layers != len(self.layer_ids) or tokens != height * width:
            raise ValueError("Context layers or patch grid do not match the model output")
        if roi.shape != (batch, height, width):
            raise ValueError("ROI must have shape (B,H,W)")
        features = levels.detach().float().reshape(batch, layers, height, width, dim)
        mask_feature = roi.detach().float()[..., None]
        mask_pool = roi.detach().float()[:, None]
        kernel = int(self.config.kernel)
        padding = kernel // 2
        errors = features.new_zeros((batch, layers, height, width))
        valid = torch.zeros((batch, layers, height, width), dtype=torch.bool, device=features.device)
        counts = []
        for level in range(layers):
            feature = features[:, level]
            weighted = (feature * mask_feature).permute(0, 3, 1, 2)
            summed = F.avg_pool2d(weighted, kernel, stride=1, padding=padding,
                                  count_include_pad=True) * (kernel * kernel)
            count = F.avg_pool2d(mask_pool, kernel, stride=1, padding=padding,
                                 count_include_pad=True) * (kernel * kernel)
            if self.config.exclude_center:
                summed = summed - feature.permute(0, 3, 1, 2) * mask_pool
                count = count - mask_pool
            context = summed / count.clamp_min(1.0)
            cosine = F.cosine_similarity(feature.permute(0, 3, 1, 2), context, dim=1)
            error = (1.0 - cosine).clamp(0.0, 2.0)
            level_valid = (count[:, 0] > 0) & roi
            errors[:, level] = torch.where(level_valid, error, torch.zeros_like(error))
            valid[:, level] = level_valid
            counts.append((level_valid.sum().item(), level_valid.numel()))
        self.last_valid_counts = counts
        return errors, valid

    def calibrate(self, errors, valid, groups, sample_ids, split="val"):
        """Only normal source validation ROI patches may set the context calibration."""
        if split != "val":
            raise ValueError("Context calibration must use normal source validation images only")
        if errors.ndim != 4 or valid.shape != errors.shape:
            raise ValueError("Context errors and validity must have shape (B,L,H,W)")
        if errors.shape[0] != len(sample_ids) or len(set(sample_ids)) != len(sample_ids):
            raise ValueError("Unique validation IDs must match the context batch")
        batch = errors.shape[0]
        if groups is None or self.config.calibration_by == "none":
            names = ["__default__"] * batch
        else:
            names = [str(value) for value in groups]
            if len(names) != batch:
                raise ValueError("Calibration groups must match the context batch")
        unique = list(dict.fromkeys(names))
        stats = {}
        for name in unique:
            index = torch.tensor([i for i, value in enumerate(names) if value == name], dtype=torch.long)
            location, scale, counts = [], [], []
            for level in range(len(self.layer_ids)):
                values = errors[index, level][valid[index, level]]
                if values.numel() < 2:
                    raise ValueError(f"Context group {name!r} layer {self.layer_ids[level]} has too few valid patches")
                median = values.median()
                mad = (values - median).abs().median()
                location.append(float(median))
                scale.append(max(float(mad) * 1.4826, self.config.scale_floor))
                counts.append(int(values.numel()))
            stats[name] = {"location": location, "scale": scale, "valid_counts": counts,
                           "slices": len(index)}
        default = "__default__" if "__default__" in stats else unique[0]
        self.calibration = {
            "method": "median_MAD_context_error",
            "groups": stats,
            "default_group": default,
            "location": stats[default]["location"],
            "scale": stats[default]["scale"],
            "group_count": len(stats),
            "status": "single_group" if len(stats) == 1 else "per_sequence",
            "normal_validation_ids": list(sample_ids),
            "kernel": self.config.kernel,
            "exclude_center": bool(self.config.exclude_center),
        }
        return self

    def _stats(self, groups, batch, device):
        if self.calibration is None:
            raise RuntimeError("Calibrate the context branch on normal validation first")
        stats = self.calibration["groups"]
        default = self.calibration["default_group"]
        if groups is None:
            names = [default] * batch
        elif isinstance(groups, str):
            names = [groups] * batch
        else:
            names = [str(value) for value in groups]
            if len(names) != batch:
                raise ValueError("Query groups must match the batch size")
        location, scale = [], []
        fallback = 0
        for name in names:
            row = stats.get(name)
            if row is None:
                row = stats[default]
                fallback += 1
            location.append(row["location"])
            scale.append(row["scale"])
        self.last_group_fallback = fallback
        return (torch.tensor(location, dtype=torch.float32, device=device),
                torch.tensor(scale, dtype=torch.float32, device=device))

    def scores(self, errors, valid, groups=None):
        """Return context_score in [0,1] and a validity mask (B,H,W)."""
        if errors.ndim != 4 or valid.shape != errors.shape:
            raise ValueError("Context errors and validity must have shape (B,L,H,W)")
        batch, layers = errors.shape[:2]
        if layers != len(self.layer_ids):
            raise ValueError("Context layer count does not match the configured layers")
        location, scale = self._stats(groups, batch, errors.device)
        normalized = (errors - location[:, :, None, None]) / scale[:, :, None, None]
        normalized = torch.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
        weights = self.layer_weights.to(errors.device)[None, :, None, None] * valid.to(errors.dtype)
        denominator = weights.sum(1)
        fused = (normalized * weights).sum(1) / denominator.clamp_min(self.config.eps)
        context_valid = denominator > 0
        context_score = torch.where(context_valid, fused.sigmoid(), torch.zeros_like(fused))
        return context_score, context_valid

    def fingerprint(self) -> str:
        payload = {"config": asdict(self.config), "layer_ids": self.layer_ids,
                   "layer_weights": self.layer_weights.tolist(),
                   "calibration": self.calibration}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
