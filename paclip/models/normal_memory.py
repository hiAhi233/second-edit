"""Frozen E2 normal patch prototypes; only normal training support may build it.

Two fusion modes are supported explicitly:

* ``doc`` (default) follows the document exactly:
  ``z_l = (d_l - median_l) / (MAD_l + eps)``, ``A_proto = sum_l w_l z_l``,
  ``A_final = alpha * A_proto + (1 - alpha) * A_text``.
  There is no remapping to the text scale and no spatial ROI fallback.
* ``text_scale`` is the previous extension: robust-scale calibration,
  remapping to the text margin distribution, and an optional ROI fallback.

Regions are image-coordinate 3x3 cells.  Anatomical labels are not claimed
without verified orientation/registration.  Empty regions are either marked
unsupported (default) or, only for backward compatibility, filled from the
whole layer.  Unsupported regions use the text margin and are recorded.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans


@dataclass
class MemoryConfig:
    prototypes_per_region: int = 8
    region_rows: int = 3
    region_cols: int = 3
    seed: int = 0
    roi_threshold: float = 20.0 / 255.0
    scale_floor: float = 1e-4
    fusion_mode: str = "doc"            # "doc" | "text_scale"
    empty_region: str = "text"          # "text" | "global"
    roi_empty: str = "invalid"          # "invalid" | "all"
    min_roi_patches: int = 5
    roi_fallback: bool = False          # only used by text_scale
    calibration_by: str = "none"        # "none" | "sequence"
    region_mode: str = "image"          # only "image" is verified
    eps: float = 1e-6

    def __post_init__(self):
        if self.prototypes_per_region < 1 or min(self.region_rows, self.region_cols) < 1:
            raise ValueError("Prototype and region counts must be positive")
        if not 0 <= self.roi_threshold <= 1:
            raise ValueError("ROI threshold must be in [0, 1]")
        if self.scale_floor <= 0 or self.min_roi_patches < 1 or self.eps <= 0:
            raise ValueError("Scale floor, ROI patch minimum and epsilon must be positive")
        if self.fusion_mode not in ("doc", "text_scale"):
            raise ValueError("fusion_mode must be 'doc' or 'text_scale'")
        if self.empty_region not in ("text", "global"):
            raise ValueError("empty_region must be 'text' or 'global'")
        if self.roi_empty not in ("invalid", "all"):
            raise ValueError("roi_empty must be 'invalid' or 'all'")
        if self.calibration_by not in ("none", "sequence"):
            raise ValueError("calibration_by must be 'none' or 'sequence'")
        if self.region_mode != "image":
            raise NotImplementedError(
                "Anatomical partition requires verified orientation/registration; "
                "the current PNG support has no reliable anatomical direction."
            )


def patch_regions(hw, rows=3, cols=3, device=None):
    h, w = hw
    y = torch.arange(h, device=device) * rows // h
    x = torch.arange(w, device=device) * cols // w
    return (y[:, None] * cols + x[None]).reshape(-1)


def image_roi(gray, hw, threshold=20.0 / 255.0, min_patches=5, empty="invalid"):
    """Image-only foreground; no lesion annotations or anatomical directions.

    ``empty='invalid'`` returns an all-false ROI when the foreground is too
    small, so callers must not silently use the whole image (background) as
    normal tissue.  ``empty='all'`` reproduces the old behaviour for ablations.
    """
    roi = F.adaptive_max_pool2d((gray[:, None] > threshold).float(), hw)[:, 0].bool()
    too_small = roi.flatten(1).sum(1) < int(min_patches)
    if empty == "all":
        roi[too_small] = True
    else:
        roi[too_small] = False
    return roi


class NormalPatchMemory:
    def __init__(self, config=None, layer_ids=(5, 8, 11), layer_weights=(.25, .35, .40)):
        self.config = config or MemoryConfig()
        self.layer_ids = list(layer_ids)
        self.layer_weights = torch.as_tensor(layer_weights, dtype=torch.float32)
        if len(self.layer_ids) != len(self.layer_weights) or (self.layer_weights < 0).any() or not self.layer_weights.sum() > 0:
            raise ValueError("Layer IDs and nonnegative weights must match")
        self.layer_weights = self.layer_weights / self.layer_weights.sum()
        self.prototypes = []
        self.supported = []
        self.metadata = {}
        self.calibration = None
        self.checkpoint_sha256 = None
        self.support_digest = None
        self.last_group_fallback = 0

    def _region_supported_flags(self, level):
        return list(self.supported[level]) if level < len(self.supported) else []

    @torch.no_grad()
    def build(self, features, roi, hw, labels, sample_ids, split="train",
              checkpoint_sha256=None, support_digest=None):
        if split != "train" or not bool((torch.as_tensor(labels) == 0).all()):
            raise ValueError("Memory may only use normal training support samples")
        if len(sample_ids) != features.shape[0] or len(set(sample_ids)) != len(sample_ids) or not len(sample_ids):
            raise ValueError("Unique normal support IDs must match the feature batch")
        b, levels, n, dim = features.shape
        if levels != len(self.layer_ids) or n != hw[0] * hw[1] or roi.numel() != b * n:
            raise ValueError("Feature layers, patch grid and ROI must agree")
        features = F.normalize(features.detach().float().cpu(), dim=-1)
        roi = roi.detach().cpu().bool().reshape(b, n)
        empty = torch.nonzero(roi.flatten(1).sum(1) == 0, as_tuple=False).flatten().tolist()
        if empty:
            raise ValueError(
                "Normal support has no valid foreground ROI (support insufficient): "
                + ", ".join(str(sample_ids[i]) for i in empty)
            )
        regions = patch_regions(hw, self.config.region_rows, self.config.region_cols)
        self.prototypes, self.supported, stats = [], [], []
        for level in range(levels):
            bank, flags = [], []
            for region in range(self.config.region_rows * self.config.region_cols):
                selected = roi & (regions[None] == region)
                data = features[:, level][selected]
                native_supported = len(data) > 0
                fallback = "none"
                if not native_supported and self.config.empty_region == "global":
                    data = features[:, level][roi]
                    fallback = "global"
                if not native_supported and self.config.empty_region == "text":
                    bank.append(torch.empty(0, dim))
                    flags.append(False)
                    stats.append({"layer": self.layer_ids[level], "region": region,
                                  "roi_patches": int(selected.sum()), "prototypes": 0,
                                  "supported": False, "fallback": "text",
                                  "fallback_global_normal_roi": False})
                    continue
                unique = np.unique(data.numpy(), axis=0)
                count = min(self.config.prototypes_per_region, len(unique))
                centers = KMeans(n_clusters=count, random_state=self.config.seed, n_init=10).fit(unique).cluster_centers_
                bank.append(F.normalize(torch.from_numpy(centers).float(), dim=-1))
                flags.append(True)
                stats.append({"layer": self.layer_ids[level], "region": region,
                              "roi_patches": int(selected.sum()), "prototypes": count,
                              "supported": True, "fallback": fallback,
                              "fallback_global_normal_roi": fallback == "global"})
            self.prototypes.append(bank)
            self.supported.append(flags)
        self.checkpoint_sha256 = checkpoint_sha256
        self.support_digest = support_digest
        total = levels * self.config.region_rows * self.config.region_cols
        unsupported = [(c["layer"], c["region"]) for c in stats if not c["supported"]]
        global_fallback = [(c["layer"], c["region"]) for c in stats if c["fallback"] == "global"]
        self.metadata = {
            "version": "E2_normal_memory_v2",
            "normal_support_ids": list(sample_ids),
            "support_split": split,
            "regions": f"image-coordinate {self.config.region_rows}x{self.config.region_cols}; anatomical orientation unverified",
            "region_mode": self.config.region_mode,
            "fusion_mode": self.config.fusion_mode,
            "empty_region": self.config.empty_region,
            "checkpoint_sha256": checkpoint_sha256,
            "support_digest": support_digest,
            "counts": stats,
            "coverage": {"total_regions": total,
                         "supported_regions": sum(1 for c in stats if c["supported"]),
                         "unsupported_regions": unsupported,
                         "global_fallback_regions": global_fallback},
        }
        self.calibration = None
        return self

    @torch.no_grad()
    def distances(self, features, hw):
        if not self.prototypes:
            raise RuntimeError("Build or load a normal memory first")
        b, levels, n, _ = features.shape
        if levels != len(self.layer_ids) or n != hw[0] * hw[1]:
            raise ValueError("Query layer count or patch grid is incompatible")
        features = F.normalize(features.float(), dim=-1)
        regions = patch_regions(hw, self.config.region_rows, self.config.region_cols, features.device)
        result = features.new_full((b, levels, n), float("nan"))
        for level, bank in enumerate(self.prototypes):
            for region, prototypes in enumerate(bank):
                selected = regions == region
                if prototypes.numel() == 0:
                    continue
                similarity = features[:, level, selected] @ prototypes.to(features.device).T
                result[:, level, selected] = (1 - similarity.max(-1).values).clamp(0, 2)
        return result.reshape(b, levels, *hw)

    @torch.no_grad()
    def calibrate(self, distances, text_maps, roi, labels, sample_ids, split="val", groups=None):
        if split != "val" or not bool((torch.as_tensor(labels) == 0).all()):
            raise ValueError("Distance calibration must use normal source validation images only")
        if len(sample_ids) != distances.shape[0] or len(set(sample_ids)) != len(sample_ids):
            raise ValueError("Validation IDs must match the distance batch")
        if set(sample_ids) & set(self.metadata["normal_support_ids"]):
            raise ValueError("Normal support and calibration IDs must be disjoint")
        roi = roi.detach().cpu().bool()
        if not roi.any():
            raise ValueError("No foreground patches available for calibration")
        distances, text_maps = distances.detach().cpu(), text_maps.detach().cpu()
        batch = distances.shape[0]
        if groups is None or self.config.calibration_by == "none":
            names = ["__default__"] * batch
        else:
            names = [str(value) for value in groups]
            if len(names) != batch:
                raise ValueError("Calibration groups must match the calibration batch")
        unique = list(dict.fromkeys(names))
        group_stats = {}
        for name in unique:
            index = torch.tensor([i for i, value in enumerate(names) if value == name], dtype=torch.long)
            location, scale = [], []
            for level in range(len(self.layer_ids)):
                distances_level = distances[index, level]
                valid = roi[index] & torch.isfinite(distances_level)
                values = distances_level[valid]
                if values.numel() < 2:
                    raise ValueError(
                        f"Calibration group {name!r} layer {self.layer_ids[level]} has too few valid ROI patches"
                    )
                median = values.median()
                mad = (values - median).abs().median()
                location.append(float(median))
                if self.config.fusion_mode == "doc":
                    scale.append(float(mad) + self.config.eps)
                else:
                    scale.append(max(float(mad) * 1.4826, self.config.scale_floor))
            text_values = text_maps[index][roi[index]]
            if text_values.numel() < 2:
                raise ValueError(f"Calibration group {name!r} has too few text patches")
            text_median = text_values.median()
            text_mad = (text_values - text_median).abs().median()
            location.append(float(text_median))
            if self.config.fusion_mode == "doc":
                scale.append(float(text_mad) + self.config.eps)
            else:
                scale.append(max(float(text_mad) * 1.4826, self.config.scale_floor))
            group_stats[name] = {"location": location, "scale": scale, "slices": len(index)}
        default = "__default__" if "__default__" in group_stats else unique[0]
        self.calibration = {
            "method": "doc_MAD" if self.config.fusion_mode == "doc" else "median_MAD_text_scale",
            "fusion_mode": self.config.fusion_mode,
            "calibration_by": "none" if names[0] == "__default__" else "sequence",
            "groups": group_stats,
            "default_group": default,
            "group_count": len(group_stats),
            "status": "single_group" if len(group_stats) == 1 else "per_sequence",
            "location": group_stats[default]["location"],
            "scale": group_stats[default]["scale"],
            "normal_validation_ids": list(sample_ids),
            "normal_validation_group_counts": {name: len([x for x in names if x == name]) for name in unique},
        }
        return self

    def _stats_for_groups(self, groups, batch, device):
        calibration = self.calibration
        stats = calibration["groups"]
        default = calibration["default_group"]
        if groups is None:
            names = [default] * batch
        elif isinstance(groups, str):
            names = [groups] * batch
        else:
            names = [str(value) for value in groups]
            if len(names) != batch:
                raise ValueError("Query groups must match the batch size")
        mean, scale, fallback = [], [], 0
        for name in names:
            row = stats.get(name)
            if row is None:
                row = stats[default]
                fallback += 1
            mean.append(row["location"])
            scale.append(row["scale"])
        self.last_group_fallback = fallback
        return (torch.tensor(mean, dtype=torch.float32, device=device),
                torch.tensor(scale, dtype=torch.float32, device=device), fallback)

    def _support_grid(self, hw, device):
        levels = len(self.layer_ids)
        regions = patch_regions(hw, self.config.region_rows, self.config.region_cols, device).reshape(hw)
        ok = torch.zeros(levels, *hw, dtype=torch.bool, device=device)
        for level in range(levels):
            for region, supported in enumerate(self._region_supported_flags(level)):
                if supported:
                    ok[level][regions == region] = True
        return ok

    def scores(self, distances, text_maps, alpha, temperature=.07, roi=None, groups=None):
        """Map distances to an anomaly margin.

        ``doc`` mode is the document formula; ``text_scale`` keeps the old
        extension for ablations.  Unsupported regions and completely invalid
        images use the text margin and are reported through ``support_mask``.
        """
        if self.calibration is None:
            raise RuntimeError("Calibrate with normal source validation images first")
        if not 0 <= alpha <= 1:
            raise ValueError("Prototype fusion alpha must be in [0, 1]")
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        batch, levels = distances.shape[:2]
        mean, scale, fallback = self._stats_for_groups(groups, batch, distances.device)
        normalized = (distances - mean[:, :levels, None, None]) / scale[:, :levels, None, None]
        normalized = torch.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
        support = self._support_grid(distances.shape[-2:], distances.device)
        weights = self.layer_weights.to(distances.device)[None, :, None, None] * support[None].to(normalized.dtype)
        denominator = weights.sum(1)
        prototype_z = (normalized * weights).sum(1) / denominator.clamp_min(self.config.eps)
        supported_patch = (denominator > 0).expand(batch, -1, -1)
        text_maps = text_maps.to(distances.device)
        if self.config.fusion_mode == "doc":
            prototype_margin = prototype_z
            fused_margin = alpha * prototype_margin + (1 - alpha) * text_maps
        else:
            prototype_margin = mean[:, -1][:, None, None] + scale[:, -1][:, None, None] * prototype_z
            if self.config.roi_fallback and roi is not None:
                prototype_margin = torch.where(roi.to(distances.device).bool(), prototype_margin, text_maps)
            fused_margin = alpha * prototype_margin + (1 - alpha) * text_maps
        prototype_margin = torch.where(supported_patch, prototype_margin, text_maps)
        fused_margin = torch.where(supported_patch, fused_margin, text_maps)
        if roi is not None:
            valid_image = roi.to(distances.device).bool().flatten(1).any(1)[:, None, None]
            prototype_margin = torch.where(valid_image, prototype_margin, text_maps)
            fused_margin = torch.where(valid_image, fused_margin, text_maps)
        return {
            "prototype": (prototype_margin / temperature).sigmoid(),
            "text": (text_maps / temperature).sigmoid(),
            "fused": (fused_margin / temperature).sigmoid(),
            "fused_margin": fused_margin,
            "support_mask": supported_patch,
            "group_fallback": fallback,
        }

    def save(self, path):
        torch.save({"config": asdict(self.config), "layer_ids": self.layer_ids,
                    "layer_weights": self.layer_weights, "prototypes": self.prototypes,
                    "supported": self.supported, "metadata": self.metadata,
                    "calibration": self.calibration,
                    "checkpoint_sha256": self.checkpoint_sha256,
                    "support_digest": self.support_digest}, Path(path))

    @classmethod
    def load(cls, path, checkpoint_sha256=None, support_digest=None):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        obj = cls(MemoryConfig(**payload["config"]), payload["layer_ids"], payload["layer_weights"])
        obj.prototypes, obj.metadata, obj.calibration = payload["prototypes"], payload["metadata"], payload["calibration"]
        obj.supported = payload.get("supported")
        if obj.supported is None:
            obj.supported = [[prototypes.numel() > 0 for prototypes in bank] for bank in obj.prototypes]
        obj.checkpoint_sha256 = payload.get("checkpoint_sha256") or obj.metadata.get("checkpoint_sha256")
        obj.support_digest = payload.get("support_digest") or obj.metadata.get("support_digest")
        if checkpoint_sha256 is not None:
            if obj.checkpoint_sha256 is None:
                raise ValueError("Memory has no checkpoint hash; rebuild it with a checkpoint hash before verification")
            if checkpoint_sha256 != obj.checkpoint_sha256:
                raise ValueError("Prototype bank checkpoint does not match the requested checkpoint")
        if support_digest is not None:
            if obj.support_digest is None:
                raise ValueError("Memory has no support digest; rebuild it with a support digest before verification")
            if support_digest != obj.support_digest:
                raise ValueError("Prototype bank support set does not match the requested support digest")
        return obj


class MemoryAugmentedModel(torch.nn.Module):
    """Optional inference wrapper; disabled mode delegates E1 outputs unchanged."""

    def __init__(self, backbone, memory, alpha=.7, enabled=True,
                 checkpoint_sha256=None, support_digest=None, require_match=False):
        super().__init__()
        self.backbone, self.memory, self.alpha, self.enabled = backbone, memory, alpha, enabled
        self.cfg = backbone.cfg
        if require_match and checkpoint_sha256 is None:
            raise ValueError("require_match=True needs the expected checkpoint_sha256")
        if require_match and support_digest is None:
            raise ValueError("require_match=True needs the expected support_digest")
        stored_checkpoint = memory.checkpoint_sha256 or memory.metadata.get("checkpoint_sha256")
        stored_support = memory.support_digest or memory.metadata.get("support_digest")
        if checkpoint_sha256 is not None:
            if stored_checkpoint is None and require_match:
                raise ValueError("Memory has no checkpoint hash; cannot verify the wrapper binding")
            if stored_checkpoint is not None and checkpoint_sha256 != stored_checkpoint:
                raise ValueError("Prototype bank checkpoint does not match the wrapper checkpoint")
        if support_digest is not None:
            if stored_support is None and require_match:
                raise ValueError("Memory has no support digest; cannot verify the wrapper binding")
            if stored_support is not None and support_digest != stored_support:
                raise ValueError("Prototype bank support set does not match the wrapper support digest")

    def encode_anchors(self, prompts):
        return self.backbone.encode_anchors(prompts)

    @torch.no_grad()
    def forward(self, images, anchors):
        outputs = self.backbone(images, anchors)
        if not self.enabled:
            return outputs
        text = outputs["anomaly_map"]
        distances = self.memory.distances(outputs["f_patch_levels"], text.shape[-2:])
        mean = images.new_tensor([.48145466, .4578275, .40821073])[None, :, None, None]
        std = images.new_tensor([.26862954, .26130258, .27577711])[None, :, None, None]
        gray = (images * std + mean).mean(1)
        roi = image_roi(gray, text.shape[-2:], self.memory.config.roi_threshold,
                        self.memory.config.min_roi_patches, self.memory.config.roi_empty)
        scores = self.memory.scores(distances, text, self.alpha, self.cfg.temperature, roi)
        fused = scores["fused"]
        outputs = dict(outputs)
        outputs["anomaly_map_text"] = text
        outputs["prototype_distances"] = distances
        outputs["anomaly_score_e2"] = fused
        outputs["prototype_support_mask"] = scores["support_mask"]
        # Existing postprocessing applies sigmoid(margin/temperature) once.
        # Keep the actual fused margin, including the exact alpha=0 identity.
        outputs["anomaly_map"] = scores["fused_margin"]
        return outputs
