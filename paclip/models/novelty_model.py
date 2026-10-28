"""Separate train policy: never invoke the legacy stage policy on the new head."""
from contextlib import nullcontext
import torch
from torch import nn
import torch.nn.functional as F
from paclip.models.dense_localizer import DenseLocalizer, top_roi_score


@torch.no_grad()
def frozen_text_direction(encoder, prompts):
    """Same frozen fused text contrast for EACH independent visual depth."""
    encoder.eval()
    anchors = encoder.encode_anchors(prompts)
    levels = list(anchors)
    weights = encoder.fusion_weights.detach().softmax(0)
    if len(levels) != len(weights):
        raise ValueError("Text fusion/anchor count mismatch")
    return sum(weights[i] * (anchors[k]["abnormal"] - anchors[k]["normal"])
               for i, k in enumerate(levels)) / encoder.cfg.temperature


class NoveltyModel(nn.Module):
    def __init__(self, encoder=None, *, head_spec=None, train_visual=False, diagnostic_direction=None):
        super().__init__()
        self.encoder = encoder
        self.head = DenseLocalizer(**(head_spec or {}))
        direction = None if diagnostic_direction is None else torch.as_tensor(diagnostic_direction, dtype=torch.float32).detach().clone()
        if direction is not None and (direction.shape != (self.head.spec["input_dim"],) or not torch.isfinite(direction).all() or not direction.any()):
            raise ValueError("Frozen diagnostic text direction must be finite, nonzero and match feature width")
        self.register_buffer("diagnostic_direction", direction)
        self.train_visual = bool(train_visual)
        if self.train_visual and encoder is None:
            raise ValueError("Visual tuning requires a live encoder")
        self.apply_policy()

    def apply_policy(self):
        if self.encoder is not None:
            self.encoder.requires_grad_(False)
            if self.train_visual:
                bank = getattr(self.encoder, "visual_inlayer_bank", None)
                if bank is None or bank.active is None:
                    raise ValueError("Visual ablation requires an active visual adapter")
                bank.adapters[bank.active].requires_grad_(True)
        self.head.requires_grad_(True)

    def train(self, mode=True):
        super().train(mode)
        if self.encoder is not None:
            self.encoder.eval()
            if self.train_visual:
                self.encoder.visual_inlayer_bank.train(mode)
        self.head.train(mode)
        return self

    def extract_levels(self, images):
        if self.encoder is None:
            raise RuntimeError("Cache miss: live encoder is required")
        with nullcontext() if self.train_visual and self.training else torch.no_grad():
            _, _, grid, levels = self.encoder.encode_image(images, return_levels=True)
        return levels, tuple(grid)

    def decode_levels(self, levels, grid, roi, roi_valid=None):
        logits = self.head(levels, grid)
        working_logits = F.interpolate(logits, roi.shape[-2:], mode="bilinear", align_corners=False)
        score, valid = top_roi_score(working_logits, roi, roi_valid)
        probability = logits.sigmoid()
        maps = {f"map_layer{k}": None for k in (5, 8, 11)}
        per_layer_logits = None
        if self.diagnostic_direction is not None:
            if levels.shape[1] != 3:
                raise ValueError("Layer 5/8/11 diagnostics require exactly three visual levels")
            per_layer_logits = (levels.detach().float() @ self.diagnostic_direction.float()).reshape(levels.shape[0], 3, *grid)
            maps = {f"map_layer{k}": per_layer_logits[:, i].sigmoid() for i, k in enumerate((5, 8, 11))}
        return {"f_patch_levels": levels, "dense_logits": logits,
                "dense_probability": probability, "dense_anomaly_map": probability[:, 0],
                "image_anomaly_score": score, "roi_valid": valid, **maps,
                "map_layer_logits": per_layer_logits,
                "layer_maps_available": self.diagnostic_direction is not None,
                "layer_map_semantics": "sigmoid of independent visual-depth dot frozen fused text contrast; zero-based blocks 5/8/11"}

    def forward(self, images, roi, roi_valid=None):
        levels, grid = self.extract_levels(images)
        return self.decode_levels(levels, grid, roi, roi_valid)

    def trainable_names(self):
        return [n for n, p in self.named_parameters() if p.requires_grad]

    def assert_optimizer(self, optimizer):
        expected = {id(p) for p in self.parameters() if p.requires_grad}
        actual = [id(p) for g in optimizer.param_groups for p in g["params"]]
        if expected != set(actual) or len(actual) != len(set(actual)):
            raise AssertionError("Optimizer missing/duplicating trainable parameters or containing frozen parameters")
        if any(p.requires_grad for n, p in self.named_parameters()
               if n.startswith("encoder.") and not n.startswith("encoder.visual_inlayer_bank.")):
            raise AssertionError("Unexpected encoder/text gradient policy")
