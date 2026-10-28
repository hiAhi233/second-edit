"""Independent visual-depth projections and a single-logit spatial decoder."""
import math
import torch
from torch import nn
import torch.nn.functional as F


def top_roi_score(logits, roi, roi_valid=None, fraction=.05):
    if not 0 < fraction <= 1:
        raise ValueError("Top fraction must be in (0,1]")
    if logits.ndim == 4:
        logits = logits[:, 0]
    roi = F.adaptive_max_pool2d(roi.float()[:, None], logits.shape[-2:])[:, 0].bool()
    scores, valid_flags = [], []
    for i, (score, mask) in enumerate(zip(logits, roi)):
        valid = bool(mask.any()) and (roi_valid is None or bool(roi_valid[i]))
        valid_flags.append(valid)
        if valid:
            values = score[mask]
            scores.append(values.topk(max(1, math.ceil(values.numel() * fraction))).values.mean())
        else:
            scores.append(score.sum() * 0)  # validity flag excludes it from image metrics
    return torch.stack(scores), torch.tensor(valid_flags, device=logits.device)


class DenseLocalizer(nn.Module):
    VERSION = "dense_3level_gn_v1"

    def __init__(self, n_levels=3, input_dim=512, projection_dim=128, decoder_variant="legacy_gn2"):
        super().__init__()
        self.spec = {"n_levels": n_levels, "input_dim": input_dim, "projection_dim": projection_dim, "decoder_variant": decoder_variant}
        if decoder_variant not in {"legacy_gn2", "doc1027"}:
            raise ValueError("Unknown dense decoder architecture")
        self.projections = nn.ModuleList([nn.Linear(input_dim, projection_dim) for _ in range(n_levels)])
        layers = [nn.Conv2d(n_levels * projection_dim, 256, 3, padding=1),
                  nn.GroupNorm(32, 256), nn.GELU(), nn.Conv2d(256, 64, 3, padding=1)]
        if decoder_variant == "legacy_gn2":
            layers += [nn.GroupNorm(8, 64), nn.GELU()]
        self.decoder = nn.Sequential(*layers, nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False), nn.Conv2d(64, 1, 1))

    def forward(self, levels, grid):
        if levels.ndim != 4 or levels.shape[1] != len(self.projections) or levels.shape[2] != grid[0] * grid[1]:
            raise ValueError("Expected B,L,N,D features and matching grid")
        maps = [proj(levels[:, i]).transpose(1, 2).reshape(levels.shape[0], -1, *grid)
                for i, proj in enumerate(self.projections)]
        return self.decoder(torch.cat(maps, dim=1))
