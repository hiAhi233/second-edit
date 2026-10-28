"""Keep native lesion masks until the model supplies its actual patch grid."""

import torch
import torch.nn.functional as F
from torch.utils.data import default_collate


def pool_mask_targets(masks, output_size, device=None):
    """Native (B,H,W) or a list of differently sized (H,W) masks -> soft/hard."""
    if torch.is_tensor(masks):
        if masks.ndim == 2:
            masks = masks.unsqueeze(0)
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
    soft, hard = [], []
    for mask in masks:
        high = torch.as_tensor(mask, device=device, dtype=torch.float32)
        if high.ndim != 2:
            raise ValueError("Each lesion mask must have shape (H, W)")
        if not torch.isfinite(high).all() or (high < 0).any() or (high > 1).any():
            raise ValueError("Lesion masks must contain finite values in [0, 1]")
        high = high[None, None]
        soft.append(F.adaptive_avg_pool2d(high, output_size)[0, 0])
        hard.append((F.adaptive_max_pool2d(high, output_size)[0, 0] > 0).float())
    return torch.stack(soft), torch.stack(hard)


def mask_fields(mask, grid):
    """Compatibility previews; training re-pools mask_high at the output size."""
    high = torch.from_numpy((mask > 0).astype("float32"))
    size = (grid, grid) if isinstance(grid, int) else tuple(grid)
    soft, hard = pool_mask_targets([high], size)
    return {"mask_high": high, "mask_soft": soft[0], "mask_hard": hard[0], "mask": hard[0]}


def collate_anomaly_samples(samples):
    """Stack images but retain native masks, whose dimensions can differ."""
    native = [get_supervision_masks(item) for item in samples]
    batch = default_collate([{k: v for k, v in item.items() if k != "mask_high"} for item in samples])
    batch["mask_high"] = native
    return batch


def training_masks(batch, device):
    masks = get_supervision_masks(batch)
    if torch.is_tensor(masks):
        return masks.to(device)
    return [mask.to(device) for mask in masks]


def get_supervision_masks(sample):
    for key in ("mask_high", "mask_soft", "mask_hard", "mask"):
        if key in sample:
            return sample[key]
    raise KeyError("Sample has no lesion mask")
