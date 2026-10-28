"""Detached multi-layer feature synthesis with relative L2, never CLS relabeling."""
import numpy as np
import torch
import torch.nn.functional as F


def synthesize_feature(levels, grid, roi, seed, *, roi_valid=True, rhos=(.05, .1, .2), max_patches=20,
                       noise_mode="relative_l2", detach_reference=True):
    # Single instance L,N,D. Keep CPU RNG independent of device and global RNG.
    if levels.ndim != 3 or levels.shape[1] != grid[0] * grid[1] or not torch.isfinite(levels).all():
        raise ValueError("Expected finite (levels, patches, channels) features and matching grid")
    valid = F.adaptive_max_pool2d(torch.as_tensor(roi).float()[None, None], grid)[0, 0].bool().cpu().numpy()
    coords = np.argwhere(valid)
    if not roi_valid or len(coords) == 0:
        raise ValueError("Synthetic features require trusted nonempty ROI")
    if max_patches < 1 or not rhos or min(rhos) <= 0:
        raise ValueError("Invalid perturbation parameters")
    if noise_mode not in {"relative_l2", "per_dimension_gaussian"}:
        raise ValueError("Unknown feature noise definition")
    rng = np.random.default_rng(seed)
    count = int(rng.integers(1, min(max_patches, len(coords)) + 1))
    n_regions = int(rng.integers(1, min(count, 5) + 1))
    centers = coords[rng.choice(len(coords), n_regions, replace=False)]
    aspect = float(rng.choice([.35, 1., 2.8]))
    dist = np.min((((coords[:, None, :] - centers[None]) * [aspect, 1 / aspect]) ** 2).sum(-1), axis=1)
    selected = coords[np.argsort(dist, kind="stable")[:count]]
    mask = np.zeros(grid, np.float32)
    mask[selected[:, 0], selected[:, 1]] = 1
    mask = torch.from_numpy(mask).to(levels.device)
    base = levels.detach() if detach_reference else levels
    noise = torch.from_numpy(rng.standard_normal(tuple(base.shape)).astype(np.float32)).to(base.device, base.dtype)
    direction = F.normalize(noise, dim=-1, eps=1e-8)
    rho = float(rng.choice(rhos))
    if noise_mode == "per_dimension_gaussian":
        delta = noise * rho * mask.flatten()[None, :, None]
    else:
        delta = direction * rho * base.norm(dim=-1, keepdim=True) * mask.flatten()[None, :, None]
    output = base + delta
    active = mask.flatten().bool()
    if (base[:, active].norm(dim=-1) <= 1e-8).any():
        raise ValueError("Selected normal reference has zero feature length; no-op synthesis is prohibited")
    cosine = F.cosine_similarity(base[:, active], output[:, active], dim=-1)
    return {"levels": output, "mask_high": mask,
            "perturbation_norm": delta.norm(dim=-1).reshape(base.shape[0], *grid),
            "parameters": {"seed": int(seed), "noise_mode": noise_mode,
                "rho_relative_l2": rho if noise_mode == "relative_l2" else None,
                "sigma_per_dimension": rho if noise_mode == "per_dimension_gaussian" else None, "patches": count,
                "renormalized": False, "reference_detached": bool(detach_reference),
                "length_before_per_level": base[:, active].norm(dim=-1).mean(1).cpu().tolist(),
                "length_after_per_level": output[:, active].norm(dim=-1).mean(1).cpu().tolist(),
                "cosine_per_level": cosine.mean(1).cpu().tolist()}}
