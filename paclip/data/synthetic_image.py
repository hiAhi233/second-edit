"""Compact-support image perturbations in [0,1], before CLIP normalization."""
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_edt, map_coordinates
import torch


@dataclass(frozen=True)
class ImageSyntheticConfig:
    area_min: float = .001
    area_max: float = .10
    alpha_min: float = .2
    alpha_max: float = .8
    max_regions: int = 5
    attempts: int = 24
    min_mean_change: float = .003
    # 1028 first-stage defaults; the other modes are explicit ablations.
    modes: tuple = ("bright", "dark", "smooth_noise", "blur_mix")


def synthesize_image(gray, roi, seed, *, roi_valid=True, config=None):
    cfg = config or ImageSyntheticConfig()
    x = np.asarray(torch.as_tensor(gray).cpu(), dtype=np.float32)
    valid = np.asarray(torch.as_tensor(roi).cpu(), dtype=bool)
    if x.ndim != 2 or valid.shape != x.shape or not np.isfinite(x).all() or x.min() < 0 or x.max() > 1:
        raise ValueError("Expected aligned finite grayscale [0,1] and ROI")
    if not roi_valid or valid.sum() < 3:
        raise ValueError("Synthetic image requires a trusted nonempty ROI")
    if not (0 < cfg.area_min <= cfg.area_max < 1 and 0 < cfg.alpha_min <= cfg.alpha_max <= 1):
        raise ValueError("Invalid synthesis ranges")
    allowed = {"bright", "dark", "smooth_noise", "gaussian_blur", "local_noise", "blur_mix", "texture_replace", "dilate", "compress"}
    if not cfg.modes or set(cfg.modes) - allowed or len(set(cfg.modes)) != len(cfg.modes):
        raise ValueError("Invalid/duplicated image synthesis modes")
    if cfg.attempts < 1 or cfg.max_regions < 1 or cfg.min_mean_change <= 0:
        raise ValueError("Synthesis attempts, regions and visibility threshold must be positive")
    rng = np.random.default_rng(seed)
    coords = np.argwhere(valid)
    yy, xx = np.indices(x.shape)
    lo, hi = int(np.ceil(x.size * cfg.area_min)), int(np.floor(x.size * cfg.area_max))
    hi = min(hi, len(coords))
    if hi < lo:
        raise ValueError("ROI cannot support requested synthetic area")
    for attempt in range(cfg.attempts):
        count = int(rng.integers(lo, hi + 1))
        regions = int(rng.integers(1, min(cfg.max_regions, count) + 1))
        field = np.full(x.shape, np.inf)
        for cy, cx in coords[rng.choice(len(coords), regions, replace=False)]:
            aspect = float(rng.uniform(.35, 2.8))
            field = np.minimum(field, ((yy - cy) * aspect) ** 2 + ((xx - cx) / aspect) ** 2)
        eligible = np.flatnonzero(valid)
        order = np.argsort(field.ravel()[eligible], kind="stable")
        hard = np.zeros(x.size, dtype=bool)
        hard[eligible[order[:count]]] = True
        hard = hard.reshape(x.shape)
        # Distance taper is compact: outside support is EXACTLY zero.
        width = float(rng.uniform(2, 8) * x.shape[0] / 224)
        blend = np.minimum(distance_transform_edt(hard) / max(width, 1), 1).astype(np.float32) * hard
        alpha = float(rng.uniform(cfg.alpha_min, cfg.alpha_max))
        mode = str(rng.choice(cfg.modes))
        transform_parameters = {}
        amplitude = float(rng.uniform(.08, .45))
        if mode == "bright":
            transformed = np.clip(x + amplitude, 0, 1)
        elif mode == "dark":
            transformed = np.clip(x - amplitude, 0, 1)
        elif mode == "smooth_noise":
            noise = gaussian_filter(rng.standard_normal(x.shape), sigma=max(1, x.shape[0] / 56))
            noise = noise / max(float(np.abs(noise).max()), 1e-6)
            transformed = np.clip(x + amplitude * noise, 0, 1)
        elif mode == "blur_mix":
            transformed = np.clip(gaussian_filter(x, max(1, x.shape[0] / 56)) + rng.choice([-1, 1]) * amplitude, 0, 1)
        elif mode == "gaussian_blur":
            sigma = float(rng.uniform(1, 4) * x.shape[0] / 224)
            transformed = gaussian_filter(x, sigma)
            transform_parameters = {"blur_sigma_pixels": sigma}
        elif mode == "local_noise":
            transformed = np.clip(x + rng.standard_normal(x.shape).astype(np.float32) * amplitude, 0, 1)
        else:
            if mode == "texture_replace":
                # Reuse actual texture from the same NORMAL support image, never target data.
                dy, dx = rng.integers(-max(2, x.shape[0] // 4), max(3, x.shape[0] // 4), size=2)
                source_y, source_x = yy + int(dy), xx + int(dx)
                transform_parameters = {"texture_source": "same_normal_support_image", "offset_yx": [int(dy), int(dx)]}
            else:
                cy, cx = np.argwhere(hard).mean(0)
                scale = float(rng.uniform(1.03, 1.10) if mode == "dilate" else rng.uniform(.90, .97))
                source_y, source_x = cy + (yy - cy) / scale, cx + (xx - cx) / scale
                transform_parameters = {"spatial_scale": scale, "center_yx": [float(cy), float(cx)]}
            y0, x0 = np.floor(source_y).astype(int), np.floor(source_x).astype(int)
            interior = (y0 >= 0) & (y0 + 1 < x.shape[0]) & (x0 >= 0) & (x0 + 1 < x.shape[1])
            safe_y, safe_x = y0.clip(0, x.shape[0] - 2), x0.clip(0, x.shape[1] - 2)
            # All interpolation neighbours must be within reviewed normal ROI.
            interior &= valid[safe_y, safe_x] & valid[safe_y + 1, safe_x] & valid[safe_y, safe_x + 1] & valid[safe_y + 1, safe_x + 1]
            warped = map_coordinates(x, [source_y, source_x], order=1, mode="nearest", prefilter=False)
            transformed = np.where(interior, warped, x).astype(np.float32)
        result = ((1 - alpha * blend) * x + alpha * blend * transformed).astype(np.float32)
        change = np.abs(result - x)
        # Clipping can leave saturated pixels unchanged. Do not label those as synthetic lesions.
        hard &= change > 1e-6
        if int(hard.sum()) < lo:
            continue
        blend *= hard
        result[~hard] = x[~hard]
        if float(change[hard].mean()) < cfg.min_mean_change:
            continue
        return {"gray": torch.from_numpy(result), "mask_high": torch.from_numpy(hard.astype(np.float32)),
                "blend_mask": torch.from_numpy(blend), "parameters": {
                    "seed": int(seed), "mode": mode, "alpha": alpha, "amplitude": amplitude,
                    "area_fraction": float(hard.mean()), "regions_requested": regions,
                    "edge_width_pixels": width, "mean_abs_change": float(change[hard].mean()),
                    **transform_parameters,
                    "failed_attempts": attempt}}
    raise RuntimeError(f"No visible valid image anomaly after {cfg.attempts} attempts; quota not substituted")
