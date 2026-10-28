"""推理期 2.5D 一致性：只在可信邻片上平滑连续分数图，不平均二值掩码。"""

from __future__ import annotations

from typing import Optional

import numpy as np


def smooth_three_slice(scores: np.ndarray, index, weights=(.2, .6, .2), final=.7):
    """A_smooth=0.2A(z-1)+0.6A(z)+0.2A(z+1); A_final=0.7A+0.3A_smooth.

    Endpoints, missing slices, or invalid neighbors fall back to the center map
    exactly.  ``scores`` may be a low-resolution or high-resolution continuous
    map; binary masks must not be passed here.
    """
    if scores.ndim != 3:
        raise ValueError("Consistency expects a (N,H,W) continuous score array")
    if len(weights) != 3 or abs(sum(weights) - 1.0) > 1e-6:
        raise ValueError("Three-slice weights must sum to one")
    if not 0 <= final <= 1:
        raise ValueError("Center weight must be in [0, 1]")
    smoothed = np.array(scores, copy=True)
    used = 0
    fallback = 0
    for i in range(scores.shape[0]):
        previous, following = index.neighbor(i)
        if previous is None or following is None:
            fallback += 1
            continue
        center = scores[i]
        smooth = (weights[0] * scores[previous] + weights[1] * center + weights[2] * scores[following])
        smoothed[i] = final * center + (1.0 - final) * smooth
        used += 1
    return smoothed, {
        "method": "three_slice_0.2_0.6_0.2",
        "merged_center_weight": float(final + (1 - final) * weights[1]),
        "merged_neighbor_weight": float((1 - final) * weights[0]),
        "n_slices": int(scores.shape[0]),
        "n_smoothed": int(used),
        "n_fallback": int(fallback),
        "available": bool(used > 0),
    }


def score_delta(scores: np.ndarray, index) -> Optional[dict]:
    """Mean absolute adjacent difference on valid pairs, for consistency diagnostics."""
    values = []
    for i in range(scores.shape[0]):
        previous, following = index.neighbor(i)
        if previous is None or following is None:
            continue
        values.append(float(np.abs(scores[i] - scores[previous]).mean()))
        values.append(float(np.abs(scores[following] - scores[i]).mean()))
    if not values:
        return None
    return {"mean_abs_adjacent_delta": float(np.mean(values)), "n_pairs": len(values)}
