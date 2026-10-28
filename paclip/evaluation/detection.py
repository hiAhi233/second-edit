"""Slice-level connected-lesion FROC at source-selected thresholds."""

import numpy as np
from scipy.ndimage import label


def detection_counts(score, gt, valid, thresholds, min_coverage=.1):
    gt = np.asarray(gt, dtype=bool) & valid
    if score.ndim not in (2, 3) or gt.shape != score.shape or valid.shape != score.shape:
        raise ValueError("Detection expects aligned 2D or 3D arrays")
    connectivity = np.ones((3,) * score.ndim)
    components, count = label(gt, connectivity)
    areas = np.bincount(components.ravel(), minlength=count + 1)[1:]
    rows = []
    for threshold in thresholds:
        prediction, n_pred = label((score >= threshold) & valid, connectivity)
        if count and n_pred:
            pairs = prediction[gt] * (count + 1) + components[gt]
            overlap = np.bincount(pairs, minlength=(n_pred + 1) * (count + 1)).reshape(n_pred + 1, count + 1)[1:, 1:]
            matches = overlap >= np.maximum(1, np.ceil(areas * min_coverage))[None]
            detected = int(matches.any(0).sum())
            false_positive = int((~matches.any(1)).sum())
        else:
            detected, false_positive = 0, n_pred
        rows.append({"lesions": count, "detected": detected, "false_positive_components": false_positive,
                     "predicted_components": n_pred})
    return rows


def best_dice_threshold(scores, masks):
    """Exact validation Dice selection, ties prefer the lower threshold."""
    scores, masks = np.asarray(scores).ravel(), np.asarray(masks).ravel().astype(bool)
    if not masks.any() or masks.all():
        raise ValueError("Threshold selection requires lesion and background validation pixels")
    order = np.argsort(scores, kind="quicksort")[::-1]
    values, positive = scores[order], masks[order]
    ends = np.r_[np.flatnonzero(values[1:] != values[:-1]), len(values) - 1]
    true_positive = np.cumsum(positive, dtype=np.int64)[ends]
    dice = 2 * true_positive / (ends + 1 + int(masks.sum()) + 1e-6)
    best = np.flatnonzero(dice == dice.max())[-1]
    return float(values[ends[best]])
