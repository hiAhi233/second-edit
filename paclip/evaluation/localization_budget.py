"""Deterministic exact pixel budgets; GT denominator always includes outside-ROI GT."""
import math
import numpy as np


def top_budget_mask(score, roi, fraction=.05):
    score, roi = np.asarray(score), np.asarray(roi, dtype=bool)
    if score.shape != roi.shape or not np.isfinite(score).all() or not 0 < fraction <= 1:
        raise ValueError("Aligned finite score/ROI and valid fraction required")
    eligible = np.flatnonzero(roi)
    result = np.zeros(score.size, dtype=bool)
    if len(eligible):
        # Stable row-major index tie-breaking; exactly ceil(fraction*ROI size).
        order = np.argsort(-score.ravel()[eligible], kind="stable")
        result[eligible[order[:max(1, math.ceil(fraction * len(eligible)))]]] = True
    return result.reshape(score.shape)


def coverage_counts(prediction, truth):
    truth = np.asarray(truth, dtype=bool)
    return {"covered_gt_pixels": int((prediction & truth).sum()), "gt_pixels": int(truth.sum()),
            "selected_pixels": int(np.asarray(prediction).sum())}
