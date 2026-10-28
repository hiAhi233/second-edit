"""Exact AP/AUROC with one sort and bounded-size tie-group reductions."""

import numpy as np


def binary_ranking_metrics(scores, truth, chunk_size=1_000_000):
    scores = np.asarray(scores).reshape(-1)
    truth = np.asarray(truth, dtype=bool).reshape(-1)
    if scores.size != truth.size or not np.isfinite(scores).all():
        raise ValueError("Aligned finite scores and labels are required")
    if int(chunk_size) < 1:
        raise ValueError("chunk_size must be positive")
    total = len(truth)
    positives = int(truth.sum())
    negatives = total - positives
    if not positives:
        return {"pixel_ap": float("nan"), "pixel_auroc": float("nan")}
    order = np.argsort(scores, kind="quicksort")
    ordered_scores, ordered_truth = scores[order], truth[order]
    del order
    ap_sum = auc_sum = 0.0
    seen = seen_positive = 0
    while seen < total:
        trial = min(total, seen + int(chunk_size))
        if trial == total:
            end = total
        else:
            # Never split equal scores across reductions. A huge constant-score
            # group is reduced directly, without a cumulative array of its size.
            end = int(np.searchsorted(ordered_scores, ordered_scores[trial - 1], side="left"))
            if end <= seen:
                end = int(np.searchsorted(ordered_scores, ordered_scores[seen], side="right"))
        block = ordered_scores[seen:end]
        labels = ordered_truth[seen:end]
        if block[0] == block[-1]:
            counts = np.array([end - seen], dtype=np.int64)
            positive_counts = np.array([np.count_nonzero(labels)], dtype=np.int64)
        else:
            ends = np.r_[np.flatnonzero(block[1:] != block[:-1]) + 1, len(block)]
            counts = np.diff(np.r_[0, ends])
            cumulative = np.cumsum(labels, dtype=np.int64)
            positive_counts = np.diff(np.r_[0, cumulative[ends - 1]])
        before_count = seen + np.r_[0, np.cumsum(counts[:-1])]
        before_positive = seen_positive + np.r_[0, np.cumsum(positive_counts[:-1])]
        # Ascending threshold groups: AP uses all predictions >= current score.
        ap_sum += np.sum(positive_counts.astype(np.float64) * (positives - before_positive) / (total - before_count))
        below_negative = before_count - before_positive
        auc_sum += np.sum(positive_counts.astype(np.float64) * (below_negative + .5 * (counts - positive_counts)))
        seen, seen_positive = end, seen_positive + int(positive_counts.sum())
    return {"pixel_ap": float(ap_sum / positives),
            "pixel_auroc": float(auc_sum / (positives * negatives)) if negatives else float("nan")}
