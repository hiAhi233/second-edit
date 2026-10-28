"""Per-image, per-source normalized losses and per-connected-lesion ranking."""
import math
import numpy as np
from scipy.ndimage import label
import torch
import torch.nn.functional as F
from paclip.data.supervision import pool_mask_targets
from paclip.losses import binary_local_loss, focal_bce


def lesion_rank_loss(logits, mask, roi, margin=.5, hard_fraction=.05, mode="per_lesion_hard"):
    """Connected components are defined before pooling, preserving tiny lesions."""
    zero = logits.sum() * 0
    valid = F.adaptive_max_pool2d(roi.float()[None, None], logits.shape)[0, 0] > 0
    _, all_positive = pool_mask_targets([mask], logits.shape, logits.device)
    negatives = valid & ~all_positive[0].bool()
    components, n = label(torch.as_tensor(mask).detach().cpu().numpy() > 0, np.ones((3, 3)))
    stats = {"lesions": int(n), "valid_pairs": 0, "no_negative": 0, "outside_roi": 0}
    if not negatives.any():
        stats["no_negative"] = int(n)
        return zero, stats
    negative_scores = logits[negatives]
    if mode == "global_mean":
        positive = all_positive[0].bool() & valid
        if not positive.any():
            stats["outside_roi"] = int(n)
            return zero, stats
        stats["valid_pairs"] = 1
        return F.relu(margin - logits[positive].mean() + negative_scores.mean()), stats
    if mode != "per_lesion_hard":
        raise ValueError("Unknown ranking definition")
    hard_score = negative_scores.topk(max(1, math.ceil(len(negative_scores) * hard_fraction))).values.mean()
    losses = []
    for k in range(1, n + 1):
        _, target = pool_mask_targets([components == k], logits.shape, logits.device)
        positive = target[0].bool() & valid
        if not positive.any():
            stats["outside_roi"] += 1
            continue
        losses.append(F.relu(margin - logits[positive].mean() + hard_score))
    stats["valid_pairs"] = len(losses)
    return (torch.stack(losses).mean() if losses else zero), stats


def grouped_novelty_loss(logits, masks, rois, kinds, synthetic_types, *, lambda_syn=1.,
                         lambda_normal=.25, lambda_rank=0., margin=.5,
                         boundary_weight=.1, recall_weight=.1, rank_mode="per_lesion_hard", rank_aggregation="source_sum"):
    if logits.ndim == 4 and logits.shape[1] == 1:
        logits = logits[:, 0]
    if logits.ndim != 3 or not all(len(x) == len(logits) for x in (masks, rois, kinds, synthetic_types)):
        raise ValueError("Aligned logits, native masks, ROIs and source labels required")
    groups = {k: [] for k in ("normal", "real", "syn_image", "syn_feature")}
    ranks = {k: [] for k in ("real", "synthetic")}
    zero = logits.sum() * 0
    rank_stats = {"lesions": 0, "valid_pairs": 0, "no_negative": 0, "outside_roi": 0}
    diagnostics = []
    for i, score in enumerate(logits.float()):
        mask, roi = torch.as_tensor(masks[i], device=score.device), rois[i].to(score.device)
        valid = F.adaptive_max_pool2d(roi.float()[None, None], score.shape)[0, 0].bool()
        if not valid.any():
            raise ValueError("Training cannot silently skip an invalid/empty ROI")
        if kinds[i] == "normal":
            if mask.any() or synthetic_types[i] != "none":
                raise ValueError("Normal supervision must be empty and unmodified")
            value = focal_bce(score[None], torch.zeros_like(score)[None], valid[None])
            group = "normal"
        elif kinds[i] in ("real", "synthetic"):
            if not mask.any():
                raise ValueError("Positive training instance has an empty mask")
            group = "real" if kinds[i] == "real" else "syn_" + synthetic_types[i]
            if group not in groups:
                raise ValueError("Unknown synthetic subtype")
            value = binary_local_loss(score[None], [mask], roi[None],
                                      boundary_weight=boundary_weight, recall_weight=recall_weight)
            rank, counts = lesion_rank_loss(score, mask, roi, margin, mode=rank_mode)
            ranks[kinds[i]].append(rank)
            for key in rank_stats:
                rank_stats[key] += counts[key]
        else:
            raise ValueError("Unknown source kind")
        groups[group].append(value)
        _, hard = pool_mask_targets([mask], score.shape, score.device)
        positive = hard[0].bool()
        background = valid & ~positive
        diagnostics.append({"source_kind": kinds[i], "synthetic_type": synthetic_types[i],
            "local_loss": float(value.detach()),
            "mean_logit": float(score[valid].detach().mean()),
            "negative_focal": float(focal_bce(score[None], torch.zeros_like(score)[None], background[None]).detach()),
            "normal_fp_area_at_logit0": float(((score >= 0) & valid).sum().detach() / valid.sum()) if group == "normal" else None,
            "positive_pixel_recall_at_logit0": float(((score >= 0) & positive & valid).sum().detach() / positive.sum().clamp_min(1)) if group != "normal" else None})
    means = {key: torch.stack(values).mean() if values else zero for key, values in groups.items()}
    syn_values = groups["syn_image"] + groups["syn_feature"]
    syn = torch.stack(syn_values).mean() if syn_values else zero
    rank_means = {k: torch.stack(v).mean() if v else zero for k, v in ranks.items()}
    if rank_aggregation == "positive_image_mean":
        rank_values = ranks["real"] + ranks["synthetic"]
        rank = torch.stack(rank_values).mean() if rank_values else zero
    elif rank_aggregation == "source_sum":
        rank = rank_means["real"] + lambda_syn * rank_means["synthetic"]
    else:
        raise ValueError("Unknown rank aggregation")
    total = means["real"] + lambda_syn * syn + lambda_normal * means["normal"] + lambda_rank * rank
    return {"total": total, "loss_syn": syn, "loss_rank": rank,
            **{"loss_" + k: v for k, v in means.items()},
            "loss_rank_real": rank_means["real"], "loss_rank_syn": rank_means["synthetic"],
            "rank_stats": rank_stats, "diagnostics": diagnostics,
            "counts": {k: len(v) for k, v in groups.items()}}


@torch.no_grad()
def source_localization_metrics(logits, masks, rois, kinds, synthetic_types, threshold=0.):
    """Training diagnostics: 224-grid connected-lesion recall, not pixel recall.

    Prediction is constrained to ROI; native GT survives max pooling and remains
    in the denominator even outside ROI. Zero-logit threshold is diagnostic only.
    """
    from paclip.evaluation.novelty_protocol import lesion_counts
    size = tuple(rois.shape[-2:])
    if logits.ndim == 3:
        logits = logits[:, None]
    scores = F.interpolate(logits.float(), size, mode="bilinear", align_corners=False)[:, 0].cpu().numpy()
    stats = {k: {"lesions": 0, "detected": 0} for k in ("real", "synthetic", "syn_image", "syn_feature")}
    normal_fp, normal_area, normal_images = 0, 0, 0
    for i, score in enumerate(scores):
        _, gt = pool_mask_targets([masks[i]], size, device="cpu")
        roi = rois[i].detach().cpu().numpy().astype(bool)
        pred = (score >= threshold) & roi
        if kinds[i] == "normal":
            normal_fp += int(pred.sum()); normal_area += int(roi.sum()); normal_images += 1
        else:
            counts = lesion_counts(pred, gt[0].numpy().astype(bool))
            keys = [kinds[i]] if kinds[i] == "real" else ["synthetic", "syn_" + synthetic_types[i]]
            for key in keys:
                stats[key]["lesions"] += counts["lesions"]
                stats[key]["detected"] += counts["detected"]
    ratios = {k: v["detected"] / v["lesions"] if v["lesions"] else None for k, v in stats.items()}
    return {"real_lesion_recall": ratios["real"], "synthetic_lesion_recall": ratios["synthetic"],
            "synthetic_image_lesion_recall": ratios["syn_image"], "synthetic_feature_lesion_recall": ratios["syn_feature"],
            "normal_false_positive_area": normal_fp / normal_area if normal_area else None,
            "normal_false_positive_pixels": normal_fp, "normal_roi_pixels": normal_area, "normal_images": normal_images,
            "localization_counts": stats,
            "localization_metric_definition": {"threshold_logit": threshold, "grid": list(size),
                "lesion_coverage_required": .1, "connectivity": 8,
                "normal_false_positive_area": "sum_fp_pixels / sum_normal_ROI_pixels",
                "purpose": "training_diagnostic_not_source_selected_test_operating_point"}}
