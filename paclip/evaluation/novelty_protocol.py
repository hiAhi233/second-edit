"""Source-only thresholds, explicit denominators, complete real-data metrics."""
from collections import defaultdict
import math
import numpy as np
from scipy.ndimage import label
from paclip.evaluation.detection import best_dice_threshold
from paclip.evaluation.ranking import binary_ranking_metrics
from paclip.evaluation.localization_budget import top_budget_mask


def safe_ratio(a, b):
    return float(a / b) if b else None


def development_guard(candidate, reference, recall_tolerance=.02, fp_relative=.1, fp_absolute=.001):
    """Engineering development guard, not a statistical/clinical noninferiority test."""
    keys = ("small_lesion_recall", "normal_fp_area_ratio")
    if any(x.get(k) is None or not np.isfinite(x[k]) for x in (candidate, reference) for k in keys):
        return {"passed": False, "reason": "guard_metric_unavailable"}
    min_recall = reference["small_lesion_recall"] - recall_tolerance
    max_fp = reference["normal_fp_area_ratio"] + max(reference["normal_fp_area_ratio"] * fp_relative, fp_absolute)
    return {"passed": bool(candidate["small_lesion_recall"] + 1e-12 >= min_recall and candidate["normal_fp_area_ratio"] <= max_fp + 1e-12),
            "minimum_small_lesion_recall": min_recall, "maximum_normal_fp_area": max_fp,
            "candidate": {k: candidate[k] for k in keys}, "reference": {k: reference[k] for k in keys},
            "meaning": "predeclared_development_guard_not_statistical_noninferiority"}


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def lesion_counts(prediction, truth, min_coverage=.1):
    """Prediction may be ROI-limited, but do NOT erase GT outside the ROI."""
    prediction, truth = np.asarray(prediction, bool), np.asarray(truth, bool)
    if prediction.shape != truth.shape or truth.ndim not in (2, 3):
        raise ValueError("Aligned 2D/3D binary arrays required")
    cc, n = label(truth, np.ones((3,) * truth.ndim))
    pc, npred = label(prediction, np.ones((3,) * truth.ndim))
    areas = np.bincount(cc.ravel(), minlength=n + 1)[1:]
    detected = np.zeros(n, bool)
    matched = np.zeros(npred, bool)
    for i in range(1, n + 1):
        overlap = np.bincount(pc[cc == i], minlength=npred + 1)[1:]
        hits = overlap >= max(1, math.ceil(int(areas[i - 1]) * min_coverage))
        detected[i - 1] = hits.any()
        matched |= hits
    small = areas <= truth.size * .01
    return {"lesions": int(n), "detected": int(detected.sum()),
            "small_lesions": int(small.sum()), "small_detected": int((small & detected).sum()),
            "false_positive_components": int((~matched).sum()), "predicted_components": int(npred)}


def validate_items(items):
    if not items:
        raise ValueError("No evaluation samples")
    for r in items:
        s, g, roi = np.asarray(r["score"]), np.asarray(r["gt"]), np.asarray(r["roi"])
        if s.ndim != 2 or s.shape != g.shape or s.shape != roi.shape or not np.isfinite(s).all():
            raise ValueError("Evaluation needs aligned finite 2D score, GT and ROI")


def fit_operating_points(items, normal_budget=.01):
    validate_items(items)
    if any(r["split"] != "validation" for r in items):
        raise ValueError("Operating points may be fitted only on source validation")
    if not 0 <= normal_budget < 1:
        raise ValueError("Invalid normal false-positive budget")
    scores = np.concatenate([r["score"].ravel() for r in items])
    gt = np.concatenate([r["gt"].ravel() for r in items]).astype(bool)
    dice = best_dice_threshold(scores, gt)
    normal = [r["score"][r["roi"]] for r in items if not r["label"] and r["roi_valid"] and r["roi"].any()]
    if not normal:
        raise ValueError("Normal ROI budget requires valid source normal validation images")
    normal = np.concatenate(normal)
    allowed = int(np.floor(len(normal) * normal_budget))
    ordered = np.sort(normal)[::-1]
    # Strictly above the (allowed+1)-th score avoids tied-score budget overshoot.
    boundary = ordered[min(allowed, len(ordered) - 1)]
    budget_threshold = float(np.nextafter(boundary, np.asarray(np.inf, dtype=normal.dtype)))
    # Dense near low-FP tail, plus a zero-prediction endpoint above all source scores.
    tail = np.r_[np.geomspace(1e-6, .1, 64), np.linspace(.1, 1, 65)]
    froc = sorted(set(map(float, np.quantile(normal, 1 - tail))) | {
        dice, budget_threshold, float(np.nextafter(scores.max(), np.asarray(np.inf, dtype=scores.dtype)))}, reverse=True)
    image_scores = np.asarray([r["image_score"] for r in items if r["roi_valid"]])
    image_truth = np.asarray([r["label"] for r in items if r["roi_valid"]], bool)
    image_threshold = best_dice_threshold(image_scores, image_truth)
    return {"version": "novelty_protocol_v1", "score_space": "raw_dense_logit", "fit_split": "validation",
            "dice_threshold": dice, "normal_budget_threshold": budget_threshold,
            "normal_budget_target": normal_budget, "normal_budget_actual": float((normal >= budget_threshold).mean()),
            "normal_calibration_pixels": len(normal), "froc_thresholds": froc,
            "image_threshold": image_threshold,
            "dice_selection": "full_224_grid_micro_dice_lower_threshold_on_tie",
            "prediction_rule": "score >= threshold; ROI reporting is separate"}


def _segmentation(items, threshold, restrict_roi):
    tp = fp = fn = 0
    macro = []
    for r in items:
        if restrict_roi and not r["roi_valid"]:
            continue
        valid = r["roi"] if restrict_roi else np.ones_like(r["gt"], bool)
        pred, gt = (r["score"] >= threshold) & valid, r["gt"] & valid
        a, b, c = int((pred & gt).sum()), int((pred & ~gt).sum()), int((~pred & gt).sum())
        tp += a; fp += b; fn += c
        macro.append(2 * a / (2 * a + b + c) if 2 * a + b + c else 1.)
    return {"dice_micro": safe_ratio(2 * tp, 2 * tp + fp + fn), "iou_micro": safe_ratio(tp, tp + fp + fn),
            "dice_macro": float(np.mean(macro)) if macro else None,
            "tp_pixels": tp, "fp_pixels": fp, "fn_pixels": fn, "images": len(macro)}


def _detection(items, threshold):
    totals = defaultdict(int)
    normal_pixels = normal_fp = normal_count = normal_components = 0
    for r in items:
        # Invalid ROI is explicitly a missed detection, not dropped GT.
        pred = (r["score"] >= threshold) & r["roi"] if r["roi_valid"] else np.zeros_like(r["gt"])
        counts = lesion_counts(pred, r["gt"])
        for k, v in counts.items():
            totals[k] += v
        if r["label"] == 0 and r["roi_valid"]:
            normal_count += 1
            normal_pixels += int(r["roi"].sum())
            normal_fp += int(pred.sum())
            normal_components += counts["predicted_components"]
    return {**totals, "lesion_recall": safe_ratio(totals["detected"], totals["lesions"]),
            "small_lesion_recall": safe_ratio(totals["small_detected"], totals["small_lesions"]),
            "fp_components_per_image": safe_ratio(totals["false_positive_components"], len(items)),
            "normal_fp_area_ratio": safe_ratio(normal_fp, normal_pixels),
            "normal_fp_pixels": normal_fp, "normal_roi_pixels": normal_pixels,
            "normal_fp_components_per_image": safe_ratio(normal_components, normal_count),
            "valid_normal_images": normal_count}


def evaluate_volumes(items, thresholds):
    groups = defaultdict(list)
    for r in items:
        groups[(r["dataset"], r["patient_id"], r.get("study_id"), r.get("volume_id"))].append(r)
    volumes, unavailable = [], []
    for key, rows in groups.items():
        zs = [r.get("slice_z") for r in rows]
        ok = (key[-1] and all(r.get("orientation_verified") is True and r.get("volume_complete") is True for r in rows)
              and all(isinstance(z, int) for z in zs) and len(set(zs)) == len(zs)
              and all(r.get("volume_depth") == len(rows) for r in rows)
              and sorted(zs) == list(range(len(rows)))
              and len({r["score"].shape for r in rows}) == 1)
        if not ok:
            unavailable.append({"volume": list(key), "reason": "missing verified complete ordered volume metadata, volume_depth, or contiguous indices 0..depth-1"})
            continue
        rows = sorted(rows, key=lambda r: r["slice_z"])
        volumes.append((np.stack([r["score"] for r in rows]), np.stack([r["gt"] for r in rows]),
                        np.stack([r["roi"] if r["roi_valid"] else np.zeros_like(r["roi"]) for r in rows])))
    curve = []
    for threshold in thresholds:
        totals = defaultdict(int)
        for scores, gt, roi in volumes:
            for k, v in lesion_counts((scores >= threshold) & roi, gt).items():
                totals[k] += v
        curve.append({"threshold": threshold, **totals,
                      "lesion_recall": safe_ratio(totals["detected"], totals["lesions"]),
                      "fp_per_volume": safe_ratio(totals["false_positive_components"], len(volumes))})
    return {"valid_volumes": len(volumes), "unavailable_volumes": unavailable,
            "grid": "canonical_224_stack_not_native_physical_volume", "curve": curve}


def evaluate(items, operating_points, include_oracle=True):
    validate_items(items)
    if operating_points.get("fit_split") != "validation" or operating_points.get("score_space") != "raw_dense_logit":
        raise ValueError("Expected saved source-validation logit operating points")
    scores = np.concatenate([r["score"].ravel() for r in items])
    gt = np.concatenate([r["gt"].ravel() for r in items]).astype(bool)
    roi_items = [r for r in items if r["roi_valid"] and r["roi"].any()]
    result = {"samples": len(items), "patients": len({r["patient_id"] for r in items}),
              "roi_invalid_images": len(items) - len(roi_items), "full_image_ranking": binary_ranking_metrics(scores, gt),
              "roi_ranking": binary_ranking_metrics(np.concatenate([r["score"][r["roi"]] for r in roi_items]),
                  np.concatenate([r["gt"][r["roi"]] for r in roi_items])) if roi_items else None,
              "roi_gt_coverage": safe_ratio(sum(int((r["gt"] & r["roi"]).sum()) for r in roi_items), int(gt.sum())),
              "denominators": {"full_image": "all working-grid pixels", "roi": "valid ROI pixels only",
                  "lesion_recall_and_topk": "all GT including outside ROI; invalid ROI counts as missed",
                  "normal_fp_area": "pooled false-positive pixels / pooled valid normal ROI pixels",
                  "empty_pair_dice_macro": 1.0}, "operating_points": {}}
    for name in ("dice", "normal_budget"):
        threshold = operating_points[name + "_threshold"]
        result["operating_points"][name] = {"threshold_logit": threshold,
            "threshold_probability": float(1 / (1 + np.exp(-np.clip(threshold, -700, 700)))),
            "full_image_segmentation": _segmentation(items, threshold, False),
            "roi_segmentation": _segmentation(items, threshold, True),
            "lesion_detection": _detection(items, threshold)}
    result["top_budget"] = {}
    for fraction in (.01, .05, .10):
        covered = selected = 0
        totals = defaultdict(int)
        for r in items:
            pred = top_budget_mask(r["score"], r["roi"], fraction) if r["roi_valid"] else np.zeros_like(r["gt"])
            covered += int((pred & r["gt"]).sum()); selected += int(pred.sum())
            for k, v in lesion_counts(pred, r["gt"]).items():
                totals[k] += v
        result["top_budget"][str(fraction)] = {"gt_pixel_coverage": safe_ratio(covered, int(gt.sum())),
            "selected_pixels": selected, **totals,
            "lesion_recall": safe_ratio(totals["detected"], totals["lesions"]),
            "small_lesion_recall": safe_ratio(totals["small_detected"], totals["small_lesions"])}
    valid_images = [r for r in items if r["roi_valid"]]
    imscores = np.asarray([r["image_score"] for r in valid_images])
    imgt = np.asarray([r["label"] for r in valid_images], bool)
    ranking = binary_ranking_metrics(imscores, imgt)
    impred = imscores >= operating_points["image_threshold"]
    tp, fp, fn = int((impred & imgt).sum()), int((impred & ~imgt).sum()), int((~impred & imgt).sum())
    result["image_detection"] = {"auroc": ranking["pixel_auroc"], "ap": ranking["pixel_ap"],
        "f1": safe_ratio(2 * tp, 2 * tp + fp + fn), "accuracy": float((impred == imgt).mean()) if len(imgt) else None,
        "threshold_logit": operating_points["image_threshold"], "valid_images": len(imgt)}
    result["froc_2d"] = [{"threshold": t, **_detection(items, t)} for t in operating_points["froc_thresholds"]]
    result["froc_2d_coverage"] = {"minimum_fp_per_image": min(r["fp_components_per_image"] for r in result["froc_2d"]),
        "maximum_fp_per_image": max(r["fp_components_per_image"] for r in result["froc_2d"]),
        "thresholds_from_source_only": True,
        "recall_at_fp_budget": {str(b): max((r["lesion_recall"] for r in result["froc_2d"]
            if r["fp_components_per_image"] <= b and r["lesion_recall"] is not None), default=None) for b in (.5, 1., 2., 4.)}}
    result["volumetric"] = evaluate_volumes(items, [operating_points["dice_threshold"],
        operating_points["normal_budget_threshold"], *operating_points["froc_thresholds"]])
    patients = defaultdict(list)
    for r in items:
        patients[r["patient_id"]].append(r)
    result["per_patient"] = {pid: {"slices": len(rs), "segmentation": _segmentation(rs, operating_points["dice_threshold"], False),
        "detection": _detection(rs, operating_points["dice_threshold"])} for pid, rs in patients.items()}
    if include_oracle and gt.any() and not gt.all():
        threshold = best_dice_threshold(scores, gt)
        result["oracle_diagnostic_only"] = {"threshold_logit": threshold,
            "full_image_segmentation": _segmentation(items, threshold, False), "used_for_selection": False}
    return clean_json(result)


def check_condition(source_rows, target_rows, condition):
    """Reject ambiguous claims, while allowing explicitly exploratory evaluation."""
    if any(r["split"] != "test" for r in target_rows):
        raise ValueError("Target evaluation requires test records")
    development = [r for r in source_rows if r["split"] != "test"]
    if {r["patient_id"] for r in development} & {r["patient_id"] for r in target_rows}:
        raise ValueError("Target patient was used in training/validation")
    srcseq, dstseq = {r["sequence"] for r in development}, {r["sequence"] for r in target_rows}
    if condition not in {"same_sequence_seen", "same_sequence_unseen", "external_same_sequence", "cross_sequence", "exploratory"}:
        raise ValueError("Unknown evaluation condition")
    if condition in {"same_sequence_seen", "same_sequence_unseen", "external_same_sequence"} and dstseq != srcseq:
        raise ValueError("Same-sequence condition has a sequence mismatch")
    if condition == "cross_sequence" and srcseq & dstseq:
        raise ValueError("Cross-sequence condition contains source sequences")
    if condition == "external_same_sequence" and {r["dataset"] for r in development} & {r["dataset"] for r in target_rows}:
        raise ValueError("External condition is not an independent data source")
    if condition in {"same_sequence_seen", "same_sequence_unseen"}:
        positives = [r for r in development + target_rows if r["label"]]
        if any(not r.get("disease_verified") or "unknown" in r["disease_labels"] for r in positives):
            raise ValueError("Seen/unseen disease claims require verified disease metadata")
        train_disease = {d for r in source_rows if r["split"] == "train" and r["label"] for d in r["disease_labels"]}
        dev_disease = {d for r in development if r["label"] for d in r["disease_labels"]}
        test_disease = {d for r in target_rows if r["label"] for d in r["disease_labels"]}
        if condition == "same_sequence_seen" and not test_disease <= train_disease:
            raise ValueError("Seen disease set contains unseen diseases")
        if condition == "same_sequence_unseen" and test_disease & dev_disease:
            raise ValueError("Final disease occurred in training/development")
        if condition == "same_sequence_unseen" and any(len(r["disease_labels"]) > 1 and not r.get("mask_disease_verified") for r in target_rows if r["label"]):
            raise ValueError("Mixed disease masks need verified class semantics")
    return {"condition": condition, "source_sequences": sorted(srcseq), "target_sequences": sorted(dstseq)}


FORMAL_CONDITIONS = ("same_sequence_seen", "same_sequence_unseen", "external_same_sequence", "cross_sequence")


def resolve_condition(source_rows, target_rows, requested="auto"):
    if requested != "auto":
        check_condition(source_rows, target_rows, requested)
        return requested
    development = [r for r in source_rows if r["split"] != "test"]
    src = {r["sequence"] for r in development}
    dst = {r["sequence"] for r in target_rows}
    if not target_rows:
        raise ValueError("Cannot resolve an evaluation condition without target records")
    if src.isdisjoint(dst):
        condition = "cross_sequence"
    elif src == dst and {r["dataset"] for r in development}.isdisjoint({r["dataset"] for r in target_rows}):
        condition = "external_same_sequence"
    else:
        positives = [r for r in development + target_rows if r["label"]]
        if src != dst or any(not r.get("disease_verified") or "unknown" in r["disease_labels"] for r in positives):
            raise ValueError("Cannot infer formal source condition from unverified/mixed metadata; explicitly request exploratory if intended")
        train = {d for r in source_rows if r["split"] == "train" and r["label"] for d in r["disease_labels"]}
        dev = {d for r in development if r["label"] for d in r["disease_labels"]}
        test = {d for r in target_rows if r["label"] for d in r["disease_labels"]}
        if test <= train:
            condition = "same_sequence_seen"
        elif test.isdisjoint(dev):
            condition = "same_sequence_unseen"
        else:
            raise ValueError("Mixed seen/unseen disease cohort requires explicit separate evaluation manifests")
    check_condition(source_rows, target_rows, condition)
    return condition


def evaluation_coverage(sets):
    return {condition: {"availability": "available" if any(c == condition for _, _, c in sets) else "unavailable",
                        "evaluations": [name for name, _, c in sets if c == condition],
                        "reason": None if any(c == condition for _, _, c in sets) else "No verified evaluation manifest supplied for this condition"}
            for condition in FORMAL_CONDITIONS}
