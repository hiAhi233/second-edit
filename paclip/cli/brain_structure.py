"""第三步结构关系与 2.5D 一致性实验入口。

默认关闭所有结构模块（B1/B2）。上下文分支只在源正常验证上校准，β 由源验证
按“先保住小病灶召回、限制正常图 FP，再比较 AP/Dice”的规则选择；2.5D 只在
可信邻片索引存在时生效，源 PNG 没有真实 slice_z 时严格回退中心图。
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch

from paclip.cli.brain_memory import (ExternalSlices, froc_thresholds, loader, sample_group,
                                     sample_views, score_views)
from paclip.data.datasets import SliceAnomalyDataset, scan_split
from paclip.data.fewshot import file_sha256, restore_support_indices, validate_splits
from paclip.data.slice_context import build_slice_index
from paclip.engine.consistency import score_delta, smooth_three_slice
from paclip.engine.trainer import atomic_write_json
from paclip.evaluation.detection import best_dice_threshold, detection_counts
from paclip.evaluation.metrics import choose_image_threshold, image_metrics, mask_overlap_metrics
from paclip.evaluation.postprocess import PostprocessConfig, refine_score_map, sigmoid_score
from paclip.evaluation.ranking import binary_ranking_metrics
from paclip.evaluation.visualization import render_four_panel, save_contact_sheet
from paclip.models.normal_memory import MemoryConfig, NormalPatchMemory, image_roi
from paclip.models.paclip import load_trained_model
from paclip.models.structural_context import ContextConfig, StructuralContext
from paclip.prompts import BRAIN_MRI_PROMPT_SETS


def _file_stat(path):
    if not path:
        return None
    item = Path(path)
    if not item.exists():
        return None
    stat = item.stat()
    return [int(stat.st_size), int(stat.st_mtime_ns)]


def dataset_fingerprint(dataset) -> str:
    if hasattr(dataset, "protocol"):
        protocol = dataset.protocol
        payload = {
            "sequence": protocol.get("sequence"),
            "source_description": protocol.get("source_description"),
            "cases": [[case.get("case_id"), case.get("series_id"), case.get("series"),
                       case.get("sequence"), case.get("image"), case.get("brain_mask"),
                       case.get("lesion_mask"), _file_stat(case.get("image")),
                       _file_stat(case.get("brain_mask")), _file_stat(case.get("lesion_mask"))]
                      for case in protocol.get("cases", [])],
        }
    else:
        payload = [[record.get("sample_id"), record.get("case_id"), record.get("series_id"),
                    record.get("sequence"), record.get("modality"), record.get("slice_z"),
                    record.get("path"), record.get("mask_path"),
                    _file_stat(record.get("path")), _file_stat(record.get("mask_path"))]
                   for record in getattr(dataset, "records", [])]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def memory_fingerprint(memory) -> Optional[str]:
    if memory is None:
        return None
    digest = hashlib.sha256()
    digest.update(json.dumps(asdict(memory.config), sort_keys=True).encode("utf-8"))
    digest.update(json.dumps({"layer_ids": memory.layer_ids,
                              "layer_weights": memory.layer_weights.tolist(),
                              "supported": memory.supported}, sort_keys=True).encode("utf-8"))
    for bank in memory.prototypes:
        for prototypes in bank:
            digest.update(np.ascontiguousarray(prototypes.detach().cpu().numpy()).tobytes())
    return digest.hexdigest()


def output_identity(args, config_name, checkpoint_sha, support_digest, betas, context_config):
    return {
        "version": "E3_structure_v1",
        "config": config_name,
        "base": args.base,
        "context": args.context,
        "consistency": args.consistency,
        "checkpoint_path": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "support_manifest": str(Path(args.support_manifest).resolve()),
        "support_digest": support_digest,
        "data_root": str(Path(args.data_root).resolve()),
        "mask_root": str(Path(args.mask_root).resolve()),
        "source_sequence": args.source_sequence,
        "alpha": args.alpha,
        "betas": betas,
        "froc_points": args.froc_points,
        "context_config": asdict(context_config),
        "e2_memory": str(Path(args.e2_memory).resolve()) if args.e2_memory else None,
        "unseen_protocols": [str(Path(path).resolve()) for path in args.unseen_protocols],
        "require_checkpoint_match": bool(args.require_checkpoint_match),
        "cache_root": str(Path(args.cache_root).resolve()) if args.cache_root else None,
        "code_fingerprint": code_fingerprint(),
    }


def code_fingerprint() -> str:
    root = Path(__file__).resolve().parents[2]
    files = [
        "paclip/cli/brain_structure.py",
        "paclip/data/slice_context.py",
        "paclip/models/structural_context.py",
        "paclip/engine/consistency.py",
        "paclip/models/normal_memory.py",
        "paclip/cli/brain_memory.py",
        "paclip/evaluation/postprocess.py",
    ]
    digest = hashlib.sha256()
    for relative in files:
        digest.update((root / relative).read_bytes())
    return digest.hexdigest()


def slice_info(dataset, index: int):
    if hasattr(dataset, "pairs"):
        case_index, z = dataset.pairs[index]
        record = dataset.protocol["cases"][case_index]
        return str(record.get("case_id") or case_index), int(z)
    record = dataset.records[index]
    value = record.get("slice_z", -1)
    return str(record.get("case_id") or ""), int(-1 if value is None else value)


def fuse_context_scores(base, context, valid, beta):
    if not 0 <= beta <= 1:
        raise ValueError("Context fusion beta must be in [0, 1]")
    if beta == 0:
        return np.array(base, copy=True)
    result = np.array(base, copy=True)
    mask = np.asarray(valid, dtype=bool)
    result[mask] = (1.0 - beta) * base[mask] + beta * context[mask]
    return result


def load_base_memory(path, checkpoint_sha, support_digest):
    memory = NormalPatchMemory.load(path, checkpoint_sha256=checkpoint_sha,
                                    support_digest=support_digest)
    if memory.config.fusion_mode != "text_scale":
        raise ValueError("E3 E2 baseline only supports the text_scale probability branch")
    if not memory.config.roi_fallback:
        raise ValueError("E3 E2 baseline expects the text_scale ROI-fallback extension")
    if memory.calibration is None:
        raise ValueError("Loaded E2 memory has no calibration")
    return memory


def build_base_memory(model, anchors, normal, checkpoint_sha, support_digest, device):
    config = MemoryConfig(
        prototypes_per_region=8,
        fusion_mode="text_scale",
        empty_region="text",
        roi_empty="invalid",
        min_roi_patches=5,
        roi_fallback=True,
        calibration_by="sequence",
    )
    memory = NormalPatchMemory(config, model.cfg.ms_layers)
    dataset = SliceAnomalyDataset(str(Path(normal[0]["path"]).parents[1]), records=normal)
    batch = next(iter(loader(dataset, len(normal))))
    with torch.no_grad():
        outputs = model(batch["image"].to(device), anchors)
        hw = outputs["anomaly_map"].shape[-2:]
        roi = image_roi(batch["gray01"].to(device), hw, config.roi_threshold,
                        config.min_roi_patches, config.roi_empty)
        memory.build(outputs["f_patch_levels"], roi, hw, batch["label"], batch["sample_id"],
                     checkpoint_sha256=checkpoint_sha, support_digest=support_digest)
    return memory


def cache_dataset(model, anchors, dataset, name, base, memory, context, cache_root,
                  device, batch_size, source_sequence, checkpoint_sha, support_digest,
                  prompt_digest):
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    path = cache_root / f"{base}_{name}.npz"
    identity_path = cache_root / f"{base}_{name}.identity.json"
    identity = {
        "version": "E3_features_v1",
        "dataset": name,
        "base": base,
        "checkpoint_sha256": checkpoint_sha,
        "support_digest": support_digest,
        "prompt_digest": prompt_digest,
        "context": asdict(context.config),
        "source_sequence": str(source_sequence),
        "memory_fingerprint": memory_fingerprint(memory),
        "roi": {"threshold": 20.0 / 255.0, "min_patches": 5, "empty": "invalid"},
        "dataset_fingerprint": dataset_fingerprint(dataset),
        "code_fingerprint": code_fingerprint(),
        "n_slices": len(dataset),
    }
    if path.exists():
        if not identity_path.exists():
            raise ValueError(f"Cache {path} has no identity; choose a new cache root")
        saved = json.loads(identity_path.read_text(encoding="utf-8"))
        if saved == identity:
            with np.load(path) as data:
                print(f"[cache] reuse {path}", flush=True)
                return {key: data[key] for key in data.files}
        raise ValueError(f"Cache identity mismatch for {path}; choose a new cache root")
    rows = {key: [] for key in ("text", "context_error", "context_valid", "roi", "labels",
                                "ids", "groups", "case_ids", "slice_z")}
    if base == "e2":
        rows["distance"] = []
    done = 0
    for batch in loader(dataset, batch_size):
        with torch.no_grad():
            outputs = model(batch["image"].to(device), anchors)
            text = outputs["anomaly_map"]
            levels = outputs["f_patch_levels"]
            hw = text.shape[-2:]
            roi = image_roi(batch["gray01"].to(device), hw, 20.0 / 255.0, 5, "invalid")
            errors, valid = context.compute_errors(levels, roi, hw)
            distance = memory.distances(levels, hw) if memory is not None else None
        count = len(text)
        rows["text"].append(text.cpu().numpy())
        rows["context_error"].append(errors.cpu().numpy())
        rows["context_valid"].append(valid.cpu().numpy())
        rows["roi"].append(roi.cpu().numpy())
        rows["labels"].append(batch["label"].numpy())
        rows["ids"].extend(batch["sample_id"])
        rows["groups"].append(np.asarray([sample_group(dataset, done + i, source_sequence)
                                           for i in range(count)]))
        for i in range(count):
            case_id, z = slice_info(dataset, done + i)
            rows["case_ids"].append(case_id)
            rows["slice_z"].append(z)
        if distance is not None:
            rows["distance"].append(distance.cpu().numpy())
        done += count
        if done % 256 < batch_size or done == len(dataset):
            print(f"[features] {name} {done}/{len(dataset)}", flush=True)
    cache = {}
    for key, value in rows.items():
        if key in ("ids", "case_ids", "slice_z"):
            cache[key] = np.asarray(value)
        else:
            cache[key] = np.concatenate(value)
    np.savez_compressed(path, **cache)
    atomic_write_json(str(identity_path), identity)
    return cache


def base_low_scores(cache, base, memory, alpha, temperature):
    if base == "e1":
        return np.stack([sigmoid_score(raw, temperature) for raw in cache["text"]])
    if memory is None:
        raise ValueError("E2 baseline requires a calibrated normal memory")
    result = memory.scores(
        torch.from_numpy(cache["distance"]),
        torch.from_numpy(cache["text"]),
        alpha,
        temperature,
        torch.from_numpy(cache["roi"]),
        groups=cache["groups"],
    )["fused"].numpy()
    return result


def context_low_scores(cache, context):
    scores, valid = context.scores(
        torch.from_numpy(cache["context_error"]),
        torch.from_numpy(cache["context_valid"]),
        groups=cache["groups"],
    )
    return scores.numpy(), valid.numpy()


def validate_scores(low, cache, dataset, froc_points=20):
    highs, refined, truth, valids, image_scores, normal_values = [], [], [], [], [], []
    for index in range(len(dataset)):
        _, gray, gt, valid = sample_views(dataset, index)
        high, final, image_score = score_views(low[index], gray, clip=True)
        highs.append(high); refined.append(final); truth.append(gt); valids.append(valid)
        image_scores.append(image_score)
        if cache["labels"][index] == 0:
            normal_values.append(final[valid])
    highs, refined, truth = np.stack(highs), np.stack(refined), np.stack(truth)
    pixel_threshold = best_dice_threshold(refined, truth)
    image_threshold = choose_image_threshold(np.asarray(image_scores), cache["labels"])
    metrics = {**binary_ranking_metrics(highs, truth),
               **mask_overlap_metrics(refined >= pixel_threshold, truth, [True] * len(truth))}
    normal_components, normal_area, normal_pixels = [], [], []
    for index in range(len(dataset)):
        if cache["labels"][index] != 0:
            continue
        final, gt, valid = refined[index], truth[index], valids[index]
        primary = detection_counts(final, gt, valid, [pixel_threshold])[0]
        normal_components.append(primary["false_positive_components"])
        prediction = (final >= pixel_threshold) & valid
        normal_pixels.append(int(prediction.sum()))
        normal_area.append(float(prediction.sum() / max(int(valid.sum()), 1)))
    metrics["normal_slice_false_positive_components"] = (
        float(np.mean(normal_components)) if normal_components else None)
    metrics["normal_false_positive_area_ratio"] = float(np.mean(normal_area)) if normal_area else None
    metrics["normal_false_positive_area_pixels"] = float(np.mean(normal_pixels)) if normal_pixels else None
    if not normal_values:
        raise ValueError("FROC selection needs at least one normal source validation slice")
    thresholds = froc_thresholds(np.concatenate(normal_values), pixel_threshold, froc_points)
    return {"pixel_threshold": pixel_threshold, "image_threshold": image_threshold,
            "froc_thresholds": thresholds, "froc_points": len(thresholds),
            "threshold_source": "source_val", "score_space": "probability",
            "metrics": metrics}


def _finite_number(value):
    return value is not None and np.isfinite(float(value))


def select_context_candidate(candidates):
    """Select beta on source validation; missing/non-finite values never count as zero."""
    if not candidates:
        raise ValueError("At least one beta candidate is required")
    baseline = candidates[0]
    baseline_small = baseline["metrics"].get("small_lesion_recall")
    baseline_fp = baseline["metrics"].get("normal_slice_false_positive_components")
    baseline_ap = baseline["metrics"].get("pixel_ap")
    baseline_dice = baseline["metrics"].get("dice_micro")
    if not all(_finite_number(value) for value in (baseline_small, baseline_fp, baseline_ap, baseline_dice)):
        raise ValueError("Baseline validation has missing or non-finite small recall, FP, AP or Dice")
    best = baseline
    best_key = (float(baseline_ap), float(baseline_dice))
    for row in candidates[1:]:
        small = row["metrics"].get("small_lesion_recall")
        fp = row["metrics"].get("normal_slice_false_positive_components")
        ap = row["metrics"].get("pixel_ap")
        dice = row["metrics"].get("dice_micro")
        if not all(_finite_number(value) for value in (small, fp, ap, dice)):
            continue
        small, fp = float(small), float(fp)
        if small < float(baseline_small) or fp > float(baseline_fp):
            continue
        key = (float(ap), float(dice))
        if key > best_key:
            best = row
            best_key = key
    return best


def evaluate_scores(low, cache, dataset, operating, output, representative_indices=()):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    shape = (len(dataset), 224, 224)
    high_maps = np.lib.format.open_memmap(output / "scores.npy", mode="w+", dtype=np.float32, shape=shape)
    truth_maps = np.lib.format.open_memmap(output / "gt.npy", mode="w+", dtype=bool, shape=shape)
    predictions = np.lib.format.open_memmap(output / "predictions.npy", mode="w+", dtype=bool, shape=shape)
    valid_maps = np.lib.format.open_memmap(output / "valid.npy", mode="w+", dtype=bool, shape=shape)
    thresholds = operating["froc_thresholds"]
    froc = [dict(lesions=0, detected=0, false_positive_components=0, predicted_components=0)
            for _ in thresholds]
    available, image_scores, per_slice, panels = [], [], [], []
    normal_components, normal_area, normal_pixels = [], [], []
    for index in range(len(dataset)):
        item, gray, gt, valid = sample_views(dataset, index)
        high, refined, image_score = score_views(low[index], gray, clip=True)
        prediction = (refined >= operating["pixel_threshold"]) & valid
        high_maps[index], truth_maps[index] = high, gt
        predictions[index], valid_maps[index] = prediction, valid
        ok = bool(valid.any())
        available.append(ok); image_scores.append(image_score)
        if ok:
            counts = detection_counts(refined, gt, valid, thresholds)
            for accum, row in zip(froc, counts):
                for key in accum:
                    accum[key] += row[key]
            primary = detection_counts(refined, gt, valid, [operating["pixel_threshold"]])[0]
            if not gt.any():
                normal_components.append(primary["false_positive_components"])
                normal_pixels.append(int(prediction.sum()))
                normal_area.append(float(prediction.sum() / valid.sum()))
        denominator = int(prediction.sum()) + int(gt.sum())
        dice = float(2 * (prediction & gt).sum() / denominator) if denominator else 1.0
        per_slice.append({"sample_id": item["sample_id"], "case_id": item.get("case_id", ""),
                          "dice": dice, "gt_pixels": int(gt.sum()), "pred_pixels": int(prediction.sum()),
                          "evaluated_pixels": int(valid.sum()), "score_local": image_score})
        if index in representative_indices:
            panel = render_four_panel(gray, high, prediction, gt, True, panel_size=224,
                                      sample_title=f"{item['sample_id']} Dice={dice:.3f}")
            panels.append(panel)
    for array in (high_maps, truth_maps, predictions, valid_maps):
        array.flush()
    metrics = mask_overlap_metrics(predictions, truth_maps, available)
    if isinstance(dataset, ExternalSlices):
        rank = binary_ranking_metrics(high_maps[valid_maps], truth_maps[valid_maps])
    else:
        rank = binary_ranking_metrics(high_maps, truth_maps)
    metrics.update(rank)
    metrics["normal_false_positive_area_ratio"] = float(np.mean(normal_area)) if normal_area else None
    metrics["normal_false_positive_area_pixels"] = float(np.mean(normal_pixels)) if normal_pixels else None
    metrics["normal_slice_false_positive_components"] = float(np.mean(normal_components)) if normal_components else None
    metrics["image"] = image_metrics(np.asarray(image_scores), cache["labels"], threshold=operating["image_threshold"])
    n_available = max(1, sum(available))
    metrics["froc"] = [{"threshold": threshold, "lesion_recall": row["detected"] / max(row["lesions"], 1),
                        "false_positives_per_slice": row["false_positive_components"] / n_available,
                        **row} for threshold, row in zip(thresholds, froc)]
    metrics["n_slices"] = len(dataset)
    metrics["n_cases"] = len({row["case_id"] for row in per_slice if row["case_id"]})
    metrics["froc_points"] = len(thresholds)
    metrics["score_space"] = operating.get("score_space", "probability")
    metrics["context_group_fallback_slices"] = 0
    atomic_write_json(str(output / "metrics.json"), metrics)
    atomic_write_json(str(output / "operating_point.json"), operating)
    with (output / "per_slice_metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_slice[0])); writer.writeheader(); writer.writerows(per_slice)
    if panels:
        save_contact_sheet(panels, str(output / "overview.png"), f"{output.parent.name} | {output.name}",
                           "Green: GT; red: prediction; fixed source-validation operating point", rows_per_page=4)
    return metrics


def volume_metrics(folder, name, dataset):
    folder = Path(folder)
    high = np.load(folder / "scores.npy", mmap_mode="r")
    gt = np.load(folder / "gt.npy", mmap_mode="r")
    valid = np.load(folder / "valid.npy", mmap_mode="r")
    predictions = np.load(folder / "predictions.npy", mmap_mode="r")
    operating = json.loads((folder / "operating_point.json").read_text(encoding="utf-8"))
    thresholds = [operating["pixel_threshold"]] + operating["froc_thresholds"]
    totals = [dict(lesions=0, detected=0, false_positive_components=0, predicted_components=0)
              for _ in thresholds]
    rows = []
    for case_index, record in enumerate(dataset.protocol["cases"]):
        indices = [i for i, (case, _) in enumerate(dataset.pairs) if case == case_index]
        refined = np.stack([refine_score_map(high[i], np.asarray(dataset[i]["gray01"]), PostprocessConfig(), clip=True)
                            for i in indices])
        truth, selected, pred = gt[indices], valid[indices], predictions[indices]
        counts = detection_counts(refined, truth, selected, thresholds)
        for accum, row in zip(totals, counts):
            for key in accum:
                accum[key] += row[key]
        denominator = int(pred.sum()) + int(truth.sum())
        dice = float(2 * (pred & truth).sum() / denominator) if denominator else 1.0
        rows.append({"case_id": record["case_id"], "dice_3d": dice, **counts[0]})
    volume = {"grid": "224x224xZ canonical axial, not native physical voxel space",
              "connectivity": 26, "min_gt_coverage": .1, "n_cases": len(rows), "cases": rows,
              "lesion_recall": totals[0]["detected"] / max(totals[0]["lesions"], 1),
              "false_positives_per_case": totals[0]["false_positive_components"] / max(len(rows), 1),
              "froc": [{"threshold": threshold, "lesion_recall": row["detected"] / max(row["lesions"], 1),
                        "false_positives_per_case": row["false_positive_components"] / max(len(rows), 1), **row}
                       for threshold, row in zip(thresholds[1:], totals[1:])]}
    atomic_write_json(str(folder / "volume_detection.json"), volume)
    return volume


def representative_indices(dataset, rows):
    if isinstance(dataset, ExternalSlices):
        result = []
        for case in sorted({case for case, _ in dataset.pairs}):
            positions = [i for i, (c, _) in enumerate(dataset.pairs) if c == case]
            result.append(max(positions, key=lambda i: int(np.asarray(dataset[i]["gt_mask_high"]).sum())))
        return result
    return np.linspace(0, len(dataset) - 1, 12, dtype=int).tolist()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--support-manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--mask-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cache-root", default=None)
    parser.add_argument("--e2-memory", default=None)
    parser.add_argument("--base", choices=("e1", "e2"), default="e1")
    parser.add_argument("--context", choices=("on", "off"), default="off")
    parser.add_argument("--consistency", choices=("on", "off"), default="off")
    parser.add_argument("--betas", default="0,0.05,0.1,0.2")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--exclude-center", action="store_true")
    parser.add_argument("--context-kernel", type=int, default=3)
    parser.add_argument("--source-sequence", default="FLAIR")
    parser.add_argument("--froc-points", type=int, default=20)
    parser.add_argument("--unseen-protocols", nargs="*", default=[])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--require-checkpoint-match", action="store_true")
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    if args.froc_points < 4:
        raise ValueError("--froc-points must be at least 4")
    betas = sorted(set([0.0] + [float(value) for value in args.betas.split(",") if value.strip()]))
    if any(not 0 <= beta <= 1 for beta in betas):
        raise ValueError("Beta candidates must be in [0, 1]")
    output = Path(args.out).resolve(); output.mkdir(parents=True, exist_ok=True)
    cache_root = Path(args.cache_root).resolve() if args.cache_root else output.parent / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    config_name = args.config or ("B" if args.context == "off" and args.consistency == "off" else
                                  ("C" if args.consistency == "off" else "D")) + ("1" if args.base == "e1" else "2")
    if args.consistency == "on" and args.context != "on":
        raise ValueError("2.5D consistency is defined on top of the context branch (D=C+2.5D)")
    splits = {name: scan_split(str(Path(args.data_root) / name), str(Path(args.mask_root) / name))
              for name in ("train", "val", "test")}
    validate_splits(splits["train"], splits["val"], splits["test"])
    support = json.loads(Path(args.support_manifest).read_text(encoding="utf-8"))
    support_digest = support.get("support_digest")
    checkpoint_sha = file_sha256(args.checkpoint)
    context_config = ContextConfig(kernel=args.context_kernel,
                                   exclude_center=bool(args.exclude_center),
                                   calibration_by="sequence")
    output_protocol_path = output / "protocol.json"
    identity = output_identity(args, config_name, checkpoint_sha, support_digest, betas, context_config)
    if output_protocol_path.exists():
        saved = json.loads(output_protocol_path.read_text(encoding="utf-8"))
        core = {key: value for key, value in saved.items()
                if key not in ("status", "selected_beta", "slice_indexes", "elapsed_seconds")}
        if core != identity:
            raise ValueError("Output directory belongs to a different E3 identity; choose a new --out")
    atomic_write_json(str(output_protocol_path), {**identity, "status": "running"})
    atomic_write_json(str(output / "completion.json"),
                      {"status": "running", "config": config_name,
                       "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    indices = restore_support_indices(splits["train"], support, str(Path(args.data_root) / "train"))
    normal = [splits["train"][index] for index in indices if splits["train"][index]["label"] == 0]
    datasets = {name: SliceAnomalyDataset(str(Path(args.data_root) / name), records=splits[name])
                for name in ("val", "test")}
    model = load_trained_model(args.checkpoint, args.device).eval()
    model.requires_grad_(False)
    if args.require_checkpoint_match:
        checkpoint_support = model.checkpoint_meta.get("support_digest")
        if not checkpoint_support or checkpoint_support != support_digest:
            raise ValueError("Checkpoint support digest and support manifest do not match")
        checkpoint_samples = (model.checkpoint_meta.get("support_set") or {}).get("samples") or []
        checkpoint_ids = {sample.get("sample_id") for sample in checkpoint_samples if sample.get("sample_id")}
        manifest_ids = {sample.get("sample_id") for sample in support.get("samples", []) if sample.get("sample_id")}
        if checkpoint_ids != manifest_ids:
            raise ValueError("Checkpoint support IDs and support manifest do not match")
    prompts = BRAIN_MRI_PROMPT_SETS[model.checkpoint_meta.get("prompt_set", "brain_mri_sentence")]
    anchors = model.encode_anchors(prompts)
    temperature = float(model.cfg.temperature)
    context = StructuralContext(context_config, model.cfg.ms_layers)
    memory = None
    if args.base == "e2":
        if args.e2_memory:
            memory = load_base_memory(args.e2_memory, checkpoint_sha, support_digest)
        else:
            memory = build_base_memory(model, anchors, normal, checkpoint_sha, support_digest, args.device)
    val_cache = cache_dataset(model, anchors, datasets["val"], "source_val", args.base, memory, context,
                              cache_root, args.device, args.batch_size, args.source_sequence,
                              checkpoint_sha, support_digest, model.checkpoint_meta.get("prompt_digest"))
    selected = val_cache["labels"] == 0
    if memory is not None and memory.calibration is None:
        memory.calibrate(torch.from_numpy(val_cache["distance"][selected]), torch.from_numpy(val_cache["text"][selected]),
                         torch.from_numpy(val_cache["roi"][selected]), val_cache["labels"][selected],
                         val_cache["ids"][selected].tolist(), groups=val_cache["groups"][selected])
        memory.save(output / "normal_memory.pt")
        atomic_write_json(str(output / "normal_memory_metadata.json"),
                          {**memory.metadata, "calibration": memory.calibration})
    if args.context == "on":
        context.calibrate(torch.from_numpy(val_cache["context_error"][selected]),
                          torch.from_numpy(val_cache["context_valid"][selected]),
                          val_cache["groups"][selected], val_cache["ids"][selected].tolist())
        atomic_write_json(str(output / "context_calibration.json"), context.calibration)
    base_val = base_low_scores(val_cache, args.base, memory, args.alpha, temperature)
    context_val, context_valid_val = context_low_scores(val_cache, context) if args.context == "on" else (np.zeros_like(base_val), np.zeros_like(base_val, dtype=bool))
    val_index = build_slice_index(datasets["val"], "source_val")
    candidates = []
    for beta in (betas if args.context == "on" else [0.0]):
        fused = fuse_context_scores(base_val, context_val, context_valid_val, beta)
        consistency_summary = None
        if args.consistency == "on":
            fused, consistency_summary = smooth_three_slice(fused, val_index)
        row = validate_scores(fused, val_cache, datasets["val"], args.froc_points)
        row["beta"] = beta
        row["consistency"] = consistency_summary
        candidates.append(row)
    best = select_context_candidate(candidates)
    atomic_write_json(str(output / "validation_search.json"),
                      {"selection": "keep small-lesion recall and normal FP <= baseline, then max AP/Dice",
                       "baseline_beta": 0.0, "selected_beta": best["beta"], "candidates": candidates})
    tasks = {"source_test": datasets["test"]}
    for unseen_protocol_path in args.unseen_protocols:
        protocol = json.loads(Path(unseen_protocol_path).read_text(encoding="utf-8"))
        name = "unseen_wmh" if "WMH" in protocol.get("source_description", "") else "unseen_atlas"
        tasks[name] = ExternalSlices(protocol)
        atomic_write_json(str(output / f"{name}_provenance.json"), protocol)
    results, indexes = {}, {}
    for name, dataset in tasks.items():
        cache = cache_dataset(model, anchors, dataset, name, args.base, memory, context, cache_root,
                              args.device, args.batch_size, args.source_sequence, checkpoint_sha,
                              support_digest, model.checkpoint_meta.get("prompt_digest"))
        base_low = base_low_scores(cache, args.base, memory, args.alpha, temperature)
        base_group_fallback = int(memory.last_group_fallback) if memory is not None else 0
        context_low, context_valid = context_low_scores(cache, context) if args.context == "on" else (np.zeros_like(base_low), np.zeros_like(base_low, dtype=bool))
        final_low = fuse_context_scores(base_low, context_low, context_valid, float(best["beta"]))
        index = build_slice_index(dataset, name)
        indexes[name] = index.summary
        consistency_summary = None
        if args.consistency == "on":
            final_low, consistency_summary = smooth_three_slice(final_low, index)
        operating = {key: best[key] for key in ("pixel_threshold", "image_threshold", "froc_thresholds",
                                                 "froc_points", "threshold_source", "score_space")}
        operating.update({"beta": best["beta"], "config": config_name, "base": args.base,
                          "context": args.context, "consistency": args.consistency})
        destination = output / name / config_name
        metrics = evaluate_scores(final_low, cache, dataset, operating, destination,
                                  representative_indices(dataset, None))
        metrics["slice_context"] = index.summary
        metrics["consistency"] = consistency_summary
        metrics["context_group_fallback_slices"] = int(context.last_group_fallback) if args.context == "on" else 0
        metrics["base_group_fallback_slices"] = base_group_fallback
        if isinstance(dataset, ExternalSlices):
            metrics["volume_detection"] = volume_metrics(destination, name, dataset)
        atomic_write_json(str(destination / "metrics.json"), metrics)
        results[name] = metrics
        atomic_write_json(str(output / "results.json"), results)
        print(f"[metrics] {name} AP={metrics['pixel_ap']:.4f} Dice={metrics['dice_micro']:.4f}", flush=True)
    atomic_write_json(str(output_protocol_path), {**identity, "status": "complete",
                                                 "selected_beta": best["beta"], "slice_indexes": indexes})
    atomic_write_json(str(output / "completion.json"),
                      {"status": "complete", "elapsed_seconds": time.time() - started,
                       "config": config_name, "selected_beta": best["beta"],
                       "context_group_count": len(context.calibration["groups"]) if context.calibration else 0,
                       "slice_indexes": indexes})
    print(f"[complete] {output}", flush=True)
    del model, anchors, memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


if __name__ == "__main__":
    main()
