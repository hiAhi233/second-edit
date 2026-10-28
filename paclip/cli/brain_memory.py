"""Frozen E1 + regional multilayer normal memory, source-only calibration."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import nibabel as nib
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from paclip.data.datasets import SliceAnomalyDataset, scan_split, _gray2clip, _normalize_slice, load_binary_mask
from paclip.data.fewshot import file_sha256, validate_splits, restore_support_indices
from paclip.data.supervision import collate_anomaly_samples
from paclip.engine.trainer import atomic_write_json
from paclip.evaluation.metrics import mask_overlap_metrics, image_metrics, choose_image_threshold
from paclip.evaluation.ranking import binary_ranking_metrics
from paclip.evaluation.detection import detection_counts, best_dice_threshold
from paclip.evaluation.postprocess import PostprocessConfig, sigmoid_score, _bilinear_resize, refine_score_map, topk_pool
from paclip.evaluation.visualization import render_four_panel, save_contact_sheet
from paclip.models.paclip import load_trained_model
from paclip.models.normal_memory import NormalPatchMemory, MemoryConfig, image_roi
from paclip.prompts import BRAIN_MRI_PROMPT_SETS


class ExternalSlices(Dataset):
    """All canonical axial slices; target GT never chooses slices or parameters."""
    def __init__(self, protocol):
        self.protocol = protocol
        self.is_wmh = "WMH" in protocol.get("source_description", "") or "white matter" in protocol.get("target", "").lower()
        self.pairs = []
        for i, record in enumerate(protocol["cases"]):
            image = nib.as_closest_canonical(nib.load(record["image"]))
            self.pairs.extend((i, z) for z in range(image.shape[2]))
        self.loaded = None
        self.volume = None

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        i, z = self.pairs[index]
        row = self.protocol["cases"][i]
        if self.loaded != i:
            images = [nib.as_closest_canonical(nib.load(row[key])) for key in ("image", "brain_mask", "lesion_mask")]
            if any(obj.shape != images[0].shape or not np.allclose(obj.affine, images[0].affine, atol=1e-4) for obj in images[1:]):
                raise ValueError("External image, brain mask and lesion mask must be aligned")
            self.volume = (np.asarray(images[0].dataobj, dtype=np.float32),
                           np.asarray(images[1].dataobj) > 0, np.asarray(images[2].dataobj))
            self.loaded = i
        volume, brain, lesion = self.volume
        plane = _normalize_slice(volume[:, :, z] * brain[:, :, z])
        # Fixed direct RAS array, as in the previous source-orientation probe.
        gray = np.array(Image.fromarray(plane).resize((224, 224), Image.Resampling.BILINEAR), dtype=np.float32)
        target = lesion[:, :, z] == 1 if self.is_wmh else lesion[:, :, z] > 0
        valid = brain[:, :, z] & (lesion[:, :, z] != 2) if self.is_wmh else brain[:, :, z]
        gt = load_binary_mask(None, 224, 224, plane=target)
        valid224 = load_binary_mask(None, 224, 224, plane=valid)
        if self.is_wmh:
            valid224 &= ~load_binary_mask(None, 224, 224, plane=lesion[:, :, z] == 2)
        return {"image": _gray2clip(gray), "gray01": torch.from_numpy(gray),
                "gt_mask_high": torch.from_numpy(gt), "valid_high": torch.from_numpy(valid224),
                "label": torch.tensor(int(target.any())), "case_id": row["case_id"],
                "sample_id": f"{row['case_id']}/z{z:04d}", "slice_z": z}


def loader(dataset, batch_size):
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                      collate_fn=collate_anomaly_samples if isinstance(dataset, SliceAnomalyDataset) else None)


def _sequence_key(value):
    text = str(value or "unknown").strip()
    for separator in (",", "(", ";", "/"):
        text = text.split(separator)[0]
    return text.strip() or "unknown"


def sample_group(dataset, index, source_sequence=None):
    """Calibration group for one slice; never uses target labels."""
    if isinstance(dataset, ExternalSlices):
        case_index, _ = dataset.pairs[index]
        record = dataset.protocol["cases"][case_index]
        return _sequence_key(record.get("sequence") or dataset.protocol.get("sequence"))
    records = getattr(dataset, "records", None)
    if not records:
        return _sequence_key(source_sequence)
    modality = records[index].get("modality")
    if modality in (None, ""):
        return _sequence_key(source_sequence)
    return _sequence_key(f"modality:{modality}")


def froc_thresholds(normal_values, pixel_threshold, points=20):
    if points < 4:
        raise ValueError("FROC requires at least four normal-value quantiles")
    if not len(normal_values):
        raise ValueError("FROC requires at least one normal validation value")
    low_count = max(2, int(points) // 2)
    high_count = max(2, int(points) - low_count)
    quantiles = np.r_[np.linspace(.5, .95, low_count), np.linspace(.96, .999, high_count)]
    return np.unique(np.r_[np.quantile(normal_values, quantiles), pixel_threshold]).tolist()


@torch.no_grad()
def infer_cache(model, anchors, dataset, memory, output, device, batch_size, source_sequence=None):
    path = output / "patch_cache.npz"
    if path.exists():
        with np.load(path) as data:
            return {key: data[key] for key in data.files}
    rows = {key: [] for key in ("text", "distance", "roi", "labels", "global", "ids", "groups")}
    done = 0
    for batch in loader(dataset, batch_size):
        outputs = model(batch["image"].to(device), anchors)
        text = outputs["anomaly_map"]
        distance = memory.distances(outputs["f_patch_levels"], text.shape[-2:])
        roi = image_roi(batch["gray01"].to(device), text.shape[-2:], memory.config.roi_threshold,
                        memory.config.min_roi_patches, memory.config.roi_empty)
        rows["text"].append(text.cpu().numpy())
        rows["distance"].append(distance.cpu().numpy())
        rows["roi"].append(roi.cpu().numpy())
        rows["global"].append(outputs["cls_probs"].cpu().numpy())
        rows["labels"].append(batch["label"].numpy())
        rows["ids"].extend(batch["sample_id"])
        rows["groups"].append(np.asarray([sample_group(dataset, done + i, source_sequence)
                                           for i in range(len(text))]))
        done += len(text)
        if done % 256 < batch_size or done == len(dataset):
            print(f"[features] {output.name} {done}/{len(dataset)}", flush=True)
    cache = {key: np.asarray(value) if key == "ids" else np.concatenate(value) for key, value in rows.items()}
    if "groups" not in cache:
        # Old caches predate sequence-aware calibration.
        cache["groups"] = np.asarray(["unknown"] * len(cache["ids"]))
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **cache)
    return cache


def low_scores(memory, cache, alpha, temperature):
    if alpha is None or alpha == 0:
        memory.last_group_fallback = 0
        return np.stack([sigmoid_score(raw, temperature) for raw in cache["text"]])
    margin = memory.scores(torch.from_numpy(cache["distance"]), torch.from_numpy(cache["text"]), alpha,
                           temperature, torch.from_numpy(cache["roi"]),
                           groups=cache.get("groups"))["fused_margin"]
    if memory.config.fusion_mode == "doc":
        # The document formula yields a margin, not a probability.  Applying the
        # text-margin sigmoid here would saturate it and make the comparison unfair.
        return margin.numpy()
    return np.stack([sigmoid_score(raw, temperature) for raw in margin.numpy()])


def sample_views(dataset, index):
    item = dataset[index]
    gray = np.asarray(item["gray01"])
    gt = np.asarray(item["gt_mask_high"]) > 0
    valid = np.asarray(item.get("valid_high", np.ones_like(gt)), dtype=bool)
    return item, gray, gt & valid, valid


def score_views(low, gray, clip=True):
    high = _bilinear_resize(low, *gray.shape).astype(np.float32)
    refined = refine_score_map(high, gray, PostprocessConfig(), clip=clip).astype(np.float32)
    return high, refined, topk_pool(high, .05)[0]


def validate(memory, cache, dataset, alpha, temperature, froc_points=20):
    lows = low_scores(memory, cache, alpha, temperature)
    doc_space = alpha not in (None, 0) and memory.config.fusion_mode == "doc"
    highs, refined, truth, scores, normal_values = [], [], [], [], []
    for i in range(len(dataset)):
        _, gray, gt, valid = sample_views(dataset, i)
        high, final, image_score = score_views(lows[i], gray, clip=not doc_space)
        highs.append(high); refined.append(final); truth.append(gt); scores.append(image_score)
        if cache["labels"][i] == 0:
            normal_values.append(final[valid])
    highs, refined, truth = np.stack(highs), np.stack(refined), np.stack(truth)
    pixel_threshold = best_dice_threshold(refined, truth)
    image_threshold = choose_image_threshold(np.asarray(scores), cache["labels"])
    metrics = {**binary_ranking_metrics(highs, truth),
               **mask_overlap_metrics(refined >= pixel_threshold, truth, [True] * len(truth))}
    normal = np.concatenate(normal_values)
    thresholds = froc_thresholds(normal, pixel_threshold, froc_points)
    score_space = "doc_margin" if doc_space else "post_v1_probability_score"
    return {"alpha_prototype": alpha, "pixel_threshold": pixel_threshold, "image_threshold": image_threshold,
            "froc_thresholds": thresholds, "froc_points": len(thresholds), "threshold_source": "source_val",
            "score_space": score_space, "metrics": metrics}


def evaluate(memory, cache, dataset, operating, temperature, output, representative_indices=()):
    output.mkdir(parents=True, exist_ok=True)
    lows = low_scores(memory, cache, operating["alpha_prototype"], temperature)
    shape = (len(dataset), 224, 224)
    high_maps = np.lib.format.open_memmap(output / "scores.npy", mode="w+", dtype=np.float32, shape=shape)
    truth_maps = np.lib.format.open_memmap(output / "gt.npy", mode="w+", dtype=bool, shape=shape)
    predictions = np.lib.format.open_memmap(output / "predictions.npy", mode="w+", dtype=bool, shape=shape)
    valid_maps = np.lib.format.open_memmap(output / "valid.npy", mode="w+", dtype=bool, shape=shape)
    available, image_scores, panels, per_slice = [], [], [], []
    thresholds = operating["froc_thresholds"]
    froc = [dict(lesions=0, detected=0, false_positive_components=0, predicted_components=0) for _ in thresholds]
    normal_component_counts, normal_area, normal_pixels = [], [], []
    for index in range(len(dataset)):
        item, gray, gt, valid = sample_views(dataset, index)
        high, refined, image_score = score_views(lows[index], gray,
                                                 clip=operating.get("score_space") != "doc_margin")
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
                normal_component_counts.append(primary["false_positive_components"])
                normal_pixels.append(int(prediction.sum()))
                normal_area.append(float(prediction.sum() / valid.sum()))
        denominator = int(prediction.sum()) + int(gt.sum())
        dice = float(2 * (prediction & gt).sum() / denominator) if denominator else 1.
        per_slice.append({"sample_id": item["sample_id"], "case_id": item.get("case_id", ""),
                          "dice": dice, "gt_pixels": int(gt.sum()), "pred_pixels": int(prediction.sum()),
                          "evaluated_pixels": int(valid.sum()), "score_local": image_score})
        if index in representative_indices:
            panel = render_four_panel(gray, high, prediction, gt, True, panel_size=224,
                                      sample_title=f"{item['sample_id']} Dice={dice:.3f}")
            panels.append(panel)
        if (index + 1) % 500 == 0:
            print(f"[postprocess] {output.parent.name}/{output.name} {index + 1}/{len(dataset)}", flush=True)
    for array in (high_maps, truth_maps, predictions, valid_maps):
        array.flush()
    metrics = mask_overlap_metrics(predictions, truth_maps, available)
    print(f"[ranking] {output.parent.name}/{output.name} exact AP/AUROC", flush=True)
    if isinstance(dataset, ExternalSlices):
        # External protocols exclude non-brain tissue / WMH label 2.
        rank = binary_ranking_metrics(high_maps[valid_maps], truth_maps[valid_maps])
    else:
        rank = binary_ranking_metrics(high_maps, truth_maps)
    metrics.update(rank)
    metrics["normal_false_positive_area_ratio"] = float(np.mean(normal_area)) if normal_area else None
    metrics["normal_false_positive_area_pixels"] = float(np.mean(normal_pixels)) if normal_pixels else None
    metrics["normal_slice_false_positive_components"] = float(np.mean(normal_component_counts)) if normal_component_counts else None
    metrics["image"] = image_metrics(np.asarray(image_scores), cache["labels"], threshold=operating["image_threshold"])
    n_available = max(1, sum(available))
    metrics["froc"] = [{"threshold": threshold, "lesion_recall": row["detected"] / max(row["lesions"], 1),
                        "false_positives_per_slice": row["false_positive_components"] / n_available,
                        **row} for threshold, row in zip(thresholds, froc)]
    metrics["n_slices"] = len(dataset)
    metrics["n_cases"] = len({row["case_id"] for row in per_slice if row["case_id"]})
    metrics["froc_points"] = len(thresholds)
    metrics["score_space"] = operating.get("score_space")
    metrics["calibration_group_count"] = len(memory.calibration.get("groups", {}))
    metrics["sequence_calibration_status"] = memory.calibration.get("status")
    metrics["calibration_group_fallback_slices"] = int(memory.last_group_fallback)
    metrics["healthy_case_evaluation"] = "unavailable_no_independent_healthy_cases"
    metrics["prototype_coverage"] = memory.metadata.get("coverage", {})
    atomic_write_json(str(output / "metrics.json"), metrics)
    atomic_write_json(str(output / "operating_point.json"), operating)
    with (output / "per_slice_metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_slice[0])); writer.writeheader(); writer.writerows(per_slice)
    if panels:
        save_contact_sheet(panels, str(output / "overview.png"), f"{output.parent.name} | {output.name}",
                           "Green: GT; red: prediction; fixed source-validation operating point", rows_per_page=4)
    print(f"[metrics] {output.parent.name}/{output.name} AP={rank['pixel_ap']:.4f} Dice={metrics['dice_micro']:.4f}", flush=True)
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--support-manifest", required=True)
    parser.add_argument("--data-root", required=True, help="Parent of train/val/test")
    parser.add_argument("--mask-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--e1-run", default=None)
    parser.add_argument("--prototypes", type=int, default=8)
    parser.add_argument("--alphas", default="0.5,0.7,0.9,1.0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--unseen-protocols", nargs="*", default=[])
    parser.add_argument("--fusion-mode", choices=("doc", "text_scale"), default="doc")
    parser.add_argument("--empty-region", choices=("text", "global"), default="text")
    parser.add_argument("--roi-empty", choices=("invalid", "all"), default="invalid")
    parser.add_argument("--roi-fallback", action="store_true",
                        help="text_scale only: keep text outside the image-foreground ROI")
    parser.add_argument("--min-roi-patches", type=int, default=5)
    parser.add_argument("--calibration-by", choices=("none", "sequence"), default="none")
    parser.add_argument("--source-sequence", default="unknown",
                        help="Sequence label for source PNG slices that carry no modality metadata")
    parser.add_argument("--froc-points", type=int, default=20)
    parser.add_argument("--require-checkpoint-match", action="store_true")
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    if args.froc_points < 4:
        raise ValueError("--froc-points must be at least 4")
    if args.min_roi_patches < 1:
        raise ValueError("--min-roi-patches must be positive")
    if not str(args.source_sequence).strip():
        raise ValueError("--source-sequence must be nonempty")
    output = Path(args.out).resolve(); output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    alphas = [float(part) for part in args.alphas.split(",")]
    if not alphas or any(not 0 < value <= 1 for value in alphas):
        raise ValueError("E2 candidates must have prototype alpha in (0, 1]")
    splits = {name: scan_split(str(Path(args.data_root) / name), str(Path(args.mask_root) / name)) for name in ("train", "val", "test")}
    validate_splits(splits["train"], splits["val"], splits["test"])
    support = json.loads(Path(args.support_manifest).read_text(encoding="utf-8"))
    checkpoint_sha = file_sha256(args.checkpoint)
    support_digest = support.get("support_digest")
    indices = restore_support_indices(splits["train"], support, str(Path(args.data_root) / "train"))
    normal = [splits["train"][index] for index in indices if splits["train"][index]["label"] == 0]
    datasets = {name: SliceAnomalyDataset(str(Path(args.data_root) / name), records=splits[name]) for name in ("val", "test")}
    model = load_trained_model(args.checkpoint, args.device).eval()
    model.requires_grad_(False)
    if args.require_checkpoint_match:
        checkpoint_support = model.checkpoint_meta.get("support_digest")
        if not checkpoint_support:
            raise ValueError("Checkpoint has no support digest; cannot verify the prototype bank support")
        if support_digest is not None and checkpoint_support != support_digest:
            raise ValueError("Checkpoint training support and prototype support manifest do not match")
        checkpoint_samples = (model.checkpoint_meta.get("support_set") or {}).get("samples") or []
        checkpoint_ids = {sample.get("sample_id") for sample in checkpoint_samples if sample.get("sample_id")}
        manifest_ids = {sample.get("sample_id") for sample in support.get("samples", []) if sample.get("sample_id")}
        if not checkpoint_ids:
            raise ValueError("Checkpoint has no support sample IDs; cannot verify the prototype bank support")
        if checkpoint_ids != manifest_ids:
            raise ValueError("Checkpoint support IDs and prototype support manifest do not match")
    prompts = BRAIN_MRI_PROMPT_SETS[model.checkpoint_meta.get("prompt_set", "brain_mri_sentence")]
    config = MemoryConfig(
        prototypes_per_region=args.prototypes,
        fusion_mode=args.fusion_mode,
        empty_region=args.empty_region,
        roi_empty=args.roi_empty,
        min_roi_patches=args.min_roi_patches,
        roi_fallback=args.roi_fallback,
        calibration_by=args.calibration_by,
    )
    with torch.no_grad():
        anchors = model.encode_anchors(prompts)
        normal_ds = SliceAnomalyDataset(str(Path(args.data_root) / "train"), records=normal)
        batch = next(iter(loader(normal_ds, len(normal))))
        outputs = model(batch["image"].to(args.device), anchors)
        hw = outputs["anomaly_map"].shape[-2:]
        roi = image_roi(batch["gray01"].to(args.device), hw, config.roi_threshold,
                        config.min_roi_patches, config.roi_empty)
        memory = NormalPatchMemory(config, model.cfg.ms_layers)
        memory.build(outputs["f_patch_levels"], roi, hw, batch["label"], batch["sample_id"],
                     checkpoint_sha256=checkpoint_sha, support_digest=support_digest)
        # Legacy fused features and logits keep the E1 path unchanged.
        legacy = sum(torch.softmax(model.fusion_weights, 0)[i] *
                     (outputs["f_patch"] @ anchors[level]["abnormal"] - outputs["f_patch"] @ anchors[level]["normal"]).reshape(-1, *hw)
                     for i, level in enumerate(anchors))
        torch.testing.assert_close(legacy, outputs["anomaly_map"], atol=1e-6, rtol=1e-6)
    print(f"[memory] normal support={batch['sample_id']} features={tuple(outputs['f_patch_levels'].shape)}", flush=True)
    identity = {"version": "E2_memory_v3", "checkpoint_sha256": checkpoint_sha,
                "support_digest": support_digest,
                "normal_support_ids": batch["sample_id"], "prototypes_per_region": args.prototypes,
                "layer_ids": memory.layer_ids, "layer_weights": memory.layer_weights.tolist(),
                "alpha_candidates": alphas, "data_root": str(Path(args.data_root).resolve()),
                "unseen_protocols": args.unseen_protocols, "regions": "image 3x3; anatomical orientation unverified",
                "fusion_mode": args.fusion_mode, "empty_region": args.empty_region,
                "roi_empty": args.roi_empty, "roi_fallback": bool(args.roi_fallback),
                "min_roi_patches": args.min_roi_patches, "calibration_by": args.calibration_by,
                "source_sequence": args.source_sequence,
                "froc_points": args.froc_points, "require_checkpoint_match": bool(args.require_checkpoint_match),
                "source_case_overlap": True, "target_tuning": False}
    protocol_path = output / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text(encoding="utf-8")) != identity:
        raise ValueError("Output directory belongs to a different protocol; choose a new --out")
    atomic_write_json(str(protocol_path), identity)
    val_output = output / "source_val"; val_output.mkdir(exist_ok=True)
    val_cache = infer_cache(model, anchors, datasets["val"], memory, val_output, args.device,
                            args.batch_size, args.source_sequence)
    selected = val_cache["labels"] == 0
    calibration_groups = val_cache["groups"][selected] if args.calibration_by == "sequence" else None
    memory.calibrate(torch.from_numpy(val_cache["distance"][selected]), torch.from_numpy(val_cache["text"][selected]),
                     torch.from_numpy(val_cache["roi"][selected]), val_cache["labels"][selected],
                     val_cache["ids"][selected].tolist(), groups=calibration_groups)
    memory.save(output / "normal_memory.pt")
    atomic_write_json(str(output / "normal_memory_metadata.json"), {**memory.metadata, "calibration": memory.calibration})
    validation = [validate(memory, val_cache, datasets["val"], alpha, model.cfg.temperature,
                           froc_points=args.froc_points) for alpha in [None, 0.] + alphas]
    candidates = [row for row in validation if row["alpha_prototype"] in alphas]
    best = max(candidates, key=lambda row: (row["metrics"]["pixel_ap"], row["metrics"]["dice_micro"], -row["metrics"]["normal_false_positive_area_ratio"]))
    # Alpha=0 is an exact E1 identity check, rather than a changed text branch.
    assert validation[0]["metrics"] == validation[1]["metrics"]
    configs = {"E1": validation[0], "E2": best}
    # Use the previously saved E1 operating point for the exact original comparator.
    if args.e1_run:
        saved = json.loads((Path(args.e1_run) / "thresholds.json").read_text(encoding="utf-8"))["post_v1_probability_score"]
        configs["E1"]["pixel_threshold"] = saved["pixel_threshold"]
        configs["E1"]["image_threshold"] = saved["image_threshold"]
    atomic_write_json(str(output / "validation_search.json"), {"selection": "highest source-validation Pixel AP, then Dice, then lower normal FP area", "candidates": validation, "selected": best})
    print(f"[selection] alpha={best['alpha_prototype']} val AP={best['metrics']['pixel_ap']:.4f}", flush=True)
    tasks = {"source_test": datasets["test"]}
    for path in args.unseen_protocols:
        protocol = json.loads(Path(path).read_text(encoding="utf-8"))
        name = "unseen_wmh" if "WMH" in protocol.get("source_description", "") else "unseen_atlas"
        tasks[name] = ExternalSlices(protocol)
        atomic_write_json(str(output / f"{name}_provenance.json"), protocol)
    caches = {}
    for name, dataset in tasks.items():
        destination = output / name; destination.mkdir(exist_ok=True)
        caches[name] = infer_cache(model, anchors, dataset, memory, destination, args.device,
                                   args.batch_size, args.source_sequence)
    temperature = model.cfg.temperature
    del model, anchors, outputs, batch
    gc.collect()
    torch.cuda.empty_cache()
    results = {}
    for name, dataset in tasks.items():
        # Fixed slice-area quantiles for source display only; all slices are evaluated.
        if isinstance(dataset, ExternalSlices):
            representative = []
            for case in sorted({index for index, _ in dataset.pairs}):
                positions = [i for i, (index, _) in enumerate(dataset.pairs) if index == case]
                # GT is used solely for a display slice after parameter selection.
                representative.append(max(positions, key=lambda i: int(np.asarray(dataset[i]["gt_mask_high"]).sum())))
        else:
            representative = np.linspace(0, len(dataset) - 1, 12, dtype=int).tolist()
        results[name] = {}
        for tag, operating in configs.items():
            results[name][tag] = evaluate(memory, caches[name], dataset, operating, temperature, output / name / tag, representative)
            atomic_write_json(str(output / "results.json"), results)
    atomic_write_json(str(output / "completion.json"), {"status": "complete", "elapsed_seconds": time.time() - started,
                                                        "checkpoint_sha256": checkpoint_sha,
                                                        "support_digest": support_digest,
                                                        "fusion_mode": args.fusion_mode,
                                                        "empty_region": args.empty_region,
                                                        "roi_empty": args.roi_empty,
                                                        "roi_fallback": bool(args.roi_fallback),
                                                        "calibration_by": args.calibration_by,
                                                        "source_sequence": args.source_sequence,
                                                        "froc_points": args.froc_points,
                                                        "require_checkpoint_match": bool(args.require_checkpoint_match),
                                                        "prototype_coverage": memory.metadata.get("coverage", {}),
                                                        "selected_alpha": best["alpha_prototype"]})
    print(f"[complete] {output}", flush=True)
    return results


if __name__ == "__main__":
    main()
