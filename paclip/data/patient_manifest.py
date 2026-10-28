"""Explicit patient identity, content audit and reproducible K-slice support."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from PIL import Image

from paclip.data.fewshot import file_sha256, stable_digest

VERSION = "patient_manifest_v1"
SPLITS = ("train", "validation", "test")


def patient_split_evidence(rows):
    groups = {s: {r.get("patient_id") for r in rows if r.get("split") == s} for s in SPLITS}
    overlap = {f"{a}/{b}": sorted(groups[a] & groups[b], key=str)
               for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))}
    complete = bool(rows) and all(groups.values()) and all(
        r.get("patient_id") and r.get("patient_verified") is True and r.get("split") in SPLITS for r in rows)
    return {"case_disjoint_verified": bool(complete and not any(overlap.values())),
            "meaning": "patients_disjoint_between_train_validation_test_not_unique_within_support",
            "patients_by_split": {s: len(g) for s, g in groups.items()}, "overlap": overlap}


def read_rows(path):
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    else:
        obj = json.loads(path.read_text(encoding="utf-8-sig"))
        rows = obj if isinstance(obj, list) else obj["samples"]
    for row in rows:
        for key in ("image_path", "mask_path", "roi_path"):
            if row.get(key):
                p = Path(row[key])
                row[key] = str((path.parent / p).resolve() if not p.is_absolute() else p.resolve())
    return rows


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def write_rows(path, rows):
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def audit_rows(rows, *, require_splits=True, verify_files=True):
    """Returns errors, never promotes inferred IDs/sequence labels to verified facts.

    patient_id must be a verified GLOBAL identifier (aliases mapped beforehand).
    Empty normal masks are not treated as duplicate images; duplicate nonempty
    masks across splits are reported separately because annotation shapes can recur.
    """
    errors, warnings = [], []
    ids, patients, paths, images, masks = {}, defaultdict(set), {}, {}, {}
    for r in rows:
        sid = r.get("sample_id", "<missing>")
        if sid in ids or sid == "<missing>":
            errors.append(f"duplicate/missing sample_id: {sid}")
        ids[sid] = True
        for key in ("patient_id", "dataset", "sequence", "metadata_source"):
            if not r.get(key) or str(r[key]).lower() == "unknown":
                errors.append(f"{sid}: missing verified {key}")
        if r.get("patient_verified") is not True:
            errors.append(f"{sid}: patient identity not verified")
        if r.get("sequence_verified") is not True:
            errors.append(f"{sid}: sequence not verified")
        if not isinstance(r.get("disease_labels"), list) or not r["disease_labels"]:
            errors.append(f"{sid}: disease_labels must be a nonempty list (unknown allowed)")
        elif "unknown" in r["disease_labels"]:
            warnings.append(f"{sid}: disease evaluation unavailable")
        if r.get("label") not in (0, 1):
            errors.append(f"{sid}: label must be 0 or 1")
        split = r.get("split")
        if require_splits and split not in SPLITS:
            errors.append(f"{sid}: split must be train/validation/test")
        patients[r.get("patient_id")].add(split)
        p = str(Path(r.get("image_path", "")).resolve()).casefold()
        if p in paths:
            errors.append(f"{sid}: duplicate image path with {paths[p]}")
        paths[p] = sid
        if not verify_files:
            continue
        try:
            with Image.open(r["image_path"]) as im:
                gray = np.asarray(im.convert("L"))
            # Decode pixels as well as hashing bytes to catch re-encoded copies.
            content = stable_digest([gray.shape, __import__("hashlib").sha256(gray.tobytes()).hexdigest()])
            if content in images and images[content][0] != split and require_splits:
                errors.append(f"{sid}: duplicate image pixels across splits with {images[content][1]}")
            elif content in images:
                warnings.append(f"{sid}: duplicate decoded image content with {images[content][1]}")
            images[content] = (split, sid)
            r["image_pixel_sha256"] = content
            sha = file_sha256(r["image_path"])
            if r.get("image_sha256") and r["image_sha256"] != sha:
                errors.append(f"{sid}: image content changed")
            r["image_sha256"] = sha
            if r.get("mask_path"):
                with Image.open(r["mask_path"]) as im:
                    mask = np.asarray(im.convert("L")) > 0
                if mask.shape != gray.shape:
                    errors.append(f"{sid}: image/mask shape mismatch")
                if bool(mask.any()) != bool(r["label"]):
                    errors.append(f"{sid}: image label and mask disagree")
                sha = file_sha256(r["mask_path"])
                if r.get("mask_sha256") and r["mask_sha256"] != sha:
                    errors.append(f"{sid}: mask content changed")
                r["mask_sha256"] = sha
                if mask.any() and sha in masks and masks[sha][0] != split:
                    warnings.append(f"{sid}: identical nonempty mask across splits: {masks[sha][1]}")
                masks[sha] = (split, sid)
            elif r.get("label") == 1:
                errors.append(f"{sid}: positive image needs a nonempty mask")
            if r.get("roi_path"):
                with Image.open(r["roi_path"]) as im:
                    if im.size != (gray.shape[1], gray.shape[0]):
                        errors.append(f"{sid}: ROI size mismatch")
                sha = file_sha256(r["roi_path"])
                if r.get("roi_sha256") and r["roi_sha256"] != sha:
                    errors.append(f"{sid}: ROI content changed")
                r["roi_sha256"] = sha
        except (OSError, KeyError, ValueError) as exc:
            errors.append(f"{sid}: {exc}")
    if require_splits:
        for pid, splits in patients.items():
            if len(splits) > 1:
                errors.append(f"patient crosses splits: {pid}: {sorted(map(str, splits))}")
    return {"version": VERSION, "valid": not errors, "errors": errors, "warnings": warnings,
            "patient_split_evidence": patient_split_evidence(rows),
            "samples": len(rows), "patients": len(patients),
            "split_counts": dict(Counter(r.get("split", "unassigned") for r in rows)),
            "digest": stable_digest(sorted(rows, key=lambda r: r["sample_id"]))}


def require_valid(rows, **kwargs):
    report = audit_rows(rows, **kwargs)
    if not report["valid"]:
        raise ValueError("Manifest audit failed:\n" + "\n".join(report["errors"][:30]))
    return report


def assign_patient_splits(rows, seed=0, fractions=(.7, .15, .15), heldout_diseases=()):
    require_valid(rows, require_splits=False)
    if len(fractions) != 3 or min(fractions) <= 0 or not np.isclose(sum(fractions), 1):
        raise ValueError("Three positive split fractions must sum to one")
    patients = sorted({r["patient_id"] for r in rows})
    heldout = set(heldout_diseases)
    if heldout and any("unknown" in r["disease_labels"] or not r.get("disease_verified") for r in rows if r["label"]):
        raise ValueError("Disease holdout requires verified disease labels on all positive images")
    final_patients = {r["patient_id"] for r in rows if heldout.intersection(r["disease_labels"])}
    remaining = [p for p in patients if p not in final_patients]
    rng = np.random.default_rng(seed)
    rng.shuffle(remaining)
    n = len(remaining)
    n_val = max(1, round(n * fractions[1]))
    n_test = 0 if heldout else max(1, round(n * fractions[2]))
    if n - n_val - n_test < 1:
        raise ValueError("Too few patients for train/validation/test")
    mapping = {p: "test" for p in final_patients}
    for i, p in enumerate(remaining):
        mapping[p] = "validation" if i < n_val else "test" if i < n_val + n_test else "train"
    result = [dict(r, split=mapping[r["patient_id"]]) for r in rows]
    require_valid(result)
    # Fail visibly instead of silently searching seeds until a favourable split.
    for s in SPLITS:
        labels = {r["label"] for r in result if r["split"] == s}
        if labels != {0, 1}:
            raise ValueError(f"{s} lacks normal/abnormal images; supply a reviewed patient split")
    return result


def choose_support(rows, seed=0, normal=2, abnormal=3):
    require_valid(rows)
    rng = np.random.default_rng(seed)
    chosen, used_patients, used_content = [], set(), set()
    for label, count in ((0, normal), (1, abnormal)):
        pool = [r for r in rows if r["split"] == "train" and r["label"] == label]
        rng.shuffle(pool)
        unique = []
        seen_content = set(used_content)
        for r in pool:
            key = r["image_pixel_sha256"]
            if key not in seen_content:
                unique.append(r)
                seen_content.add(key)
        pool = unique
        if len(pool) < count:
            raise ValueError(f"Not enough training slices for label={label}, quota={count}")
        for _ in range(count):
            idx = next((i for i, r in enumerate(pool) if r["patient_id"] not in used_patients), 0)
            r = pool.pop(idx)
            chosen.append(r["sample_id"])
            used_patients.add(r["patient_id"])
            used_content.add(r["image_pixel_sha256"])
    return {"version": VERSION, "unit": "slice", "seed": seed, "sample_ids": chosen,
            "normal": normal, "abnormal": abnormal, "patients": len(used_patients),
            "split_digest": stable_digest(sorted(rows, key=lambda r: r["sample_id"])),
            "support_digest": stable_digest(chosen)}


def resolve_support(rows, support, audit=None):
    audit = require_valid(rows) if audit is None else audit
    if not audit.get("valid") or audit["digest"] != stable_digest(sorted(rows, key=lambda r: r["sample_id"])):
        raise ValueError("Stale/invalid input audit")
    if support["split_digest"] != audit["digest"] or support["support_digest"] != stable_digest(support["sample_ids"]):
        raise ValueError("Support/split digest mismatch")
    lookup = {r["sample_id"]: r for r in rows}
    ids = support["sample_ids"]
    if len(set(ids)) != len(ids) or any(s not in lookup for s in ids):
        raise ValueError("Support must contain unique existing sample IDs")
    selected = [lookup[s] for s in ids]
    if len({r["image_pixel_sha256"] for r in selected}) != len(selected):
        raise ValueError("Support contains duplicated decoded image content; K must count distinct slices")
    if any(r["split"] != "train" for r in selected):
        raise ValueError("Only training support may supply real or synthetic samples")
    if [sum(r["label"] == k for r in selected) for k in (0, 1)] != [support["normal"], support["abnormal"]]:
        raise ValueError("Support class counts mismatch")
    if support.get("patients") != len({r["patient_id"] for r in selected}):
        raise ValueError("Support patient count mismatch")
    return selected
