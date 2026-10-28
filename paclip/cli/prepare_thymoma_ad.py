"""Build patient-disjoint thymoma CT + public NLST slice manifests.

Keep the existing 47-patient private test list. The support pool contains
private lesion slices and public screening-negative controls only. Additional
private lesion-free slices are held-out source-confounding diagnostics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


def gray(hu):
    return np.round(np.clip((hu.astype(np.float32) + 160.0) / 400.0, 0, 1) * 255).astype(np.uint8)


def save_slice(root, split, case, z, plane, mask, label, source):
    name = f"{case}_z{z:04d}.png"
    relative = Path(split) / ("abnormal" if label else "normal") / name
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(gray(plane)).resize((224, 224), Image.Resampling.BILINEAR)
    image.save(target)
    mask_relative = None
    if label:
        mask_relative = Path("masks") / split / "abnormal" / name
        target_mask = root / mask_relative
        target_mask.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray((mask > 0).astype(np.uint8) * 255).resize((224, 224), Image.Resampling.NEAREST).save(target_mask)
        assert np.asarray(Image.open(target_mask)).any()
    return {"split": split, "sample_id": relative.as_posix(), "image": relative.as_posix(),
            "mask": None if mask_relative is None else mask_relative.as_posix(), "label": int(label),
            "case_id": case, "source": source, "slice_z_lps": int(z),
            "image_sha256": hashlib.sha256(target.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--private-split", type=Path, required=True)
    parser.add_argument("--normal-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    volumes = sorted(args.private_root.rglob("*000-origin.nii.gz"))
    assert len(volumes) == 233, f"Expected 233 plain CT volumes, got {len(volumes)}"
    ids = {p.name[:3] for p in volumes}
    original = json.loads(args.private_split.read_text(encoding="utf-8"))
    test = set(original["test_patients"])
    assert len(test) == 47 and test.issubset(ids)
    pool = sorted(ids - test)
    rng = np.random.RandomState(0)
    rng.shuffle(pool)
    val = set(pool[:23])
    train = set(pool[23:])
    assert not (train & val or train & test or val & test)
    plan = {"train": sorted(train), "val": sorted(val), "test": sorted(test)}
    (args.out / "private_patient_split.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")

    def prepare(path):
        pid = path.name[:3]
        split = "test" if pid in test else "val" if pid in val else "train"
        mask_path = path.with_name(path.name.replace("-origin", ""))
        image_nifti, mask_nifti = nib.load(path), nib.load(mask_path)
        assert image_nifti.shape == mask_nifti.shape and np.allclose(image_nifti.affine, mask_nifti.affine, atol=1e-4)
        # LPS voxel axes, followed by transpose, give the same axial display as
        # the standard LPS DICOM controls. Native NIfTI (x,y,z) is not image(y,x).
        orient = nib.orientations.ornt_transform(nib.orientations.io_orientation(image_nifti.affine),
                                               nib.orientations.axcodes2ornt(("L", "P", "S")))
        volume = nib.orientations.apply_orientation(np.asanyarray(image_nifti.dataobj), orient)
        roi = nib.orientations.apply_orientation(np.asanyarray(mask_nifti.dataobj), orient)
        counts = (roi > 0).sum(axis=(0, 1))
        positive = np.flatnonzero(counts > 0)
        assert len(positive), f"No lesion for private case {pid}"
        chosen = sorted(positive[np.argsort(-counts[positive], kind="stable")[:8]].tolist())
        records = [save_slice(args.out, split, "PRIVATE_" + pid, z, volume[:, :, z].T, roi[:, :, z].T, 1, "private_lesion") for z in chosen]
        if split != "train":
            eligible = np.flatnonzero((counts == 0) & (np.arange(len(counts)) >= .15 * len(counts)) & (np.arange(len(counts)) <= .65 * len(counts)))
            if len(eligible):
                indices = np.unique(np.linspace(0, len(eligible) - 1, min(4, len(eligible))).round().astype(int))
                for z in eligible[indices]:
                    records.append(save_slice(args.out, split, "PRIVATE_" + pid, int(z), volume[:, :, z].T, roi[:, :, z].T, 0, "private_lesion_free_slice"))
        return records

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for i, records in enumerate(executor.map(prepare, volumes), 1):
            rows.extend(records)
            if i % 20 == 0 or i == len(volumes):
                print(f"private preparation {i}/{len(volumes)}", flush=True)
    controls = pd.read_csv(args.normal_root / "manifest.csv", dtype={"patient_id": str})
    for row in controls.to_dict("records"):
        src = args.normal_root / row["npz_path"]
        assert hashlib.sha256(src.read_bytes()).hexdigest() == row["npz_sha256"]
        with np.load(src, allow_pickle=False) as sample:
            assert int(sample["label"]) == 0
            png = np.round(sample["image"] * 255).astype(np.uint8)
        relative = Path(row["split"]) / "normal" / (row["sample_id"] + ".png")
        target = args.out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(png).save(target)
        rows.append({"split": row["split"], "sample_id": relative.as_posix(), "image": relative.as_posix(),
                     "mask": None, "label": 0, "case_id": "NLST_" + row["patient_id"], "source": "NLST_screening_negative",
                     "image_sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
    rows.sort(key=lambda r: (r["split"], r["sample_id"]))
    seen_cases = {}
    for row in rows:
        assert row["case_id"] not in seen_cases or seen_cases[row["case_id"]] == row["split"]
        seen_cases[row["case_id"]] = row["split"]
    with (args.out / "splits.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {"private_patients": {k: len(v) for k, v in plan.items()}, "samples": {},
               "orientation": "LPS voxel axes then axial transpose to DICOM display", "window": {"level": 40, "width": 400},
               "positive_slice_selection": "up to 8 largest lesion-area slices per patient, using annotation for dataset construction only",
               "private_negative_role": "validation/test slice-level source-confounding diagnostic; not healthy subjects and not training support",
               "normal_provenance": str(args.normal_root), "original_test_split_preserved": True}
    for split in ["train", "val", "test"]:
        subset = [r for r in rows if r["split"] == split]
        summary["samples"][split] = {"normal": sum(r["label"] == 0 for r in subset), "abnormal": sum(r["label"] == 1 for r in subset),
                                     "private_lesion_free": sum(r["source"] == "private_lesion_free_slice" for r in subset)}
    (args.out / "preparation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    panels = []
    for split in ["train", "val", "test"]:
        for label in [0, 1]:
            first = next(r for r in rows if r["split"] == split and r["label"] == label)
            panel = Image.new("RGB", (224, 248), "black")
            panel.paste(Image.open(args.out / first["image"]).convert("RGB"), (0, 24))
            ImageDraw.Draw(panel).text((3, 4), f"{split} label={label} {first['case_id']}", fill="white")
            panels.append(panel)
    preview = Image.new("RGB", (3 * 224, 2 * 248), "black")
    for i, panel in enumerate(panels):
        preview.paste(panel, ((i // 2) * 224, (i % 2) * 248))
    preview.save(args.out / "input_preview.jpg", quality=95)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
