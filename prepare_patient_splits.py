"""Audit explicit metadata; never infer verified patient identity from filenames."""
import argparse
from pathlib import Path
from paclip.data.patient_manifest import (read_rows, write_rows, write_json, audit_rows,
                                         assign_patient_splits, choose_support)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path)
    source.add_argument("--scan-root", type=Path, help="Write an UNVERIFIED inventory from existing slice folders")
    p.add_argument("--mask-root", type=Path)
    p.add_argument("--dataset", default="local_mri")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--assign-splits", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--heldout-disease", action="append", default=[])
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    if a.scan_root:
        rows = []
        for split in ("train", "val", "test"):
            for label, category in ((0, "normal"), (1, "abnormal")):
                folder = a.scan_root / split / category
                for image in sorted(folder.rglob("*")):
                    if not image.is_file() or image.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
                        continue
                    relative = image.relative_to(a.scan_root)
                    mask = a.mask_root / relative if a.mask_root else None
                    rows.append({"sample_id": a.dataset + "/" + relative.as_posix(),
                        "patient_id": "", "patient_verified": False, "study_id": "", "volume_id": "",
                        "image_path": str(image.resolve()), "mask_path": str(mask.resolve()) if mask and mask.exists() else None,
                        "label": label, "dataset": a.dataset, "sequence": "unknown", "sequence_verified": False,
                        "disease_labels": ["unknown"] if label else ["normal"], "disease_verified": False,
                        "slice_z": None, "orientation_verified": False, "volume_complete": False,
                        "roi_verified": False, "metadata_source": "", "legacy_split": split, "split": None})
        write_rows(a.output / "inventory_UNVERIFIED.jsonl", rows)
        write_json(a.output / "missing_metadata.json", {"samples": len(rows), "certified": False,
            "normal": sum(r["label"] == 0 for r in rows), "abnormal": sum(r["label"] == 1 for r in rows),
            "positive_mask_missing": sum(r["label"] == 1 and not r["mask_path"] for r in rows),
            "missing_for_all": ["global_patient_id_and_evidence", "verified_sequence", "reviewed_roi_policy", "patient_split"],
            "unknown_disease_experiment_also_requires": ["verified_disease_labels", "heldout_disease_protocol"],
            "warning": "Original slice splits retained as legacy_split only. Do not set verified=true without evidence."})
        print(f"Wrote {len(rows)} UNVERIFIED inventory rows. No patient split was certified.")
        return
    rows = read_rows(a.manifest)
    report = audit_rows(rows, require_splits=not a.assign_splits)
    write_json(a.output / "audit.json", report)
    if not report["valid"]:
        raise SystemExit("Metadata audit failed; see audit.json. No split was certified.")
    if a.assign_splits:
        rows = assign_patient_splits(rows, a.seed, heldout_diseases=a.heldout_disease)
    support = choose_support(rows, seed=a.seed)
    write_rows(a.output / "split_manifest.jsonl", rows)
    write_json(a.output / "support_manifest.json", support)
    write_json(a.output / "audit.json", audit_rows(rows))
    print(f"Audited {len(rows)} slices; support={len(support['sample_ids'])}, patients={support['patients']}")


if __name__ == "__main__":
    main()
