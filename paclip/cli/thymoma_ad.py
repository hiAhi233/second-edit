"""Run the same K=5/seed=0/2700-step TVJ protocol as the recorded brain run."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from paclip.paths import PROJECT_ROOT

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from PIL import Image, ImageDraw

from paclip.data.datasets import SliceAnomalyDataset, load_manifest_splits
from paclip.data.fewshot import sample_support_indices, validate_splits, file_sha256
from paclip.engine.trainer import make_args, run_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data/thymoma_nlst_training_20261021")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "runs/thymoma_nlst_20261021")
    options = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError("Use the existing torch-gpu1 environment; CPU-only torch is not the training runtime")
    root = PROJECT_ROOT
    data = options.data_dir.resolve()
    out = options.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    # E1 keeps tiny positive lesions; only truly empty training masks are excluded.
    original_rows = [json.loads(line) for line in (data / "splits.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    eligible_rows, excluded = [], []
    for row in original_rows:
        if row["split"] == "train" and row["label"] == 1:
            native = np.asarray(Image.open(data / row["mask"]).convert("L"))
            if not (native > 0).any():
                excluded.append(row["sample_id"])
                continue
        eligible_rows.append(row)
    manifest = data / "training_splits.jsonl"
    manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in eligible_rows), encoding="utf-8")
    (out / "support_eligibility.json").write_text(json.dumps({"rule": "positive training mask contains a lesion at native resolution; tiny lesions retained by adaptive max pooling", "excluded_train_slices": excluded, "evaluation_unchanged": True}, indent=2), encoding="utf-8")
    splits = load_manifest_splits(str(manifest))
    source_by_id = {row["sample_id"]: row["source"] for row in
                    (json.loads(line) for line in (data / "splits.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())}
    assert validate_splits(splits["train"], splits["val"], splits["test"], require_cases=True)["case_disjoint_verified"]
    indices, info = sample_support_indices(splits["train"], 5, 0)
    assert info["normal_count"] == 2 and info["abnormal_count"] == 3 and info["unique_cases"] == 5
    dataset = SliceAnomalyDataset(str(data), records=splits["train"])
    preview = Image.new("RGB", (5 * 224, 248), "black")
    draw = ImageDraw.Draw(preview)
    support = []
    for position, index in enumerate(indices):
        item, row = dataset[index], splits["train"][index]
        assert int(item["label"]) == row["label"]
        if row["label"]:
            assert item["mask"].any(), "Positive support lesion vanished on the patch grid"
        else:
            assert source_by_id[row["sample_id"]] == "NLST_screening_negative" and not item["mask"].any()
        preview.paste(Image.open(row["path"]).convert("RGB"), (224 * position, 24))
        draw.text((224 * position + 2, 3), f"label={row['label']} {row['case_id']}", fill="white")
        support.append({k: row[k] for k in ["sample_id", "case_id", "label"]})
    preview.save(out / "support_preview.jpg", quality=95)
    args = make_args(
        str(data), task="thymoma", split_manifest=str(manifest),
        data_format="slice", n_support=5, seed=0, steps=2700,
        batch_size=8, lr=1e-4, num_workers=0, device="cuda",
        organ="thymoma", prompt_set="thymoma_sentence",
        inlayer=True, visual_inlayer=True, train_stage="joint",
        save_dir=str(out), score_mode="local_topk", topk_ratio=.05,
        mask_mode="fixed", rows_per_page=8, panel_size=256,
    )
    protocol = {"task": "thymoma", "reference": "brain MRI TVJ K=5 seed=0 checkpoint source_step=2700",
                "k_total_slices": 5, "seed": 0, "steps": 2700, "requested_batch": 8, "effective_batch": 8,
                "optimizer": "AdamW", "lr": 1e-4, "weight_decay": 1e-5,
                "loss_weights": {"text": 1, "global": 1, "local": 1, "diversity": 0},
                "local_loss": {"focal": 1.0, "dice": 1.0, "boundary": 0.1, "gamma": 2.0, "alpha_positive": 0.75},
                "training_sampling": "50% positive, 25% normal, 25% hard when available; see training_sampling.json",
                "text_layers": [8, 9, 10, 11], "visual_layers": [5, 8, 11],
                "inlayer_positions": ["attn", "ffn"], "inlayer_bottleneck": 64,
                "output_text_adapter": "disabled and frozen by existing brain TVJ protocol",
                "support": support, "patient_disjoint": True,
                "thresholds": "validation only; held-out test is never used for threshold tuning",
                "private_test_list": "existing 47-patient seed111 list preserved",
                "support_pool_filter": "native training mask must be nonempty; no coarse-grid exclusion; held-out data unchanged",
                "excluded_training_slices": len(excluded),
                "slice_level_task": "visible annotated lesion on a selected slice, not patient-level diagnosis",
                "source_files": {name: file_sha256(str(root / name)) for name in [
                    "paclip/models/paclip.py", "paclip/engine/trainer.py", "paclip/prompts/__init__.py",
                    "paclip/prompts/base.py", "paclip/prompts/thymoma.py", "paclip/data/fewshot.py",
                    "paclip/cli/prepare_thymoma_ad.py",
                ]}}
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[preflight]", json.dumps(info), "patient-disjoint=True", flush=True)
    result = run_experiment(args)
    compact = {k: v for k, v in result.items() if k not in {"seen_sample_ids", "stage_optimizer_ids"}}
    (out / "result_summary.json").write_text(json.dumps(compact, indent=2, ensure_ascii=False), encoding="utf-8")
    if result["status"] != "visualized" or result["completed_steps"] not in [0, 2700]:
        raise RuntimeError("Training/evaluation/visualization incomplete; inspect saved status")
    print("[finished]", result["run_dir"], flush=True)


if __name__ == "__main__":
    main()
