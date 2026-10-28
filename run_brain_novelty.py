"""Independent audited baseline and novelty experiments. No training on import."""
import argparse
import json
import os
from pathlib import Path
import subprocess

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from paclip.data.patient_manifest import read_rows, write_rows, write_json, require_valid, resolve_support, patient_split_evidence
from paclip.data.fewshot import file_sha256, stable_digest
from paclip.engine.novelty_provenance import BASELINE_VERSION, validate_baseline_provenance, validate_embedded_baseline


def check_device(device):
    if not str(device).startswith("cuda"):
        return
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    query = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
                           capture_output=True, text=True, check=True)
    other = [line for line in query.stdout.splitlines() if line.strip() and line.split(",")[0].strip() != str(os.getpid())]
    if other:
        raise RuntimeError("GPU already has another compute process; no training started:\n" + "\n".join(other))


def audited_inputs(args):
    rows = read_rows(args.manifest)
    report = require_valid(rows)
    support = json.loads(args.support.read_text(encoding="utf-8"))
    selected = resolve_support(rows, support, audit=report)
    if len(selected) != 5 or support["normal"] != 2 or support["abnormal"] != 3:
        raise ValueError("Primary experiment requires K=5 total slices (2 normal + 3 real abnormal)")
    development = [r for r in rows if r["split"] != "test"]
    if len({r["sequence"] for r in development}) != 1:
        raise ValueError("Source train/validation must have a single verified sequence")
    from paclip.data.novelty_dataset import NoveltyDataset
    from paclip.data.supervision import pool_mask_targets
    support_samples = list(NoveltyDataset(selected))
    invalid = [s["sample_id"] for s in support_samples if not s["roi_valid"]]
    if invalid:
        raise ValueError(f"Training support ROI not verified: {invalid}")
    for s in support_samples:
        if s["label"]:
            _, gt = pool_mask_targets([s["mask_high"]], s["roi"].shape)
            if not (gt[0].bool() & s["roi"]).any():
                raise ValueError(f"Support lesion is completely outside the reviewed ROI: {s['sample_id']}")
    for split in ("train", "validation", "test"):
        if {r["label"] for r in rows if r["split"] == split} != {0, 1}:
            raise ValueError(f"Source {split} must have normal and abnormal slices")
    valid_validation_labels = {s["label"] for s in NoveltyDataset([r for r in rows if r["split"] == "validation"]) if s["roi_valid"]}
    if valid_validation_labels != {0, 1}:
        raise ValueError("Source validation requires reviewed ROIs in both classes before training")
    return rows, report, support, selected


def load_evaluation_plan(rows, plan_path):
    from paclip.evaluation.novelty_protocol import resolve_condition
    sets, identities, names = [], [], set()
    if not plan_path:
        return sets, identities
    plan_path = Path(plan_path)
    for entry in json.loads(plan_path.read_text(encoding="utf-8")):
        name = entry["name"]
        if not name or name in names or name in {"源集测试", "source_validation"} or any(x in name for x in '/\\:') or name in {".", ".."}:
            raise ValueError("Evaluation names must be unique safe folder names")
        names.add(name)
        path = Path(entry["manifest"])
        if not path.is_absolute():
            path = plan_path.parent / path
        target = read_rows(path)
        target_audit = require_valid(target)
        condition = resolve_condition(rows, target, entry.get("condition", "auto"))
        require_valid([r for r in rows if r["split"] != "test"] + target)
        sets.append((name, target, condition))
        identities.append({"name": name, "condition": condition, "digest": target_audit["digest"]})
    return sets, identities


def run_baseline(args, rows, audit, support, selected):
    """Adapt new manifest to the existing FIRST-step trainer; no old learned init."""
    from paclip.engine.trainer import make_args, run_experiment, default_model_factory
    from paclip.data.datasets import load_manifest_splits
    from paclip.data.fewshot import make_support_manifest
    from paclip.engine.novelty_trainer import checked_config
    task_config = checked_config({"task": getattr(args, "task", None) or "brain_mri",
        "organ": getattr(args, "organ", None), "prompt_set": getattr(args, "prompt_set", None)})
    # Revalidate the audit digest and support mapping even for direct Python callers.
    canonical = resolve_support(rows, support, audit=audit)
    if stable_digest(canonical) != stable_digest(selected):
        raise ValueError("Baseline support rows differ from the audited support")
    evidence = patient_split_evidence(rows)
    if not evidence["case_disjoint_verified"]:
        raise ValueError("Baseline requires verified disjoint train/validation/test patients")
    unique_patients = len({r["patient_id"] for r in selected})
    from paclip.evaluation.novelty_protocol import resolve_condition
    condition = resolve_condition(rows, [r for r in rows if r["split"] == "test"], getattr(args, "source_condition", None) or "auto")
    extra_sets, eval_identity = load_evaluation_plan(rows, getattr(args, "evaluation_plan", None))
    args.output.mkdir(parents=True, exist_ok=False)
    legacy_rows = [{"sample_id": r["sample_id"], "case_id": r["patient_id"],
        "image": r["image_path"], "mask": r.get("mask_path"), "label": r["label"],
        "split": "val" if r["split"] == "validation" else r["split"]} for r in rows]
    legacy_path = args.output / "legacy_split.jsonl"
    write_rows(legacy_path, legacy_rows)
    records = load_manifest_splits(legacy_path)["train"]
    lookup = {r["sample_id"]: i for i, r in enumerate(records)}
    legacy_support = make_support_manifest(records, [lookup[r["sample_id"]] for r in selected], str(args.output),
        {"task": task_config["task"], "support_unit": "slice", "requested_k": 5, "seed": args.seed,
         "case_disjoint_verified": evidence["case_disjoint_verified"], "unique_cases": unique_patients})
    legacy_support_path = args.output / "legacy_support.json"
    write_json(legacy_support_path, legacy_support)
    cfg = make_args(str(args.output), split_manifest=str(legacy_path), support_manifest=str(legacy_support_path),
        # Explicit mask roots enable legacy local supervision; actual files come from manifest.
        mask_root=str(args.output), val_mask_root=str(args.output), eval_mask_root=str(args.output),
        n_support=5, seed=args.seed, steps=args.steps, batch_size=8, lr=1e-4, num_workers=0,
        inlayer=True, visual_inlayer=True, train_stage="joint", device=args.device,
        task=task_config["task"], organ=task_config["organ"],
        prompt_set=task_config["prompt_set"], save_dir=str(args.output / "legacy_run"),
        score_mode="local_topk", topk_ratio=.05, mask_mode="fixed", no_heatmaps=True)
    live_models = []
    def capture_model(*factory_args):
        model = default_model_factory(*factory_args)
        live_models.append(model)
        return model
    result = run_experiment(cfg, model_factory=capture_model)
    write_json(args.output / "legacy_result.json", result)
    if result.get("evaluation_status") != "ok":
        raise RuntimeError("Baseline run did not complete evaluation")
    if result.get("completed_steps") != args.steps:
        raise RuntimeError("Baseline actual completed steps do not match the requested budget")
    checkpoint = Path(result["checkpoint"]).resolve()
    base_sha = file_sha256(checkpoint)
    from paclip.models.novelty_model import frozen_text_direction
    from paclip.prompts import TASK_PROMPT_SETS
    prompts = TASK_PROMPT_SETS[task_config["task"]][task_config["prompt_set"]]
    direction = frozen_text_direction(live_models[-1], prompts).detach().float().cpu().tolist()
    provenance = {"version": BASELINE_VERSION,
        "checkpoint": str(checkpoint), "checkpoint_sha256": base_sha,
        "task": task_config["task"], "organ": task_config["organ"], "prompt_set": task_config["prompt_set"],
        "split_digest": audit["digest"], "support_digest": support["support_digest"],
        "initialization": "original_biomedclip_pretrained_no_learned_checkpoint", "seed": args.seed,
        "steps": args.steps, "completed_steps": result["completed_steps"],
        "patient_disjoint_verified": evidence["case_disjoint_verified"], "patient_split_evidence": evidence,
        "support_unique_patients": unique_patients, "support_repeats_patients": unique_patients < len(selected),
        "diagnostic_direction": direction, "diagnostic_direction_digest": stable_digest(direction),
        "source_rows_digest": audit["digest"], "legacy_support_digest": legacy_support["support_digest"],
        "ms_layers_zero_based": [5, 8, 11]}
    # Retain the live trained model: do not reload a multi-GB checkpoint for evaluation.
    from paclip.evaluation.novelty_baseline import predict_baseline
    from paclip.evaluation.novelty_protocol import fit_operating_points, evaluate
    from paclip.engine.novelty_cache import CleanFeatureCache
    model = live_models[-1]
    cache = CleanFeatureCache(args.cache, base_sha) if args.cache else None
    if cache:
        predict_baseline(model, selected, prompts, args.device, cache)
    validation = predict_baseline(model, [r for r in rows if r["split"] == "validation"], prompts, args.device,
                                  cache, args.output / "evaluation" / "source_validation" / "scores")
    points = fit_operating_points(validation)
    sets = [("源集测试", [r for r in rows if r["split"] == "test"], condition), *extra_sets]
    from paclip.evaluation.novelty_protocol import evaluation_coverage
    results = {"source_validation": evaluate(validation, points, False)}
    for name, target, target_condition in sets:
        predictions = predict_baseline(model, target, prompts, args.device, cache, args.output / "evaluation" / name / "scores")
        results[name] = {"condition": target_condition, **evaluate(predictions, points)}
    write_json(args.output / "protocol.json", {**points, "source_condition": condition, "four_condition_coverage": evaluation_coverage(sets)})
    write_json(args.output / "results.json", results)
    write_json(args.output / "config.json", {"mode": "first_step", "seed": args.seed, "steps": args.steps,
        "task": task_config["task"], "organ": task_config["organ"], "prompt_set": task_config["prompt_set"],
        "lambda_rank": 0., "train_visual": True})
    write_json(args.output / "identity.json", {"base_sha256": base_sha, "split_digest": audit["digest"],
                                               "support_digest": support["support_digest"], "evaluation_plan": eval_identity})
    write_json(args.output / "baseline_provenance.json", provenance)
    write_json(args.output / "completion.json", {"status": "complete", "steps": args.steps,
        "baseline_provenance_sha256": file_sha256(args.output / "baseline_provenance.json"),
        "checkpoint_sha256": base_sha, "results_sha256": file_sha256(args.output / "results.json")})
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["preflight", "baseline", "train"])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--support", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--baseline-provenance", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--evaluation-plan", type=Path, help="JSON list: name, manifest, condition")
    parser.add_argument("--guard-reference", type=Path, help="Completed matched control results.json; source-validation guard only")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--task", choices=["brain_mri", "thymoma"])
    parser.add_argument("--organ")
    parser.add_argument("--prompt-set")
    parser.add_argument("--seed", type=int, help="Baseline seed or explicit override of dense config seed")
    parser.add_argument("--steps", type=int, help="Baseline budget or explicit override of dense config steps")
    parser.add_argument("--baseline-steps", type=int, help="Required completed baseline budget for dense training (default 2700); independent of --steps")
    parser.add_argument("--source-condition", choices=["auto", "exploratory", "same_sequence_seen", "same_sequence_unseen", "external_same_sequence", "cross_sequence"])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    import torch
    torch.set_num_threads(4)
    rows, audit, support, selected = audited_inputs(args)
    if args.action == "preflight":
        write_json(args.output, {"audit": audit, "support": support,
            "formal_unknown_disease_ready": all(r.get("disease_verified") and "unknown" not in r["disease_labels"] for r in rows if r["label"])})
        print("Input preflight passed. No model loaded; no training started.")
        return
    check_device(args.device)
    if args.action == "baseline":
        if args.resume:
            raise ValueError("Legacy baseline has no exact-resume contract; use a fresh output directory")
        args.seed = 0 if args.seed is None else args.seed
        args.steps = 2700 if args.steps is None else args.steps
        if args.seed < 0 or args.steps < 1:
            raise ValueError("Baseline seed/steps invalid")
        run_baseline(args, rows, audit, support, selected)
        return
    if args.config is None or args.baseline_provenance is None:
        parser.error("train requires --config and --baseline-provenance")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    for field in ("task", "organ", "prompt_set"):
        if getattr(args, field) is not None:
            config[field] = getattr(args, field)
    if args.seed is not None:
        config["seed"] = args.seed
    if args.steps is not None:
        config["steps"] = args.steps
    if args.baseline_steps is not None:
        config["baseline_steps"] = args.baseline_steps
    if args.source_condition is not None:
        config["source_condition"] = args.source_condition
    from paclip.engine.novelty_trainer import NoveltyTrainer, checked_config
    config = checked_config(config)
    provenance = json.loads(args.baseline_provenance.read_text(encoding="utf-8"))
    validate_baseline_provenance(provenance, config, audit, support)
    completion = json.loads((args.baseline_provenance.parent / "completion.json").read_text(encoding="utf-8"))
    if (completion.get("status") != "complete" or completion.get("steps") != provenance["completed_steps"]
            or completion.get("checkpoint_sha256") != provenance["checkpoint_sha256"]
            or completion.get("baseline_provenance_sha256") != file_sha256(args.baseline_provenance)):
        raise ValueError("Baseline completion/step/provenance integrity check failed")
    checkpoint = Path(provenance["checkpoint"])
    if file_sha256(checkpoint) != provenance["checkpoint_sha256"]:
        raise ValueError("Baseline checkpoint hash mismatch")
    evaluation_sets, eval_identity = load_evaluation_plan(rows, args.evaluation_plan)
    identity = {"base_sha256": provenance["checkpoint_sha256"], "split_digest": audit["digest"],
                "support_digest": support["support_digest"], "evaluation_plan": eval_identity,
                "baseline_completed_steps": provenance["completed_steps"],
                "baseline_provenance_digest": stable_digest(provenance)}
    guard_reference = None
    if args.guard_reference:
        folder = args.guard_reference.parent
        ref_identity = json.loads((folder / "identity.json").read_text(encoding="utf-8"))
        ref_protocol = json.loads((folder / "protocol.json").read_text(encoding="utf-8"))
        ref_completion = json.loads((folder / "completion.json").read_text(encoding="utf-8"))
        if any(ref_identity.get(k) != identity[k] for k in ("base_sha256", "split_digest", "support_digest")):
            raise ValueError("Guard reference must match baseline weights, patient split and support")
        if ref_protocol.get("fit_split") != "validation" or ref_completion.get("status") != "complete":
            raise ValueError("Guard reference must be a completed source-validation experiment")
        guard_reference = json.loads(args.guard_reference.read_text(encoding="utf-8"))["source_validation"]["operating_points"]["dice"]["lesion_detection"]
        identity["guard_reference_sha256"] = file_sha256(args.guard_reference)
    from paclip.engine.novelty_cache import CleanFeatureCache
    cache = CleanFeatureCache(args.cache, identity["base_sha256"]) if args.cache else None
    def load_encoder():
        from paclip.models.paclip import load_trained_model
        model = load_trained_model(checkpoint, args.device)
        if model.cfg.ms_layers != [5, 8, 11]:
            raise ValueError("Encoder visual depth mismatch")
        meta = model.checkpoint_meta
        validate_embedded_baseline(meta, provenance, selected)
        return model
    trainer = NoveltyTrainer(config, rows, selected, identity, args.output, load_encoder,
                             device=args.device, cache=cache, guard_reference=guard_reference,
                             diagnostic_direction=provenance["diagnostic_direction"])
    trainer.run(resume=args.resume, evaluation_sets=evaluation_sets)


if __name__ == "__main__":
    main()
