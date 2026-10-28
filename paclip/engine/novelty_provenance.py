"""Validate completed baseline budget and embedded support/patient evidence."""
import numpy as np
from paclip.data.fewshot import stable_digest

BASELINE_VERSION = "audited_first_step_v2"


def validate_baseline_provenance(provenance, config, audit, support):
    required = {"version": BASELINE_VERSION, "split_digest": audit["digest"],
        "support_digest": support["support_digest"], "seed": config["seed"],
        "initialization": "original_biomedclip_pretrained_no_learned_checkpoint",
        "patient_disjoint_verified": audit["patient_split_evidence"]["case_disjoint_verified"]}
    if not required["patient_disjoint_verified"] or any(provenance.get(k) != v for k, v in required.items()):
        raise ValueError("Baseline split/support/seed or audited patient evidence does not match")
    # Original v2 records were exclusively brain_mri/brain_mri_sentence/brain.
    for field, old_default in (("task", "brain_mri"), ("organ", "brain"), ("prompt_set", "brain_mri_sentence")):
        if provenance.get(field, old_default) != config[field]:
            raise ValueError(f"Baseline {field} does not match requested task")
    expected = config["baseline_steps"]
    if provenance.get("steps") != expected or provenance.get("completed_steps") != expected:
        raise ValueError(f"Baseline budget mismatch: expected {expected} completed steps; "
                         f"requested={provenance.get('steps')}, completed={provenance.get('completed_steps')}")
    if provenance.get("support_unique_patients") != support["patients"]:
        raise ValueError("Baseline support patient count does not match")
    direction = np.asarray(provenance.get("diagnostic_direction", []), dtype=np.float32)
    if direction.shape != (512,) or not np.isfinite(direction).all() or not np.any(direction):
        raise ValueError("Baseline must contain a finite nonzero frozen text diagnostic direction")
    if provenance.get("diagnostic_direction_digest") != stable_digest(direction.tolist()):
        raise ValueError("Diagnostic text direction digest mismatch")


def validate_embedded_baseline(meta, provenance, selected):
    for field, meta_field, default in (("task", "task", "brain_mri"), ("organ", "active_organ", "brain"), ("prompt_set", "prompt_set", "brain_mri_sentence")):
        if meta.get(meta_field, default) != provenance.get(field, default):
            raise ValueError("Embedded baseline task/organ/prompts do not match")
    if meta.get("support_digest") != provenance["legacy_support_digest"] or meta.get("case_disjoint_verified") is not True:
        raise ValueError("Baseline embedded provenance does not match")
    if meta.get("step") != provenance["completed_steps"] or meta.get("steps_this_run") != provenance["completed_steps"]:
        raise ValueError("Embedded baseline training step count does not match provenance")
    if meta.get("seed") != provenance["seed"] or meta.get("init_checkpoint_sha256") not in (None, ""):
        raise ValueError("Embedded baseline seed/initialization does not match")
    expected = sorted((r["sample_id"], r["patient_id"], r["label"]) for r in selected)
    actual = sorted((r["sample_id"], r.get("case_id"), r["label"]) for r in (meta.get("support_set") or {}).get("samples", []))
    if actual != expected:
        raise ValueError("Embedded support samples/patient identities do not match audited support")
