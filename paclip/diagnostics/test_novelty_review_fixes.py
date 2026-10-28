"""Regression coverage for the user's P1/P2 identity and interface review."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch

from paclip.data.patient_manifest import audit_rows, choose_support, resolve_support, patient_split_evidence
from paclip.data.novelty_dataset import NoveltyDataset
from paclip.data.fewshot import stable_digest
from paclip.data.synthetic_image import ImageSyntheticConfig, synthesize_image
from paclip.engine.novelty_trainer import NoveltyTrainer, checked_config
from paclip.engine.novelty_cache import CleanFeatureCache
from paclip.engine.novelty_provenance import BASELINE_VERSION, validate_baseline_provenance, validate_embedded_baseline
from paclip.implementation_identity import implementation_fingerprint
from paclip.models.novelty_model import NoveltyModel
from paclip.losses_novelty import grouped_novelty_loss, source_localization_metrics
from paclip.evaluation.novelty_protocol import resolve_condition, evaluation_coverage
from paclip.diagnostics.novelty_fixtures import fixture_rows, TinyEncoder


class ReviewFixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.rows = fixture_rows(self.root)
        self.support = choose_support(self.rows)
        self.audit = audit_rows(self.rows)
        self.selected = resolve_support(self.rows, self.support)

    def copy_identity_sources(self):
        package = Path(__file__).resolve().parents[1]
        copied = self.root / "source_copy" / "paclip"
        for source in package.rglob("*.py"):
            if "diagnostics" in source.relative_to(package).parts:
                continue
            target = copied / source.relative_to(package)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
        for name in ("run_brain_novelty.py", "prepare_patient_splits.py"):
            (copied.parent / name).write_bytes((package.parent / name).read_bytes())
        return copied

    def provenance(self):
        direction = torch.linspace(-1, 1, 512).tolist()
        return {"version": BASELINE_VERSION, "split_digest": self.audit["digest"],
            "support_digest": self.support["support_digest"], "legacy_support_digest": "legacy-fixture",
            "seed": 0, "initialization": "original_biomedclip_pretrained_no_learned_checkpoint",
            "patient_disjoint_verified": True, "support_unique_patients": self.support["patients"],
            "steps": 2700, "completed_steps": 2700,
            "diagnostic_direction": direction, "diagnostic_direction_digest": stable_digest(direction)}

    def test_support_repeated_patient_is_distinct_from_cross_split_leakage(self):
        for r in self.rows:
            if r["split"] == "train":
                r["patient_id"] = "one-training-patient"
        support = choose_support(self.rows)
        selected = resolve_support(self.rows, support)
        self.assertEqual(len(selected), 5)
        self.assertEqual(support["patients"], 1)
        self.assertTrue(patient_split_evidence(self.rows)["case_disjoint_verified"])
        self.rows[-1]["patient_id"] = "one-training-patient"
        self.assertFalse(patient_split_evidence(self.rows)["case_disjoint_verified"])
        self.assertFalse(audit_rows(self.rows)["valid"])

    def test_false_support_patient_count_rejected(self):
        self.support["patients"] = 100
        with self.assertRaisesRegex(ValueError, "patient count"):
            resolve_support(self.rows, self.support)

    def test_checkpoint_rejects_changes_in_all_previously_omitted_dependencies(self):
        copied = self.copy_identity_sources()
        fingerprint = lambda scope: implementation_fingerprint(scope, copied)
        with patch("paclip.engine.novelty_trainer.implementation_fingerprint", side_effect=fingerprint):
            trainer = NoveltyTrainer({"steps": 1}, self.rows, self.selected, {}, self.root / "run", TinyEncoder)
            trainer.output.mkdir()
            trainer.save_checkpoint("checkpoint_last.pt")
            for name in ("models/paclip.py", "models/visual_inlayer.py", "models/text_inlayer.py", "data/datasets.py",
                         "data/novelty_dataset.py", "data/supervision.py", "evaluation/ranking.py", "evaluation/detection.py",
                         "evaluation/localization_budget.py", "../run_brain_novelty.py"):
                with self.subTest(file=name):
                    path = copied / name
                    original = path.read_bytes()
                    path.write_bytes(original + b"\n# identity regression probe\n")
                    try:
                        updated = NoveltyTrainer({"steps": 1}, self.rows, self.selected, {}, self.root / "other", TinyEncoder)
                        with self.assertRaisesRegex(ValueError, "identity mismatch"):
                            updated.load_checkpoint(trainer.output / "checkpoint_last.pt")
                    finally:
                        path.write_bytes(original)

    def test_novelty_dataset_change_invalidates_feature_cache(self):
        copied = self.copy_identity_sources()
        with patch("paclip.engine.novelty_cache.implementation_fingerprint", side_effect=lambda scope: implementation_fingerprint(scope, copied)):
            sample = NoveltyDataset(self.selected)[0]
            first = CleanFeatureCache(self.root / "cache", "base-fixture")
            first.put(sample, torch.ones(3, 196, 512), (14, 14))
            self.assertIsNotNone(first.get(sample))
            path = copied / "data/novelty_dataset.py"
            path.write_bytes(path.read_bytes() + b"\n# changed ROI/preprocess\n")
            second = CleanFeatureCache(self.root / "cache", "base-fixture")
            self.assertNotEqual(first.key(sample), second.key(sample))
            self.assertIsNone(second.get(sample))

    def test_short_baseline_cannot_satisfy_formal_budget(self):
        provenance = self.provenance()
        config = checked_config({"steps": 20})  # Head short run does not lower baseline requirement.
        validate_baseline_provenance(provenance, config, self.audit, self.support)
        for requested, completed in ((20, 20), (2700, 20), (20, 2700)):
            with self.subTest(requested=requested, completed=completed), self.assertRaisesRegex(ValueError, "budget mismatch"):
                validate_baseline_provenance({**provenance, "steps": requested, "completed_steps": completed}, config, self.audit, self.support)
        validate_baseline_provenance({**provenance, "steps": 20, "completed_steps": 20},
                                    checked_config({"baseline_steps": 20}), self.audit, self.support)

    def test_embedded_steps_and_patient_ids_must_match(self):
        p = self.provenance()
        meta = {"support_digest": p["legacy_support_digest"], "case_disjoint_verified": True,
                "step": 2700, "steps_this_run": 2700, "seed": 0, "init_checkpoint_sha256": "",
                "support_set": {"samples": [{"sample_id": r["sample_id"], "case_id": r["patient_id"], "label": r["label"]} for r in self.selected]}}
        validate_embedded_baseline(meta, p, self.selected)
        with self.assertRaisesRegex(ValueError, "step count"):
            validate_embedded_baseline({**meta, "step": 20}, p, self.selected)
        meta["support_set"]["samples"][0]["case_id"] = "wrong"
        with self.assertRaisesRegex(ValueError, "patient identities"):
            validate_embedded_baseline(meta, p, self.selected)

    def test_maps_are_actual_frozen_text_matches_and_dense_alias(self):
        direction = torch.linspace(-.2, .2, 512)
        model = NoveltyModel(diagnostic_direction=direction)
        levels = torch.randn(2, 3, 6, 512, requires_grad=True)
        output = model.decode_levels(levels, (2, 3), torch.ones(2, 8, 12))
        for i, k in enumerate((5, 8, 11)):
            expected = (levels[:, i].detach() @ direction).reshape(2, 2, 3).sigmoid()
            torch.testing.assert_close(output[f"map_layer{k}"], expected, atol=1e-6, rtol=1e-6)
            self.assertFalse(output[f"map_layer{k}"].requires_grad)
        self.assertTrue(torch.equal(output["dense_anomaly_map"], output["dense_probability"][:, 0]))
        self.assertTrue(output["layer_maps_available"])

    def test_cached_maps_and_checkpoint_do_not_require_encoder_loading(self):
        direction = torch.linspace(-.2, .2, 512)
        loader = lambda: (_ for _ in ()).throw(AssertionError("unexpected encoder loading"))
        a = NoveltyTrainer({"steps": 1}, self.rows, self.selected, {}, self.root / "a", loader, diagnostic_direction=direction)
        a.output.mkdir(); a.save_checkpoint("last.pt")
        b = NoveltyTrainer({"steps": 1}, self.rows, self.selected, {}, self.root / "b", loader, diagnostic_direction=direction)
        b.load_checkpoint(a.output / "last.pt")
        levels = torch.randn(1, 3, 6, 512)
        for key in ("map_layer5", "map_layer8", "map_layer11", "dense_anomaly_map"):
            self.assertTrue(torch.equal(a.model.decode_levels(levels, (2, 3), torch.ones(1, 8, 12))[key],
                                        b.model.decode_levels(levels, (2, 3), torch.ones(1, 8, 12))[key]))

    def test_loss_syn_is_group_mean_not_sum_of_subgroup_means(self):
        masks = [torch.eye(8)] * 3
        logits = torch.randn(3, 1, 8, 8)
        result = grouped_novelty_loss(logits, masks, torch.ones(3, 8, 8), ["synthetic"] * 3, ["image", "image", "feature"])
        self.assertTrue(torch.allclose(result["loss_syn"], (2 * result["loss_syn_image"] + result["loss_syn_feature"]) / 3))
        self.assertIn("loss_rank", result)

    def test_source_metrics_count_lesions_and_use_pooled_normal_roi_area(self):
        score = torch.full((4, 1, 224, 224), -10.)
        roi = torch.ones(4, 224, 224, dtype=torch.bool)
        masks = [torch.zeros(224, 224) for _ in range(4)]
        masks[0][10:12, 10:12] = 1; masks[0][200:202, 200:202] = 1
        roi[0, 200:202, 200:202] = False
        score[0, 0, 10:12, 10:12] = 10
        masks[1][40:44, 40:44] = 1; score[1, 0, 40:44, 40:44] = 10
        roi[2:] = False; roi[2, :10, :10] = True; roi[3, :20, :20] = True
        score[2, 0, 0, :5] = 10; score[3, 0, 0, :20] = 10
        result = source_localization_metrics(score, masks, roi, ["real", "synthetic", "normal", "normal"], ["none", "image", "none", "none"])
        self.assertEqual(result["real_lesion_recall"], .5)
        self.assertEqual(result["synthetic_lesion_recall"], 1.)
        self.assertEqual(result["normal_false_positive_area"], 25 / 500)
        empty = source_localization_metrics(score[2:], masks[2:], roi[2:], ["normal"] * 2, ["none"] * 2)
        self.assertIsNone(empty["synthetic_lesion_recall"])

    def test_extended_transforms_are_replayable_and_roi_limited(self):
        x = torch.from_numpy(np.random.default_rng(4).uniform(.2, .8, (224, 224)).astype(np.float32))
        roi = torch.zeros(224, 224, dtype=torch.bool); roi[20:204, 20:204] = True
        for mode in ("texture_replace", "dilate", "compress"):
            with self.subTest(mode=mode):
                cfg = ImageSyntheticConfig(modes=(mode,))
                a = synthesize_image(x, roi, 19, config=cfg)
                b = synthesize_image(x, roi, 19, config=cfg)
                self.assertTrue(torch.equal(a["gray"], b["gray"]))
                self.assertTrue(torch.equal(a["gray"][~roi], x[~roi]))
                self.assertTrue(torch.equal((a["gray"] - x).abs() > 1e-6, a["mask_high"].bool()))
                self.assertEqual(a["parameters"]["mode"], mode)

    def test_first_round_and_visual_ablation_defaults_are_distinct(self):
        root = Path(__file__).resolve().parents[2]
        self.assertFalse(checked_config({})["train_visual"])
        config = json.loads((root / "configs/novelty/joint_visual_tuning.json").read_text())
        self.assertTrue(checked_config(config)["train_visual"])
        self.assertEqual(len(ImageSyntheticConfig().modes), 4)

    def test_auto_evaluation_refuses_unverified_disease_and_reports_missing_conditions(self):
        target = [r for r in self.rows if r["split"] == "test"]
        self.assertEqual(resolve_condition(self.rows, target), "same_sequence_seen")
        for r in target:
            if r["label"]:
                r["disease_labels"] = ["unknown"]
        with self.assertRaisesRegex(ValueError, "Cannot infer"):
            resolve_condition(self.rows, target)
        self.assertEqual(resolve_condition(self.rows, target, "exploratory"), "exploratory")
        coverage = evaluation_coverage([("source", target, "same_sequence_seen")])
        self.assertEqual(len(coverage), 4)
        self.assertEqual(coverage["same_sequence_unseen"]["availability"], "unavailable")
