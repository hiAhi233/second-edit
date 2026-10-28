"""No downloads: python -m unittest paclip.diagnostics.test_e2 -v."""

import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from paclip.models.normal_memory import MemoryConfig, NormalPatchMemory, MemoryAugmentedModel, patch_regions, image_roi
from paclip.evaluation.ranking import binary_ranking_metrics
from paclip.models.paclip import TextSideAnomalyModel
from paclip.evaluation.detection import detection_counts, best_dice_threshold
from paclip.evaluation.metrics import choose_pixel_threshold
from paclip.evaluation.postprocess import PostprocessConfig, refine_score_map


class E2Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(0)
        self.features = F.normalize(torch.randn(2, 3, 36, 8), dim=-1)
        self.roi = torch.ones(2, 6, 6, dtype=torch.bool)

    def build(self, **kwargs):
        checkpoint_sha256 = kwargs.pop("checkpoint_sha256", None)
        support_digest = kwargs.pop("support_digest", None)
        return NormalPatchMemory(MemoryConfig(prototypes_per_region=2, **kwargs)).build(
            self.features, self.roi, (6, 6), [0, 0], ["normal/a", "normal/b"],
            checkpoint_sha256=checkpoint_sha256, support_digest=support_digest)

    def calibrate(self, memory):
        distances = memory.distances(self.features, (6, 6))
        memory.calibrate(distances, torch.randn(2, 6, 6), self.roi, [0, 0], ["normal/v1", "normal/v2"])
        return distances

    def test_no_abnormal_or_heldout_bank_data(self):
        for labels, split in [([0, 1], "train"), ([0, 0], "test"), ([0, 0], "val")]:
            with self.assertRaises(ValueError):
                NormalPatchMemory().build(self.features, self.roi, (6, 6), labels, ["a", "b"], split)

    def test_regions_on_rectangular_grid(self):
        ids = patch_regions((6, 9)).reshape(6, 9)
        self.assertEqual(int(ids[0, 0]), 0)
        self.assertEqual(int(ids[-1, -1]), 8)
        self.assertEqual(set(ids.flatten().tolist()), set(range(9)))

    def test_model_retains_legacy_fusion_and_returns_normalized_layers(self):
        final = torch.randn(2, 7, 8)
        levels = [torch.randn(2, 6, 8) for _ in range(3)]
        class Trunk(torch.nn.Module):
            patch_embed = SimpleNamespace(grid_size=(2, 3))
            def forward_intermediates(self, *_args, **_kwargs):
                return final, levels
        model = TextSideAnomalyModel.__new__(TextSideAnomalyModel)
        torch.nn.Module.__init__(model)
        model.cfg = SimpleNamespace(ms_layers=[5, 8, 11], temperature=.07)
        visual = torch.nn.Module(); visual.trunk = Trunk(); visual.head = torch.nn.Linear(8, 4)
        model.clip = torch.nn.Module(); model.clip.visual = visual
        model.fusion_weights = torch.nn.Parameter(torch.zeros(3))
        legacy = F.normalize(torch.stack([visual.head(t) for t in levels]).mean(0), dim=-1)
        standard = model.encode_image(torch.zeros(2, 3, 32, 48))
        expanded = model.encode_image(torch.zeros(2, 3, 32, 48), return_levels=True)
        self.assertEqual(len(standard), 3)
        torch.testing.assert_close(expanded[1], legacy)
        torch.testing.assert_close(expanded[3].norm(dim=-1), torch.ones(2, 3, 6))
        anchors = {str(i): {"normal": F.normalize(torch.randn(4), dim=0),
                            "abnormal": F.normalize(torch.randn(4), dim=0)} for i in range(3)}
        outputs = model(torch.zeros(2, 3, 32, 48), anchors)
        torch.testing.assert_close(outputs["f_patch"], legacy)
        self.assertEqual(tuple(outputs["f_patch_levels"].shape), (2, 3, 6, 4))

    def test_froc_counts_missed_lesion_and_false_blob(self):
        gt = np.zeros((20, 20), dtype=bool)
        gt[1:3, 1:3] = gt[10:14, 10:14] = True
        score = np.zeros((20, 20))
        score[1:3, 1:3] = .8
        score[18, 18] = .9
        row = detection_counts(score, gt, np.ones_like(gt), [.5])[0]
        self.assertEqual(row["lesions"], 2)
        self.assertEqual(row["detected"], 1)
        self.assertEqual(row["false_positive_components"], 1)

    def test_volume_froc_uses_26_connectivity(self):
        gt = np.zeros((4, 20, 20), dtype=bool)
        gt[0:2, 1:3, 1:3] = True
        gt[2:4, 10:12, 10:12] = True
        score = np.zeros_like(gt, dtype=float)
        score[0:2, 1:3, 1:3] = .8
        score[0, 18, 18] = .9
        result = detection_counts(score, gt, np.ones_like(gt), [.5])[0]
        self.assertEqual(result["lesions"], 2)
        self.assertEqual(result["detected"], 1)
        self.assertEqual(result["false_positive_components"], 1)

    def test_prototype_only_changes_foreground_and_zero_preserves_margin(self):
        memory = self.build(fusion_mode="text_scale", roi_fallback=True)
        distances = self.calibrate(memory)
        text = torch.randn(2, 6, 6)
        roi = torch.zeros_like(text, dtype=torch.bool)
        roi[:, :2, :2] = True
        zero = memory.scores(distances, text, 0, roi=roi)
        one = memory.scores(distances, text, 1, roi=roi)
        torch.testing.assert_close(zero["fused_margin"], text, rtol=0, atol=0)
        torch.testing.assert_close(one["fused_margin"][~roi], text[~roi], rtol=0, atol=0)

    def test_vectorized_validation_threshold_matches_existing_rule(self):
        rng = np.random.default_rng(5)
        scores = np.round(rng.random((3, 8, 9)), 1)
        gt = rng.random((3, 8, 9)) > .8
        self.assertEqual(best_dice_threshold(scores, gt), choose_pixel_threshold(scores, gt))

    def test_global_fallback_is_opt_in(self):
        roi = torch.zeros_like(self.roi)
        roi[:, 0, 0] = True
        bank = NormalPatchMemory(MemoryConfig(prototypes_per_region=2, empty_region="global")).build(
            self.features, roi, (6, 6), [0, 0], ["a", "b"])
        self.assertTrue(any(row["fallback_global_normal_roi"] for row in bank.metadata["counts"]))
        self.assertTrue(all(len(p) <= 2 for level in bank.prototypes for p in level))
        self.assertTrue(bank.metadata["coverage"]["global_fallback_regions"])

    def test_empty_region_text_does_not_borrow(self):
        roi = torch.zeros_like(self.roi)
        roi[:, :2, :2] = True
        bank = NormalPatchMemory(MemoryConfig(prototypes_per_region=2, empty_region="text")).build(
            self.features, roi, (6, 6), [0, 0], ["a", "b"])
        self.assertTrue(bank.supported[0][0])
        self.assertFalse(any(bank.supported[0][1:]))
        self.assertEqual(len(bank.prototypes[0][1]), 0)
        distances = bank.distances(self.features, (6, 6))
        self.assertTrue(torch.isnan(distances[:, 0, 1, 2]).all())
        bank.calibrate(distances, torch.randn(2, 6, 6), roi, [0, 0], ["v1", "v2"])
        text = torch.randn(2, 6, 6)
        scores = bank.scores(distances, text, 1.0, roi=roi)
        regions = patch_regions((6, 6)).reshape(6, 6)
        outside = regions != 0
        torch.testing.assert_close(scores["fused_margin"][:, outside], text[:, outside])
        self.assertTrue(scores["support_mask"][:, regions == 0].all())
        self.assertFalse(scores["support_mask"][:, outside].any())

    def test_only_roi_features_enter_bank(self):
        roi = torch.zeros_like(self.roi)
        roi[:, 0, 0] = True
        first = NormalPatchMemory().build(self.features, roi, (6, 6), [0, 0], ["a", "b"])
        changed = self.features.clone()
        changed[:, :, 1:] = 900
        second = NormalPatchMemory().build(changed, roi, (6, 6), [0, 0], ["a", "b"])
        for a, b in zip(first.prototypes, second.prototypes):
            for x, y in zip(a, b):
                torch.testing.assert_close(x, y)

    def test_distance_formula_and_separate_layers(self):
        memory = self.build()
        d = memory.distances(self.features, (6, 6))
        p = memory.prototypes[0][0]
        expected = 1 - (self.features[0, 0, 0] @ p.T).max()
        torch.testing.assert_close(d[0, 0, 0, 0], expected)
        self.assertFalse(torch.equal(d[:, 0], d[:, 1]))
        self.assertTrue(((d >= 0) & (d <= 2)).all())

    def test_calibration_rejects_non_normal_and_support_reuse(self):
        memory = self.build()
        distances = memory.distances(self.features, (6, 6))
        for labels, ids, split in [([0, 1], ["v1", "v2"], "val"), ([0, 0], ["normal/a", "v2"], "val"), ([0, 0], ["v1", "v2"], "test")]:
            with self.assertRaises(ValueError):
                memory.calibrate(distances, torch.zeros(2, 6, 6), self.roi, labels, ids, split)

    def test_calibration_scale_floor_and_fusion_endpoints(self):
        memory = self.build(fusion_mode="text_scale")
        d = torch.ones(2, 3, 6, 6)
        text = torch.zeros(2, 6, 6)
        memory.calibrate(d, text, self.roi, [0, 0], ["v1", "v2"])
        self.assertTrue(all(v >= memory.config.scale_floor for v in memory.calibration["scale"]))
        self.assertEqual(memory.calibration["method"], "median_MAD_text_scale")
        zero, one = memory.scores(d, text, 0), memory.scores(d, text, 1)
        torch.testing.assert_close(zero["fused"], zero["text"])
        torch.testing.assert_close(one["fused"], one["prototype"])
        self.assertTrue(torch.isfinite(one["fused"]).all())

    def test_memory_save_load(self):
        memory = self.build()
        d = self.calibrate(memory)
        tmp = Path(tempfile.mkdtemp(prefix="_test_e2_", dir="."))
        try:
            path = tmp / "memory.pt"
            memory.save(path)
            loaded = NormalPatchMemory.load(path)
            torch.testing.assert_close(memory.distances(self.features, (6, 6)), loaded.distances(self.features, (6, 6)))
            torch.testing.assert_close(memory.scores(d, torch.zeros(2, 6, 6), .7)["fused"], loaded.scores(d, torch.zeros(2, 6, 6), .7)["fused"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_fusion_disabled_preserves_e1_and_no_double_sigmoid(self):
        memory = self.build()
        self.calibrate(memory)
        features = self.features
        class Fake(torch.nn.Module):
            cfg = SimpleNamespace(temperature=.07)
            def forward(self, images, anchors):
                return {"anomaly_map": torch.zeros(2, 6, 6), "f_patch_levels": features}
        model = Fake()
        wrapper = MemoryAugmentedModel(model, memory, enabled=False)
        images = torch.zeros(2, 3, 224, 224)
        torch.testing.assert_close(wrapper(images, None)["anomaly_map"], model(images, None)["anomaly_map"])
        wrapper.enabled = True
        result = wrapper(images, None)
        torch.testing.assert_close((result["anomaly_map"] / .07).sigmoid(), result["anomaly_score_e2"])
        wrapper.alpha = 0.
        torch.testing.assert_close(wrapper(images, None)["anomaly_map"], model(images, None)["anomaly_map"], rtol=0, atol=0)

    def test_exact_ranking_matches_sklearn_including_chunk_ties(self):
        rng = np.random.default_rng(2)
        for size in (2, 50, 2001):
            for tied in (False, True):
                scores = rng.random(size).astype(np.float32)
                if tied:
                    scores = np.round(scores, 1)
                truth = rng.random(size) < .3
                truth[0], truth[-1] = True, False
                result = binary_ranking_metrics(scores, truth, chunk_size=7)
                self.assertAlmostEqual(result["pixel_ap"], average_precision_score(truth, scores), places=12)
                self.assertAlmostEqual(result["pixel_auroc"], roc_auc_score(truth, scores), places=12)
        result = binary_ranking_metrics(np.ones(100), np.arange(100) < 10, chunk_size=3)
        self.assertAlmostEqual(result["pixel_ap"], .1)
        self.assertAlmostEqual(result["pixel_auroc"], .5)


    def test_image_roi_empty_policy_is_explicit(self):
        gray = torch.zeros(1, 224, 224)
        invalid = image_roi(gray, (14, 14), empty="invalid")
        legacy = image_roi(gray, (14, 14), empty="all")
        self.assertFalse(invalid.any())
        self.assertTrue(legacy.all())

    def test_refine_score_map_can_preserve_unbounded_doc_margin(self):
        score = np.asarray([[-2.0, .5], [3.0, -1.0]], dtype=np.float32)
        gray = np.zeros_like(score)
        config = PostprocessConfig(refine_method="none")
        clipped = refine_score_map(score, gray, config, clip=True)
        raw = refine_score_map(score, gray, config, clip=False)
        self.assertTrue(((clipped >= 0) & (clipped <= 1)).all())
        np.testing.assert_allclose(raw, score)

    def test_doc_mode_is_unscaled_and_has_no_roi_fallback(self):
        memory = self.build()
        distances = self.calibrate(memory)
        text = torch.randn(2, 6, 6)
        roi = torch.zeros_like(text, dtype=torch.bool)
        roi[:, :2, :2] = True
        result = memory.scores(distances, text, 1.0, roi=roi)
        calibration = memory.calibration
        self.assertEqual(calibration["method"], "doc_MAD")
        mean = torch.tensor(calibration["location"][:-1])
        scale = torch.tensor(calibration["scale"][:-1])
        expected = (((distances - mean[None, :, None, None]) / scale[None, :, None, None])
                    * memory.layer_weights[None, :, None, None]).sum(1)
        torch.testing.assert_close(result["fused_margin"], expected)
        self.assertFalse(torch.equal(result["fused_margin"][~roi], text[~roi]))

    def test_checkpoint_and_support_binding(self):
        memory = self.build(checkpoint_sha256="checkpoint-abc", support_digest="support-abc")
        tmp = Path(tempfile.mkdtemp(prefix="_test_e2_", dir="."))
        try:
            path = tmp / "memory.pt"
            memory.save(path)
            loaded = NormalPatchMemory.load(path, checkpoint_sha256="checkpoint-abc", support_digest="support-abc")
            self.assertEqual(loaded.checkpoint_sha256, "checkpoint-abc")
            with self.assertRaises(ValueError):
                NormalPatchMemory.load(path, checkpoint_sha256="checkpoint-other")
            with self.assertRaises(ValueError):
                NormalPatchMemory.load(path, support_digest="support-other")
            backbone = SimpleNamespace(cfg=SimpleNamespace(temperature=.07))
            with self.assertRaises(ValueError):
                MemoryAugmentedModel(backbone, loaded, checkpoint_sha256="checkpoint-other")
            with self.assertRaises(ValueError):
                MemoryAugmentedModel(backbone, loaded, support_digest="support-other")
            with self.assertRaises(ValueError):
                MemoryAugmentedModel(backbone, loaded, require_match=True)
            with self.assertRaises(ValueError):
                MemoryAugmentedModel(backbone, loaded, checkpoint_sha256="checkpoint-abc", require_match=True)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_sequence_calibration_groups(self):
        from paclip.cli.brain_memory import _sequence_key
        self.assertEqual(_sequence_key("FLAIR (matching BMAD Brain training sequence)"), "FLAIR")
        self.assertEqual(_sequence_key("T1, while checkpoint adapted on BMAD FLAIR"), "T1")
        memory = self.build(calibration_by="sequence")
        distances = memory.distances(self.features, (6, 6))
        text = torch.randn(2, 6, 6)
        memory.calibrate(distances, text, self.roi, [0, 0], ["v1", "v2"], groups=["FLAIR", "T1"])
        self.assertEqual(set(memory.calibration["groups"]), {"FLAIR", "T1"})
        self.assertEqual(memory.calibration["calibration_by"], "sequence")
        self.assertEqual(memory.calibration["default_group"], "FLAIR")
        memory.scores(distances, text, .5, groups=["T1", "UNKNOWN"])
        self.assertEqual(memory.last_group_fallback, 1)

    def test_froc_threshold_helper_is_denser_than_legacy(self):
        from paclip.cli.brain_memory import froc_thresholds
        values = np.linspace(0, 1, 1000)
        points = froc_thresholds(values, .5, 12)
        self.assertGreaterEqual(len(points), 12)
        self.assertTrue(all(np.isfinite(points)))


if __name__ == "__main__":
    unittest.main()
