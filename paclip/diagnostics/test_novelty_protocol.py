import unittest
import numpy as np
from paclip.evaluation.localization_budget import top_budget_mask
from paclip.evaluation.novelty_protocol import fit_operating_points, evaluate, lesion_counts, evaluate_volumes, check_condition, development_guard
from paclip.evaluation.branch_calibration import NormalCDF, calibrated_fusion


def items():
    a = np.arange(100, dtype=np.float64).reshape(10, 10) / 100
    gt = a > .8
    common = {"split": "validation", "roi": np.ones((10, 10), bool), "roi_valid": True,
              "dataset": "fixture", "sequence": "FLAIR", "orientation_verified": False, "slice_z": None,
              "disease_verified": True}
    return [{**common, "sample_id": "n", "patient_id": "n", "score": a, "gt": np.zeros_like(gt), "label": 0,
             "image_score": .5, "disease_labels": ["normal"]},
            {**common, "sample_id": "p", "patient_id": "p", "score": a + 1, "gt": gt, "label": 1,
             "image_score": 1.5, "disease_labels": ["lesion"]}]


class NoveltyProtocolTests(unittest.TestCase):
    def test_development_guard_refuses_recall_or_fp_tradeoff(self):
        reference = {"small_lesion_recall": .4, "normal_fp_area_ratio": .01}
        self.assertTrue(development_guard({"small_lesion_recall": .38, "normal_fp_area_ratio": .011}, reference)["passed"])
        self.assertFalse(development_guard({"small_lesion_recall": .37, "normal_fp_area_ratio": .001}, reference)["passed"])
        self.assertFalse(development_guard({"small_lesion_recall": .5, "normal_fp_area_ratio": .012}, reference)["passed"])
        self.assertFalse(development_guard({"small_lesion_recall": None, "normal_fp_area_ratio": .001}, reference)["passed"])

    def test_exact_top_budget_with_ties(self):
        pred = top_budget_mask(np.zeros((10, 10)), np.ones((10, 10)), .05)
        self.assertEqual(int(pred.sum()), 5)
        self.assertTrue(pred.ravel()[:5].all())

    def test_roi_does_not_delete_missed_lesion(self):
        gt = np.zeros((10, 10), bool); gt[5:7, 5:7] = True
        counts = lesion_counts(np.zeros_like(gt), gt)
        self.assertEqual(counts["lesions"], 1)
        self.assertEqual(counts["detected"], 0)

    def test_tied_normal_budget_respects_actual(self):
        rows = items(); rows[0]["score"][:] = 1
        thresholds = fit_operating_points(rows)
        self.assertLessEqual(thresholds["normal_budget_actual"], .01)
        self.assertGreater(thresholds["normal_budget_threshold"], 1)

    def test_test_cannot_fit_thresholds(self):
        rows = items(); rows[0]["split"] = "test"
        with self.assertRaises(ValueError):
            fit_operating_points(rows)

    def test_float32_budget_threshold_does_not_round_back_to_tie(self):
        rows = items()
        for r in rows:
            r["score"] = r["score"].astype(np.float32)
        rows[0]["score"][:] = 1
        points = fit_operating_points(rows)
        self.assertEqual(points["normal_budget_actual"], 0)
        self.assertFalse((rows[0]["score"] >= points["normal_budget_threshold"]).any())

    def test_full_metrics_and_oracle_separate(self):
        rows = items()
        points = fit_operating_points(rows)
        result = evaluate(rows, points)
        self.assertIn("small_lesion_recall", result["operating_points"]["dice"]["lesion_detection"])
        self.assertIn("0.05", result["top_budget"])
        self.assertFalse(result["oracle_diagnostic_only"]["used_for_selection"])
        self.assertEqual(result["volumetric"]["valid_volumes"], 0)
        self.assertTrue(any(r["false_positive_components"] == 0 for r in result["froc_2d"]))

    def test_volume_requires_complete_verified_order(self):
        rows = items()
        for i, r in enumerate(rows):
            r.update(patient_id="p", study_id="s", volume_id="v", orientation_verified=True, volume_complete=True, volume_depth=2, slice_z=i)
        self.assertEqual(evaluate_volumes(rows, [1])["valid_volumes"], 1)
        rows[1]["slice_z"] = 3
        self.assertEqual(evaluate_volumes(rows, [1])["valid_volumes"], 0)

    def test_unknown_disease_claim_rejected(self):
        source = items(); source[1]["split"] = "train"
        target = [{**source[1], "patient_id": "new", "sample_id": "new", "split": "test"}]
        with self.assertRaisesRegex(ValueError, "occurred"):
            check_condition(source, target, "same_sequence_unseen")
        target[0]["disease_labels"] = ["unknown"]
        with self.assertRaisesRegex(ValueError, "verified disease"):
            check_condition(source, target, "same_sequence_seen")

    def test_cdf_tails_finite_and_no_test_calibration(self):
        c = NormalCDF(np.arange(10), {"split": "validation", "normal_only": True, "sequence": "FLAIR"})
        z = c.transform([-100, 0, 4, 100])
        self.assertTrue(np.isfinite(z).all())
        self.assertTrue((np.diff(z) >= 0).all())
        self.assertTrue(np.allclose(calibrated_fusion({"dense": np.array([2, 3])}, {"dense": c}, {"dense": 1.}), c.transform([2, 3])))
        with self.assertRaises(ValueError):
            NormalCDF([1, 2], {"split": "test", "normal_only": True, "sequence": "T1"})
