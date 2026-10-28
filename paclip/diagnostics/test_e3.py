"""No downloads: python -m unittest paclip.diagnostics.test_e3 -v."""

import shutil
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from types import SimpleNamespace

from paclip.cli.brain_structure import (fuse_context_scores, memory_fingerprint,
                                         output_identity, select_context_candidate)
from paclip.data.slice_context import SliceIndex, SliceMeta, build_slice_index
from paclip.engine.consistency import smooth_three_slice
from paclip.models.normal_memory import MemoryConfig, NormalPatchMemory
from paclip.models.structural_context import ContextConfig, StructuralContext


def meta(case, z, index, orientation="RAS", spacing=1.0, brain=True):
    return SliceMeta(sample_id=f"{case}/z{z}", case_id=case, series_id=case,
                     sequence="FLAIR", slice_z=z, split="test",
                     spacing_z=spacing, orientation=orientation, brain_valid=brain)


class E3Tests(unittest.TestCase):
    def test_source_png_index_is_explicitly_unavailable(self):
        class Source:
            records = [
                {"sample_id": "normal/a.png", "case_id": "", "slice_z": -1},
                {"sample_id": "normal/b.png", "case_id": "", "slice_z": -1},
            ]
        index = build_slice_index(Source(), "source_val")
        self.assertFalse(index.available)
        self.assertTrue(all(prev is None and nxt is None for prev, nxt in index.neighbors.values()))
        self.assertTrue(all("source_png_slice_z_unavailable" in reason for reason in index.reasons.values()))

    def test_external_neighbors_do_not_cross_cases(self):
        entries = {}
        for case in ("A", "B"):
            for z in range(3):
                entries[len(entries)] = meta(case, z, len(entries))
        index = SliceIndex.from_meta(entries, dataset="external")
        self.assertTrue(index.available)
        self.assertEqual(index.neighbor(1), (0, 2))
        self.assertEqual(index.neighbor(4), (3, 5))
        self.assertIsNone(index.neighbor(0)[0])
        self.assertIsNone(index.neighbor(2)[1])
        self.assertNotIn(3, index.neighbor(1))

    def test_neighbor_requires_same_orientation_and_spacing(self):
        entries = {0: meta("A", 0, 0), 1: meta("A", 1, 1, orientation="LAS"),
                   2: meta("A", 2, 2, spacing=3.0)}
        index = SliceIndex.from_meta(entries, dataset="external")
        self.assertFalse(index.available)
        self.assertTrue(any("orientation" in issue or "spacing" in issue for issue in index.reasons[1]))

    def test_context_errors_and_calibration(self):
        torch.manual_seed(0)
        levels = F.normalize(torch.randn(2, 3, 36, 8), dim=-1)
        roi = torch.ones(2, 6, 6, dtype=torch.bool)
        context = StructuralContext(ContextConfig(), layer_ids=(5, 8, 11))
        errors, valid = context.compute_errors(levels, roi, (6, 6))
        self.assertEqual(tuple(errors.shape), (2, 3, 6, 6))
        self.assertTrue(valid.all())
        self.assertTrue(((errors >= 0) & (errors <= 2)).all())
        context.calibrate(errors, valid, ["FLAIR", "FLAIR"], ["v1", "v2"])
        scores, context_valid = context.scores(errors, valid, ["FLAIR", "FLAIR"])
        self.assertEqual(tuple(scores.shape), (2, 6, 6))
        self.assertTrue(context_valid.all())
        self.assertTrue(((scores >= 0) & (scores <= 1)).all())
        self.assertEqual(context.calibration["status"], "single_group")

    def test_context_calibration_rejects_non_validation(self):
        context = StructuralContext(ContextConfig())
        errors = torch.zeros(1, 3, 4, 4)
        valid = torch.ones(1, 3, 4, 4, dtype=torch.bool)
        with self.assertRaises(ValueError):
            context.calibrate(errors, valid, ["FLAIR"], ["v1"], split="test")

    def test_exclude_center_changes_context(self):
        torch.manual_seed(1)
        levels = F.normalize(torch.randn(2, 3, 36, 8), dim=-1)
        roi = torch.ones(2, 6, 6, dtype=torch.bool)
        include = StructuralContext(ContextConfig(exclude_center=False))
        exclude = StructuralContext(ContextConfig(exclude_center=True))
        e1, v1 = include.compute_errors(levels, roi, (6, 6))
        e2, v2 = exclude.compute_errors(levels, roi, (6, 6))
        self.assertTrue(v1.any() and v2.any())
        self.assertFalse(torch.equal(e1, e2))

    def test_consistency_weights_and_endpoint_fallback(self):
        entries = {i: meta("A", i, i) for i in range(5)}
        index = SliceIndex.from_meta(entries, dataset="external")
        scores = np.zeros((5, 2, 2), dtype=np.float32)
        scores[2] = 1.0
        smoothed, summary = smooth_three_slice(scores, index)
        self.assertEqual(smoothed[0].tolist(), scores[0].tolist())
        self.assertEqual(smoothed[4].tolist(), scores[4].tolist())
        self.assertAlmostEqual(float(smoothed[2].mean()), .88, places=6)
        self.assertEqual(summary["n_smoothed"], 3)
        self.assertEqual(summary["n_fallback"], 2)
        self.assertAlmostEqual(summary["merged_neighbor_weight"], .06, places=6)

    def test_beta_zero_is_exact_base_and_context_is_roi_limited(self):
        base = np.asarray([[.1, .2], [.3, .4]], dtype=np.float32)
        context = np.asarray([[.9, .8], [.7, .6]], dtype=np.float32)
        valid = np.asarray([[True, False], [True, False]])
        self.assertTrue(np.array_equal(fuse_context_scores(base, context, valid, 0.0), base))
        fused = fuse_context_scores(base, context, valid, .1)
        self.assertAlmostEqual(float(fused[0, 0]), .9 * .1 + .1 * .9, places=6)
        self.assertAlmostEqual(float(fused[0, 1]), .2, places=6)
        self.assertAlmostEqual(float(fused[1, 0]), .9 * .3 + .1 * .7, places=6)

    def test_missing_normal_fp_is_not_treated_as_zero(self):
        baseline = {"beta": 0.0, "metrics": {"small_lesion_recall": .5,
                                             "normal_slice_false_positive_components": 0.0,
                                             "pixel_ap": .7, "dice_micro": .6}}
        candidate = {"beta": .1, "metrics": {"small_lesion_recall": .5,
                                             "normal_slice_false_positive_components": 1.0,
                                             "pixel_ap": .9, "dice_micro": .8}}
        missing = {"beta": .2, "metrics": {"small_lesion_recall": .5,
                                           "pixel_ap": .99, "dice_micro": .9}}
        non_finite = {"beta": .3, "metrics": {"small_lesion_recall": .5,
                                              "normal_slice_false_positive_components": float("nan"),
                                              "pixel_ap": .99, "dice_micro": .9}}
        self.assertIs(select_context_candidate([baseline, candidate, missing, non_finite]), baseline)
        bad_baseline = {"beta": 0.0, "metrics": {"small_lesion_recall": float("nan"),
                                                 "normal_slice_false_positive_components": 0.0,
                                                 "pixel_ap": .7, "dice_micro": .6}}
        with self.assertRaises(ValueError):
            select_context_candidate([bad_baseline])

    def test_memory_fingerprint_binds_prototypes(self):
        torch.manual_seed(0)
        features = F.normalize(torch.randn(2, 3, 36, 8), dim=-1)
        roi = torch.ones(2, 6, 6, dtype=torch.bool)
        first = NormalPatchMemory(MemoryConfig(prototypes_per_region=2, seed=0)).build(
            features, roi, (6, 6), [0, 0], ["a", "b"])
        second = NormalPatchMemory(MemoryConfig(prototypes_per_region=2, seed=1)).build(
            features, roi, (6, 6), [0, 0], ["a", "b"])
        self.assertNotEqual(memory_fingerprint(first), memory_fingerprint(second))
        changed = memory_fingerprint(second)
        second.prototypes[0][0][0] = second.prototypes[0][0][0] + 1
        self.assertNotEqual(changed, memory_fingerprint(second))

    def test_exclude_center_corner_has_no_valid_neighbor(self):
        torch.manual_seed(0)
        levels = F.normalize(torch.randn(1, 3, 36, 8), dim=-1)
        roi = torch.zeros(1, 6, 6, dtype=torch.bool)
        roi[:, 0, 0] = True
        context = StructuralContext(ContextConfig(exclude_center=True))
        _, valid = context.compute_errors(levels, roi, (6, 6))
        self.assertFalse(valid.any())
        include = StructuralContext(ContextConfig(exclude_center=False))
        _, include_valid = include.compute_errors(levels, roi, (6, 6))
        self.assertTrue(include_valid[:, :, 0, 0].all())

    def test_external_index_uses_canonical_affine(self):
        tmp = Path(tempfile.mkdtemp(prefix="_test_e3_", dir="."))
        try:
            data = np.zeros((4, 4, 3), dtype=np.int16)
            data[1:3, 1:3, :] = 1
            affine = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -2, 0], [0, 0, 0, 1]], dtype=float)
            image_path = tmp / "image.nii.gz"
            mask_path = tmp / "mask.nii.gz"
            nib.Nifti1Image(data, affine).to_filename(str(image_path))
            nib.Nifti1Image((data > 0).astype(np.uint8), affine).to_filename(str(mask_path))
            protocol = {"cases": [{"case_id": "c", "image": str(image_path),
                                   "brain_mask": str(mask_path), "lesion_mask": str(mask_path)}]}
            class External:
                pairs = [(0, z) for z in range(3)]
                pass
            dataset = External()
            dataset.protocol = protocol
            index = build_slice_index(dataset, "test")
            canonical = nib.as_closest_canonical(nib.load(str(image_path)))
            expected_orientation = "".join(nib.aff2axcodes(canonical.affine))
            expected_spacing = float(nib.affines.voxel_sizes(canonical.affine)[2])
            self.assertEqual(index.meta[0].orientation, expected_orientation)
            self.assertAlmostEqual(index.meta[0].spacing_z, expected_spacing, places=6)
            self.assertTrue(index.available)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


    def test_output_identity_binds_weights_and_fusion_parameters(self):
        args = SimpleNamespace(base="e1", context="on", consistency="off",
                               checkpoint="checkpoint.pt", support_manifest="support.json",
                               data_root="data", mask_root="masks", source_sequence="FLAIR",
                               alpha=1.0, froc_points=20, e2_memory=None,
                               unseen_protocols=[], require_checkpoint_match=True, cache_root=None)
        first = output_identity(args, "C1", "sha", "support", [0.0, 0.1], ContextConfig())
        args.alpha = 0.9
        second = output_identity(args, "C1", "sha", "support", [0.0, 0.1], ContextConfig())
        self.assertNotEqual(first, second)
        args.alpha = 1.0
        args.context_kernel = 5
        third = output_identity(args, "C1", "sha", "support", [0.0, 0.1], ContextConfig(kernel=5))
        self.assertNotEqual(first, third)


if __name__ == "__main__":
    unittest.main()
