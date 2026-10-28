"""E1 regression checks, no model downloads: python -m unittest paclip.diagnostics.test_e1 -v."""

import tempfile
import unittest
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from paclip.data.datasets import SliceAnomalyDataset, VolumeAnomalyDataset, VolumeFixedSliceDataset, load_binary_mask
from paclip.data.sampling import LesionBalancedBatchSampler, add_adjacent_volume_negatives, refresh_hard_negatives
from paclip.config import Config
from paclip.data.supervision import collate_anomaly_samples, pool_mask_targets, training_masks
from paclip.data.thymoma import ThymomaSliceDataset
from paclip.engine.trainer import _run_steps, _predict, make_args, run_experiment
from paclip.evaluation.metrics import mask_overlap_metrics, pixel_metrics, pixel_ranking_metrics
from paclip.losses import TotalLoss, local_alignment_loss, focal_bce, soft_dice_loss, boundary_loss


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.1))

    def encode_anchors(self, _):
        return {str(i): {"normal": torch.tensor([1., 0.]), "abnormal": torch.tensor([0., 1.])} for i in range(2)}

    def forward(self, images, _):
        logits = self.weight * torch.ones(len(images), 2, 3, 5)
        logits = logits * torch.tensor([0., 1.])[None, :, None, None]
        cls = logits.mean((-2, -1))
        return {"patch_logits": logits, "cls_logits": cls, "cls_probs": cls.softmax(1)[:, 1],
                "anomaly_map": logits[:, 1] - logits[:, 0]}


class E1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_single_pixel_survives_and_has_correct_occupancy(self):
        mask = torch.zeros(224, 224)
        mask[5, 7] = 1
        soft, hard = pool_mask_targets([mask], (14, 14))
        self.assertEqual(float(hard.sum()), 1)
        self.assertEqual(float(soft[0, 0, 0]), 1 / 256)
        self.assertEqual(float((soft > 0.5).sum()), 0)  # old threshold loses it

    def test_variable_native_masks_rectangular_output(self):
        masks = [torch.zeros(35, 77), torch.zeros(96, 80)]
        masks[0][0, 0] = masks[1][-1, -1] = 1
        soft, hard = pool_mask_targets(masks, (7, 11))
        self.assertEqual(tuple(soft.shape), (2, 7, 11))
        self.assertTrue((hard.sum((1, 2)) >= 1).all())

    def test_high_resolution_evaluation_retains_tiny_lesion(self):
        native = np.zeros((512, 512), dtype=np.uint8)
        native[7, 9] = 1
        resized = load_binary_mask(None, 224, 224, plane=native)
        self.assertTrue(resized.any())

    def test_slice_loader_binary_one_and_255_and_collation(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = []
            for i, (shape, value) in enumerate([((32, 48), 1), ((64, 80), 255)]):
                image, mask = Path(tmp) / f"i{i}.png", Path(tmp) / f"m{i}.png"
                Image.fromarray(np.full(shape, 100, dtype=np.uint8)).save(image)
                native = np.zeros(shape, dtype=np.uint8)
                native[1, 1] = value
                Image.fromarray(native).save(mask)
                records.append(dict(path=str(image), mask_path=str(mask), label=1, sample_id=str(i), mask_found=True))
            ds = SliceAnomalyDataset(tmp, records=records)
            batch = next(iter(DataLoader(ds, batch_size=2, collate_fn=collate_anomaly_samples)))
            self.assertEqual([tuple(m.shape) for m in batch["mask_high"]], [(32, 48), (64, 80)])
            self.assertTrue((batch["mask_hard"].sum((1, 2)) >= 1).all())
            self.assertTrue((batch["mask_soft"].sum((1, 2)) > 0).all())
            model = TinyModel()
            before = model.weight.detach().clone()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            seen = []
            _run_steps(model, [batch], optimizer, TotalLoss(), None, "cpu", 2, None, True, seen, 2, 0)
            self.assertFalse(torch.equal(before, model.weight))
            _, _, maps, masks, _ = _predict(model, [batch], None, "cpu")
            self.assertEqual(maps.shape, masks.shape)
            self.assertEqual(maps.shape, (2, 3, 5))
            self.assertTrue((masks.sum((1, 2)) >= 1).all())

    def test_volume_native_mask_adjacent_labels_and_support_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            volume = np.random.default_rng(0).random((32, 40, 7)).astype(np.float32)
            mask = np.zeros_like(volume)
            mask[3, 4, 2:5] = 1
            path, mask_path = Path(tmp) / "image.nii.gz", Path(tmp) / "mask.nii.gz"
            nib.save(nib.Nifti1Image(volume, np.eye(4)), path)
            nib.save(nib.Nifti1Image(mask, np.eye(4)), mask_path)
            records = [dict(path=str(path), mask_path=str(mask_path), label=1, sample_id="selected", mask_found=True)]
            base = VolumeAnomalyDataset(tmp, records=records)
            ds = add_adjacent_volume_negatives(VolumeFixedSliceDataset(base, [(0, 2), (0, 3)]))
            self.assertEqual(set(ds.pairs), {(0, 2), (0, 3), (0, 4), (0, 1), (0, 5)})
            self.assertEqual([int(ds[i]["label"]) for i in range(len(ds))], [1, 1, 1, 0, 0])
            self.assertTrue(ds[0]["lesion_boundary"])
            sampler = LesionBalancedBatchSampler(ds, 4)
            self.assertEqual(sampler.summary["positive_slices"], 3)
            self.assertEqual(sampler.summary["negative_slices"], 2)
            self.assertEqual(sampler.summary["boundary_slices"], 2)
            self.assertEqual(sampler.summary["boundary_priority_status"], "active")

    def test_focal_dice_boundary_formula_and_gradients(self):
        torch.manual_seed(0)
        logits = torch.randn(2, 2, 6, 8, requires_grad=True)
        native = [torch.zeros(24, 32), torch.ones(24, 32)]
        native[0][3, 5] = 1
        target, _ = pool_mask_targets(native, (6, 8))
        anomaly = logits[:, 1] - logits[:, 0]
        expected = focal_bce(anomaly, target) + soft_dice_loss(anomaly.sigmoid(), target) + .1 * boundary_loss(anomaly.sigmoid(), target)
        loss = local_alignment_loss(logits, native, recall_weight=0)
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()), 0)

    def test_correct_localization_reduces_loss(self):
        target = torch.zeros(1, 8, 8)
        target[:, :4, :4] = 1
        anomaly = (target * 2 - 1) * 8
        correct = torch.stack([torch.zeros_like(target), anomaly], dim=1)
        self.assertLess(local_alignment_loss(correct, target), local_alignment_loss(-correct, target))

    def test_extreme_logits_empty_and_full_targets_remain_finite(self):
        for target_value in (0., 1.):
            for score in (-1000., 1000.):
                logits = torch.tensor([[[[0.]], [[score]]]], requires_grad=True)
                loss = local_alignment_loss(logits, torch.full((1, 9, 13), target_value))
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(torch.isfinite(logits.grad).all())

    def test_empty_roi_has_zero_loss_and_gradient(self):
        logits = torch.randn(2, 2, 3, 5, requires_grad=True)
        loss = local_alignment_loss(logits, torch.ones(2, 9, 15), torch.zeros(2, 9, 15))
        loss.backward()
        self.assertEqual(float(loss.detach()), 0)
        self.assertEqual(float(logits.grad.abs().sum()), 0)

    def test_hard_recall_constraint_enabled_by_default_and_has_gradient(self):
        logits = torch.zeros(1, 2, 7, 11, requires_grad=True)
        mask = torch.zeros(1, 70, 110)
        mask[0, 2, 2] = 1
        base = local_alignment_loss(logits, mask, recall_weight=0)
        constrained = local_alignment_loss(logits, mask)
        torch.testing.assert_close(constrained - base, torch.tensor(.05))
        grad = torch.autograd.grad(constrained - base, logits)[0]
        self.assertLess(float(grad[0, 1, 0, 0]), 0)
        self.assertEqual(float(grad[0, 1, 1:, 1:].abs().sum()), 0)
        self.assertEqual(Config().recall_weight, .1)
        self.assertEqual(TotalLoss().local_options["recall_weight"], .1)

    def test_per_level_loss_uses_native_targets(self):
        model = TinyModel()
        out = model(torch.zeros(2, 3, 224, 224), None)
        out["patch_logits_per_level"] = out["patch_logits"][:, None].expand(-1, 3, -1, -1, -1)
        losses = TotalLoss(w_level=.5)(model.encode_anchors(None), out, torch.tensor([0, 1]), [torch.zeros(17, 23), torch.ones(19, 29)])
        torch.testing.assert_close(losses["level"], losses["local"])
        losses["total"].backward()
        self.assertTrue(torch.isfinite(model.weight.grad))

    def _sampling_items(self):
        items = []
        for i in range(8):
            mask = torch.zeros(16, 16)
            if i < 4:
                mask[:i + 1, :i + 1] = 1
            items.append({"image": torch.full((3, 32, 32), float(i + 1)), "mask": mask,
                          "mask_high": mask, "label": torch.tensor(i < 4)})
        return items

    def test_balanced_sampling_reproducible_and_small_lesion_priority(self):
        items = self._sampling_items()
        a, b = LesionBalancedBatchSampler(items, 8, 7), LesionBalancedBatchSampler(items, 8, 7)
        self.assertEqual(list(a), list(b))
        self.assertGreater(a.positive_weights[0], a.positive_weights[-1])
        for batch in a:
            self.assertEqual(sum(i in a.positive for i in batch), 4)
            self.assertEqual(sum(i in a.normal for i in batch), 2)
            self.assertEqual(sum(i in a.hard for i in batch), 2)

    def test_single_class_sampling_reports_fallback(self):
        sampler = LesionBalancedBatchSampler(self._sampling_items()[:4], 4)
        self.assertIsNotNone(sampler.summary["fallback"])
        self.assertEqual(len(next(iter(sampler))), 4)

    def test_model_mining_ranks_only_selected_negatives_and_restores_mode(self):
        class ScoringModel(TinyModel):
            def forward(self, images, anchors):
                score = self.weight * images.mean((1, 2, 3))
                zero = torch.zeros(len(images), 2, 3, 5)
                zero[:, 1] = score[:, None, None]
                return {"patch_logits": zero}

        items = self._sampling_items()
        sampler = LesionBalancedBatchSampler(items, 8)
        model = ScoringModel().train()
        refresh_hard_negatives(model, sampler, None, "cpu", 1)
        self.assertTrue(model.training)
        self.assertIsNone(model.weight.grad)
        self.assertEqual(set(sampler.hard), {6, 7})
        self.assertEqual(set(sampler.summary["hard_mining_history"][0]["scores"]), {"4", "5", "6", "7"})
        with torch.no_grad():
            model.weight.neg_()
        refresh_hard_negatives(model, sampler, None, "cpu", 21)
        self.assertEqual(set(sampler.hard), {4, 5})
        self.assertEqual(len(sampler.summary["hard_mining_history"]), 2)
        with self.assertRaises(ValueError):
            sampler.update_hard_negatives({0: .9}, 22)

    def test_singleton_batches_alternate_classes(self):
        sampler = LesionBalancedBatchSampler(self._sampling_items(), 1)
        classes = [int(batch[0] < 4) for batch in sampler]
        self.assertEqual(classes, [1, 0, 1, 0, 1, 0, 1, 0])

    def test_experiment_records_e1_and_k5_forms_full_batch(self):
        class ExperimentModel(TinyModel):
            def __init__(self, cfg):
                super().__init__()
                self.cfg = cfg

            def build_optimizer(self):
                return torch.optim.SGD(self.parameters(), lr=.01)

            def save_checkpoint(self, path, optimizer=None, **extra):
                torch.save({"state_dict": self.state_dict(), **extra}, path)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for label, count in [(0, 2), (1, 3)]:
                folder = "abnormal" if label else "normal"
                (root / "train" / folder).mkdir(parents=True)
                (root / "masks" / folder).mkdir(parents=True)
                for index in range(count):
                    image = np.full((64, 80), 80 + index * 10, dtype=np.uint8)
                    Image.fromarray(image).save(root / "train" / folder / f"{index}.png")
                    if label:
                        mask = np.zeros_like(image)
                        mask[7, 9] = 255
                        Image.fromarray(mask).save(root / "masks" / folder / f"{index}.png")
            args = make_args(str(root / "train"), mask_root=str(root / "masks"), n_support=5,
                             steps=1, batch_size=8, device="cpu", no_heatmaps=True, save_dir=str(root / "runs"),
                             raw_image_threshold=.5, raw_pixel_threshold=0.,
                             post_image_threshold=.5, post_pixel_threshold=.5)
            result = run_experiment(args, model_factory=lambda cfg, _text, _visual: ExperimentModel(cfg))
            self.assertEqual(result["status"], "evaluated")
            self.assertEqual(len(result["seen_sample_ids"]), 8)
            run_dir = Path(result["run_dir"])
            config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["effective_batch_size"], 8)
            self.assertIn("E1", config["local_supervision"]["version"])
            self.assertEqual(config["local_supervision"]["recall_weight"], .1)
            self.assertEqual(config["support_set"]["actual_support"], 5)
            blob = torch.load(result["checkpoint"], weights_only=False)
            self.assertIn("E1", blob["local_supervision"]["version"])
            self.assertEqual(blob["training_sampling"]["positive_slices"], 3)
            self.assertEqual(blob["training_sampling"]["hard_mining_status"], "model_refreshed")
            self.assertEqual(blob["training_sampling"]["boundary_priority_status"], "unavailable_no_verified_slice_context")
            self.assertEqual(blob["training_sampling"], json.loads((run_dir / "training_sampling.json").read_text(encoding="utf-8")))
            self.assertTrue(set(result["seen_sample_ids"]).issubset(set(result["support_ids"])))

    def test_cache_prefers_original_mask(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "slice.npz"
            native = np.zeros((48, 64), dtype=np.uint8)
            native[1, 1] = 1
            np.savez(path, image=np.zeros((224, 224)), mask=np.zeros((14, 14)), mask_high=native)
            sample = ThymomaSliceDataset(tmp)[0]
            self.assertEqual(float(sample["mask_high"].sum()), 1)
            self.assertGreater(float(sample["mask_soft"].sum()), 0)

    def test_acceptance_metrics_and_normal_false_positives(self):
        gt = np.zeros((2, 20, 20), dtype=bool)
        gt[0, 1:3, 1:3] = True  # small: 1% of image
        gt[0, 10:14, 10:14] = True
        pred = np.zeros_like(gt)
        pred[0, 1:3, 1:3] = True
        pred[1, 0, :2] = True
        result = mask_overlap_metrics(pred, gt, [True, True])
        self.assertEqual(result["lesion_recall"], .5)
        self.assertEqual(result["small_lesion_recall"], 1)
        self.assertEqual(result["normal_false_positive_area_pixels"], 2)
        self.assertEqual(result["normal_false_positive_area_ratio"], 2 / 400)
        ranking = pixel_ranking_metrics(gt.astype(float), gt, [True, True])
        self.assertEqual(ranking["pixel_ap"], 1)
        metrics = pixel_metrics(gt.astype(float), gt, threshold=.5)
        self.assertEqual(metrics["pixel_ap"], 1)
        self.assertEqual(metrics["lesion_recall"], 1)
        all_normal = pixel_metrics(np.zeros((1, 20, 20)), np.zeros((1, 20, 20)), threshold=.5)
        self.assertEqual(all_normal["normal_false_positive_area_ratio"], 0)


if __name__ == "__main__":
    unittest.main()
