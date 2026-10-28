import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from paclip.data.synthetic_image import synthesize_image
from paclip.data.synthetic_feature import synthesize_feature
from paclip.data.mixed_sampling import MixedSourceSampler, MODES
from paclip.data.patient_manifest import audit_rows, choose_support, resolve_support
from paclip.data.novelty_dataset import NoveltyDataset
from paclip.engine.novelty_trainer import NoveltyTrainer
from paclip.engine.novelty_cache import CleanFeatureCache
from paclip.losses_novelty import grouped_novelty_loss, lesion_rank_loss
from paclip.diagnostics.novelty_fixtures import fixture_rows, TinyEncoder


class SyntheticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_image_roi_support_and_repeatability(self):
        x = torch.ones(224, 224) * .5
        roi = torch.zeros_like(x, dtype=torch.bool); roi[20:200, 20:200] = True
        a = synthesize_image(x, roi, 7); b = synthesize_image(x, roi, 7)
        self.assertTrue(torch.equal(a["gray"], b["gray"]))
        self.assertTrue(torch.equal(a["gray"][~roi], x[~roi]))
        hard = a["mask_high"].bool()
        self.assertEqual(float(a["blend_mask"][~hard].sum()), 0)
        self.assertTrue(torch.equal(a["gray"][~hard], x[~hard]))
        self.assertGreaterEqual(float(hard.float().mean()), .001)
        self.assertLessEqual(float(hard.float().mean()), .10)
        self.assertFalse(torch.equal(a["gray"], synthesize_image(x, roi, 8)["gray"]))

    def test_empty_or_unverified_roi_fails(self):
        with self.assertRaises(ValueError):
            synthesize_image(torch.ones(224, 224), torch.zeros(224, 224), 0)
        with self.assertRaises(ValueError):
            synthesize_feature(torch.ones(3, 196, 512), (14, 14), torch.ones(224, 224), 0, roi_valid=False)

    def test_feature_relative_norm_shared_mask_detached(self):
        base = F.normalize(torch.randn(3, 196, 512), dim=-1).requires_grad_()
        synth = synthesize_feature(base, (14, 14), torch.ones(224, 224), 6)
        active = synth["mask_high"].flatten().bool()
        delta = synth["levels"] - base.detach()
        self.assertFalse(synth["levels"].requires_grad)
        self.assertEqual(float(delta[:, ~active].abs().sum()), 0)
        self.assertTrue(torch.allclose(delta[:, active].norm(dim=-1), torch.full((3, int(active.sum())), synth["parameters"]["rho_relative_l2"]), atol=1e-6))
        self.assertLessEqual(int(active.sum()), 20)
        self.assertTrue(torch.equal(synth["levels"], synthesize_feature(base, (14, 14), torch.ones(224, 224), 6)["levels"]))

    def test_all_quotas_dynamic_sources(self):
        support = [{"sample_id": str(i), "label": int(i >= 2), "split": "train"} for i in range(5)]
        for mode, expected in MODES.items():
            sampler = MixedSourceSampler(support, mode)
            first = sampler.batch(1)
            actual = [sum(r["source_kind"] == "normal" for r in first), sum(r["source_kind"] == "real" for r in first),
                      sum(r["synthetic_type"] == "image" for r in first), sum(r["synthetic_type"] == "feature" for r in first)]
            self.assertEqual(tuple(actual), expected)
            self.assertEqual(first, sampler.batch(1))
            self.assertNotEqual(first, sampler.batch(2))
            self.assertTrue(all(r["support_index"] < 2 for r in first if r["source_kind"] == "synthetic"))
        with self.assertRaises(ValueError):
            MixedSourceSampler(support, "joint_synthetic", batch_size=4)

    def test_loss_group_normalization_not_sample_count(self):
        score = torch.randn(2, 1, 4, 5, requires_grad=True)
        masks = [torch.zeros(30, 40), torch.zeros(30, 40)]; masks[1][2, 2] = 1
        roi = torch.ones(2, 20, 25)
        a = grouped_novelty_loss(score, masks, roi, ["normal", "real"], ["none", "none"])
        b = grouped_novelty_loss(torch.cat([score[:1], score[:1], score[1:]]), [masks[0], *masks],
            torch.ones(3, 20, 25), ["normal", "normal", "real"], ["none"] * 3)
        self.assertTrue(torch.allclose(a["total"], b["total"]))
        a["total"].backward()
        self.assertTrue(torch.isfinite(score.grad).all())

    def test_rank_empty_no_background_and_tiny_lesion(self):
        x = torch.zeros(4, 5, requires_grad=True)
        for mask in (torch.zeros(30, 40), torch.ones(30, 40)):
            loss, stats = lesion_rank_loss(x, mask, torch.ones(20, 25))
            self.assertTrue(torch.isfinite(loss))
        mask = torch.zeros(30, 40); mask[1, 1] = 1
        loss, stats = lesion_rank_loss(x, mask, torch.ones(20, 25))
        self.assertEqual(stats["valid_pairs"], 1)
        loss.backward()
        self.assertGreater(float(x.grad.abs().sum()), 0)

    def test_cache_identity_and_clean_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = fixture_rows(tmp)
            audit_rows(rows)
            sample = NoveltyDataset(rows)[0]
            cache = CleanFeatureCache(Path(tmp) / "cache", "baseA")
            levels = F.normalize(torch.randn(3, 196, 512), dim=-1)
            cache.put(sample, levels, (14, 14))
            self.assertTrue(torch.equal(cache.get(sample)[0], levels))
            changed = CleanFeatureCache(Path(tmp) / "cache", "baseB")
            self.assertIsNone(changed.get(sample))
            self.assertIsNone(cache.get({**sample, "image_sha256": "changed"}))

    def test_resume_exact_training_and_complete_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = fixture_rows(root)
            support = choose_support(rows)
            selected = resolve_support(rows, support)
            cfg = {"mode": "joint_synthetic", "steps": 3, "validate_every": 3, "save_every": 1,
                   "preview_every": 3, "lambda_rank": .2}
            identity = {"base_sha256": "fixture", "split_digest": support["split_digest"], "support_digest": support["support_digest"]}
            a = NoveltyTrainer(cfg, rows, selected, identity, root / "uninterrupted", TinyEncoder)
            a.run()
            b = NoveltyTrainer(cfg, rows, selected, identity, root / "resumed", TinyEncoder)
            b.run(stop_after=1)
            b = NoveltyTrainer(cfg, rows, selected, identity, root / "resumed", TinyEncoder)
            b.run(resume=True)
            last_a = torch.load(a.output / "checkpoint_last.pt", weights_only=False)
            last_b = torch.load(b.output / "checkpoint_last.pt", weights_only=False)
            self.assertEqual(last_a["counts"], {"normal": 12, "real": 6, "syn_image": 3, "syn_feature": 3})
            self.assertEqual(last_a["counts"], last_b["counts"])
            for k in last_a["head"]:
                self.assertTrue(torch.equal(last_a["head"][k], last_b["head"][k]), k)
            import json
            results_a = json.loads((a.output / "results.json").read_text(encoding="utf-8"))
            results_b = json.loads((b.output / "results.json").read_text(encoding="utf-8"))
            self.assertEqual(results_a, results_b)

    def test_frozen_cache_hit_does_not_load_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = fixture_rows(tmp)
            support = choose_support(rows)
            selected = resolve_support(rows, support)
            cache = CleanFeatureCache(Path(tmp) / "cache", "fixture")
            for sample in NoveltyDataset(selected):
                cache.put(sample, F.normalize(torch.randn(3, 196, 512), dim=-1), (14, 14))
            def forbidden():
                raise AssertionError("cache-only path loaded encoder")
            trainer = NoveltyTrainer({"mode": "feature_synthetic", "steps": 1}, rows, selected, {}, Path(tmp) / "run", forbidden, cache=cache)
            trainer.output.mkdir()
            logged = trainer.train_step()
            for field in ("loss_syn", "loss_rank", "real_lesion_recall", "synthetic_lesion_recall", "normal_false_positive_area"):
                self.assertIn(field, logged)
            self.assertIsNone(trainer.model.encoder)

    def test_visual_training_cannot_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "cache"):
                NoveltyTrainer({"train_visual": True}, [], [], {}, tmp, TinyEncoder,
                               cache=CleanFeatureCache(Path(tmp) / "cache", "fixture"))

    def test_visual_adapter_checkpoint_and_detached_feature_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = fixture_rows(tmp)
            support = choose_support(rows)
            selected = resolve_support(rows, support)
            cfg = {"mode": "joint_synthetic", "steps": 2, "train_visual": True}
            a = NoveltyTrainer(cfg, rows, selected, {}, Path(tmp) / "a", TinyEncoder)
            a.output.mkdir()
            a.train_step()
            self.assertIsNone(a.model.encoder.backbone.weight.grad)
            self.assertGreater(float(a.model.encoder.visual_inlayer_bank.adapters["brain"].weight.grad.abs().sum()), 0)
            a.save_checkpoint("checkpoint_last.pt")
            b = NoveltyTrainer(cfg, rows, selected, {}, Path(tmp) / "b", TinyEncoder)
            b.load_checkpoint(a.output / "checkpoint_last.pt")
            for k, v in a.model.encoder.visual_inlayer_bank.state_dict().items():
                self.assertTrue(torch.equal(v, b.model.encoder.visual_inlayer_bank.state_dict()[k]))
            self.assertEqual(b.step, 1)

    def test_invalid_configuration_is_not_silently_ignored(self):
        from paclip.engine.novelty_trainer import checked_config
        for config in ({"lambda_rnak": .2}, {"steps": 0}, {"train_visual": "false"}, {"head_spec": {"input_dim": 128}}):
            with self.assertRaises(ValueError):
                checked_config(config)
