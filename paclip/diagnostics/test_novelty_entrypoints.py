"""Baseline manifest bridge, reused live model and report completeness."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch

from paclip.diagnostics.novelty_fixtures import fixture_rows, TinyEncoder
from paclip.data.patient_manifest import audit_rows, choose_support, resolve_support


class ToyLegacy(TinyEncoder):
    def __init__(self):
        super().__init__()
        self.cfg = SimpleNamespace(temperature=.07)
        self.register_buffer("fusion_weights", torch.zeros(3))

    def encode_anchors(self, prompts):
        eye = torch.eye(512, device=self.channels.device)
        return {str(i): {"normal": eye[0], "abnormal": eye[i + 1]} for i in range(3)}

    def forward(self, image, anchors):
        _, _, grid, features = self.encode_image(image, True)
        logits = features[:, 0, :, 0].reshape(len(image), 1, *grid)
        return {"patch_logits": torch.cat([-logits, logits], 1), "f_patch_levels": features}


class EntryPointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_baseline_seed_can_change_without_resampling_support_and_report(self):
        from run_brain_novelty import run_baseline
        from report_brain_novelty import main as report_main
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = fixture_rows(root)
            support = choose_support(rows, seed=0)
            audit = audit_rows(rows)
            selected = resolve_support(rows, support)
            args = SimpleNamespace(output=root / "baseline", seed=1, steps=1, device="cpu", cache=root / "cache")
            model = ToyLegacy()
            def fake_run(cfg, model_factory):
                self.assertIsNone(cfg.init_ckpt)
                self.assertEqual(cfg.seed, 1)
                saved = json.loads(Path(cfg.support_manifest).read_text(encoding="utf-8"))
                self.assertEqual(saved["seed"], 1)
                self.assertEqual([r["sample_id"] for r in saved["samples"]], support["sample_ids"])
                self.assertIs(model_factory(None, None, None), model)
                checkpoint = args.output / "FIXTURE_NOT_A_REAL_CHECKPOINT.pt"
                checkpoint.write_bytes(b"test fixture only")
                return {"checkpoint": str(checkpoint), "evaluation_status": "ok", "completed_steps": cfg.steps}
            with patch("paclip.engine.trainer.default_model_factory", return_value=model), patch("paclip.engine.trainer.run_experiment", side_effect=fake_run):
                run_baseline(args, rows, audit, support, selected)
            self.assertTrue((args.output / "baseline_provenance.json").exists())
            protocol = json.loads((args.output / "protocol.json").read_text(encoding="utf-8"))
            self.assertEqual(protocol["fit_split"], "validation")
            report = root / "report.md"
            with patch("sys.argv", ["report_brain_novelty.py", "--root", str(root), "--output", str(report)]):
                report_main()
            self.assertIn("患者独立第一步基准", report.read_text(encoding="utf-8"))
            self.assertIn("small_lesion_recall", report.read_text(encoding="utf-8"))
