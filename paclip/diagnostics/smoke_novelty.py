"""Explicit short synthetic-fixture smoke. Never a real MRI efficacy experiment."""
import argparse
from pathlib import Path
import torch
from paclip.diagnostics.novelty_fixtures import fixture_rows, TinyEncoder
from paclip.data.patient_manifest import choose_support, resolve_support, write_json
from paclip.engine.novelty_trainer import NoveltyTrainer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    from run_brain_novelty import check_device
    check_device(a.device)
    torch.set_num_threads(2)
    a.output.mkdir(parents=True, exist_ok=False)
    data = a.output / "synthetic_fixtures"; data.mkdir()
    rows = fixture_rows(data)
    support = choose_support(rows)
    selected = resolve_support(rows, support)
    trainer = NoveltyTrainer({"mode": "joint_synthetic", "steps": 2, "save_every": 1, "validate_every": 2,
        "lambda_rank": .2}, rows, selected,
        {"base_sha256": "SYNTHETIC_TEST_ENCODER_NOT_BIOMEDCLIP", "split_digest": support["split_digest"],
         "support_digest": support["support_digest"]}, a.output / "fixture_training", TinyEncoder, device=a.device)
    trainer.run()
    write_json(a.output / "smoke_result.json", {"passed": True, "device": a.device, "steps": trainer.step,
        "counts": trainer.counts, "encoder": "TinyEncoder test substitute", "real_mri_experiment": False,
        "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name() if a.device.startswith("cuda") else None})


if __name__ == "__main__":
    main()
