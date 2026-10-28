"""Independent three-source training, lazy encoder loading and exact step resume."""
from __future__ import annotations
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from paclip.data.datasets import _gray2clip
from paclip.data.novelty_dataset import NoveltyDataset
from paclip.data.mixed_sampling import MixedSourceSampler
from paclip.data.synthetic_image import synthesize_image, ImageSyntheticConfig
from paclip.data.synthetic_feature import synthesize_feature
from paclip.data.supervision import pool_mask_targets
from paclip.data.patient_manifest import write_json, write_rows
from paclip.data.fewshot import stable_digest, file_sha256
from paclip.models.novelty_model import NoveltyModel
from paclip.models.dense_localizer import top_roi_score
from paclip.losses_novelty import grouped_novelty_loss, source_localization_metrics
from paclip.implementation_identity import implementation_fingerprint
from paclip.evaluation.novelty_protocol import fit_operating_points, evaluate, clean_json, check_condition

DEFAULTS = {"mode": "dense_only", "seed": 0, "steps": 2700, "batch_size": 8,
    "task": "brain_mri", "organ": None, "prompt_set": None,
    "lr": 1e-4, "weight_decay": 1e-5, "image_size": 224, "train_visual": False,
    "lambda_syn": 1., "lambda_normal": .25, "lambda_rank": 0., "margin": .5,
    "boundary_weight": .1, "recall_weight": .1, "log_every": 20,
    "preview_every": 100, "validate_every": 100, "save_every": 100,
    "source_condition": "auto", "normal_budget": .01, "baseline_steps": 2700,
    "image_synthetic_options": {},
    "feature_noise_mode": "relative_l2", "feature_strengths": [.05, .1, .2], "feature_detach_reference": True,
    "rank_mode": "per_lesion_hard", "rank_aggregation": "source_sum",
    "selection": "source_validation_full_image_ap", "head_spec": {}}


def checked_config(config):
    unknown = set(config) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
    c = {**DEFAULTS, **config}
    from paclip.prompts import TASK_PROMPT_SETS, DEFAULT_TASK_PROMPT_SETS
    organs = {"brain_mri": "brain", "thymoma": "thymoma"}
    if c["task"] not in organs:
        raise ValueError("Unsupported novelty task")
    c["organ"] = c["organ"] or organs[c["task"]]
    c["prompt_set"] = c["prompt_set"] or DEFAULT_TASK_PROMPT_SETS[c["task"]]
    if c["organ"] != organs[c["task"]] or c["prompt_set"] not in TASK_PROMPT_SETS[c["task"]]:
        raise ValueError("Task, organ and prompt set disagree")
    for key in ("steps", "baseline_steps", "batch_size", "log_every", "preview_every", "validate_every", "save_every"):
        if not isinstance(c[key], int) or c[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if c["image_size"] != 224:
        raise ValueError("Formal BiomedCLIP pipeline uses 224; standalone head tests may use other grids")
    if c["selection"] != "source_validation_full_image_ap":
        raise ValueError("Unsupported selection rule")
    for key in ("lambda_syn", "lambda_normal", "lambda_rank", "margin", "boundary_weight", "recall_weight", "weight_decay"):
        if not np.isfinite(c[key]) or c[key] < 0:
            raise ValueError(f"Invalid nonnegative {key}")
    if c["lr"] <= 0 or not np.isfinite(c["lr"]):
        raise ValueError("Learning rate must be finite and positive")
    if not isinstance(c["seed"], int) or c["seed"] < 0 or not isinstance(c["train_visual"], bool):
        raise ValueError("Seed must be a nonnegative integer and train_visual must be boolean")
    spec = c["head_spec"]
    if not isinstance(spec, dict) or set(spec) - {"n_levels", "input_dim", "projection_dim", "decoder_variant"}:
        raise ValueError("Unknown head architecture fields")
    if spec.get("n_levels", 3) != 3 or spec.get("input_dim", 512) != 512 or spec.get("projection_dim", 128) < 1:
        raise ValueError("Expected three 512-D visual levels and positive projection dimension")
    if c["source_condition"] not in {"auto", "exploratory", "same_sequence_seen", "same_sequence_unseen", "external_same_sequence", "cross_sequence"}:
        raise ValueError("Unknown source evaluation condition")
    if c["feature_noise_mode"] not in {"relative_l2", "per_dimension_gaussian"} or not isinstance(c["feature_detach_reference"], bool):
        raise ValueError("Invalid feature noise/gradient policy")
    if not c["feature_strengths"] or any(not np.isfinite(x) or x <= 0 for x in c["feature_strengths"]):
        raise ValueError("Invalid feature noise strengths")
    if c["rank_mode"] not in {"per_lesion_hard", "global_mean"} or c["rank_aggregation"] not in {"source_sum", "positive_image_mean"}:
        raise ValueError("Invalid rank policy")
    try:
        ImageSyntheticConfig(**c["image_synthetic_options"])
    except TypeError as exc:
        raise ValueError("Invalid image synthesis configuration") from exc
    return c


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None}


def restore_rng(state):
    random.setstate(state["python"]); np.random.set_state(state["numpy"]); torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_torch_save(path, payload):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


class NoveltyTrainer:
    VERSION = "novelty_checkpoint_v2"

    def __init__(self, config, rows, support_rows, identity, output, encoder_loader, *, device="cpu", cache=None, guard_reference=None, diagnostic_direction=None):
        self.cfg = checked_config(config)
        self.rows, self.support_rows = rows, support_rows
        self.output, self.device, self.encoder_loader = Path(output), torch.device(device), encoder_loader
        self.identity = {**identity, "config": self.cfg, "architecture": "dense_3level_gn_v1",
                         "visual_layers_zero_based": [5, 8, 11], "visual_blocks_one_based": [6, 9, 12]}
        self.guard_reference = guard_reference
        self.identity["development_guard_reference"] = guard_reference
        self.identity["implementation"] = implementation_fingerprint("training")
        direction = None if diagnostic_direction is None else torch.as_tensor(diagnostic_direction).detach().float().cpu().tolist()
        self.identity["diagnostic_direction_digest"] = stable_digest(direction)
        self.identity["torch_version"] = torch.__version__
        self.identity_digest = stable_digest(self.identity)
        self.cache = cache
        if cache is not None and self.cfg["train_visual"]:
            raise ValueError("Static visual cache is prohibited during visual-adapter tuning")
        random.seed(self.cfg["seed"]); np.random.seed(self.cfg["seed"]); torch.manual_seed(self.cfg["seed"])
        self.model = NoveltyModel(head_spec=self.cfg["head_spec"], diagnostic_direction=direction).to(self.device)
        self.encoded_images = 0
        if self.cfg["train_visual"]:
            self.ensure_encoder()
        self.model.train()
        self.optimizer = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad],
                                           lr=self.cfg["lr"], weight_decay=self.cfg["weight_decay"])
        self.model.assert_optimizer(self.optimizer)
        self.sampler = MixedSourceSampler(support_rows, self.cfg["mode"], self.cfg["batch_size"], self.cfg["seed"])
        self.dataset = NoveltyDataset(support_rows)
        self.step, self.best_ap, self.best_step = 0, -float("inf"), None
        self.counts = {k: 0 for k in ("normal", "real", "syn_image", "syn_feature")}
        self.elapsed_seconds = 0.

    def ensure_encoder(self):
        if self.model.encoder is None:
            # Loading a model must not alter the resumed stochastic training state.
            state = rng_state()
            try:
                self.model.encoder = self.encoder_loader().to(self.device)
            finally:
                restore_rng(state)
            self.model.train_visual = self.cfg["train_visual"]
            self.model.apply_policy()
            self.model.train(self.model.training)

    def encode(self, sample, *, image_override=None, detach=False):
        cached = None
        if self.cache is not None and image_override is None:
            cached = self.cache.get(sample)
        if cached is not None:
            return cached[0].to(self.device), cached[1]
        self.ensure_encoder()
        image = sample["image"] if image_override is None else image_override
        if detach:
            with torch.no_grad():
                levels, grid = self.model.extract_levels(image[None].to(self.device))
        else:
            levels, grid = self.model.extract_levels(image[None].to(self.device))
        self.encoded_images += 1
        if self.cache is not None and image_override is None:
            self.cache.put(sample, levels[0], grid)
        return levels[0].detach() if detach else levels[0], grid

    def build_batch(self, step):
        instances, features, masks, rois, metadata, previews = self.sampler.batch(step), [], [], [], [], []
        common_grid = None
        # Reuse a clean source only within this optimizer step. The graph remains
        # live for visual adapters; nothing is retained across parameter updates.
        # Image syntheses always take the separate fresh-encoding branch above.
        clean = {}
        for slot, inst in enumerate(instances):
            sample = self.dataset[inst["support_index"]]
            if not sample["roi_valid"]:
                raise ValueError(f"Unverified/empty training ROI: {sample['sample_id']}")
            mask, parameters = sample["mask_high"], {}
            if inst["synthetic_type"] == "image":
                synth = synthesize_image(sample["gray"], sample["roi"], inst["synthetic_seed"], roi_valid=sample["roi_valid"],
                                         config=ImageSyntheticConfig(**self.cfg["image_synthetic_options"]))
                levels, grid = self.encode(sample, image_override=_gray2clip(synth["gray"].numpy()))
                mask, parameters = synth["mask_high"], synth["parameters"]
                previews.append((slot, sample, synth))
            else:
                sid = sample["sample_id"]
                if sid in clean:
                    levels, grid = clean[sid]
                else:
                    levels, grid = self.encode(sample, detach=inst["synthetic_type"] == "feature" and self.cfg["feature_detach_reference"])
                    clean[sid] = levels, grid
                if inst["synthetic_type"] == "feature":
                    synth = synthesize_feature(levels, grid, sample["roi"], inst["synthetic_seed"], roi_valid=sample["roi_valid"],
                        rhos=self.cfg["feature_strengths"], noise_mode=self.cfg["feature_noise_mode"], detach_reference=self.cfg["feature_detach_reference"])
                    levels, mask, parameters = synth["levels"], synth["mask_high"], synth["parameters"]
                    previews.append((slot, sample, synth))
            if common_grid is not None and grid != common_grid:
                raise ValueError("Mixed visual grids in one batch")
            common_grid = grid
            features.append(levels); masks.append(mask); rois.append(sample["roi"])
            metadata.append({**inst, "patient_id": sample["patient_id"], "parameters": parameters})
        return torch.stack(features), common_grid, masks, torch.stack(rois).to(self.device), metadata, previews

    def train_step(self):
        self.model.train()
        levels, grid, masks, rois, metadata, previews = self.build_batch(self.step)
        output = self.model.decode_levels(levels, grid, rois)
        kwargs = {k: self.cfg[k] for k in ("lambda_syn", "lambda_normal", "lambda_rank", "margin", "boundary_weight", "recall_weight", "rank_mode", "rank_aggregation")}
        loss = grouped_novelty_loss(output["dense_logits"], masks, rois,
            [r["source_kind"] for r in metadata], [r["synthetic_type"] for r in metadata], **kwargs)
        if not torch.isfinite(loss["total"]):
            raise FloatingPointError("Nonfinite training loss")
        self.optimizer.zero_grad(set_to_none=True)
        loss["total"].backward()
        self.model.assert_optimizer(self.optimizer)
        for name, param in self.model.named_parameters():
            if not param.requires_grad and param.grad is not None:
                raise AssertionError(f"Frozen parameter has gradient: {name}")
            if param.requires_grad and (param.grad is None or not torch.isfinite(param.grad).all()):
                raise AssertionError(f"Missing/nonfinite trainable gradient: {name}")
        grad_norm = float(torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], 10.))
        self.optimizer.step()
        self.step += 1
        for key, n in loss["counts"].items():
            self.counts[key] += n
        row = {k: float(v.detach()) if torch.is_tensor(v) else v for k, v in loss.items()}
        row.update(source_localization_metrics(output["dense_logits"], masks, rois,
            [r["source_kind"] for r in metadata], [r["synthetic_type"] for r in metadata]))
        row.update(step=self.step, instances=metadata, cumulative_counts=dict(self.counts),
                   encoded_images=self.encoded_images, gradient_norm=grad_norm, gradient_policy_verified=True)
        if self.step == 1 or self.step % self.cfg["preview_every"] == 0:
            self.save_previews(previews, output["dense_logits"].detach(), loss["diagnostics"])
        return clean_json(row)

    def save_previews(self, previews, logits, diagnostics=None):
        from PIL import Image
        folder = self.output / "synthetic_preview" / f"step_{self.step:06d}"
        if not previews:
            return
        folder.mkdir(parents=True, exist_ok=True)
        for slot, sample, synth in previews:
            arrays = {"source_gray": sample["gray"].numpy(), "mask": synth["mask_high"].detach().cpu().numpy(),
                      "dense_logits": logits[slot, 0].cpu().numpy(),
                      "synthetic_anomaly_prediction": logits[slot, 0].sigmoid().cpu().numpy()}
            if diagnostics is not None:
                arrays["synthetic_loss"] = np.asarray(diagnostics[slot]["local_loss"], dtype=np.float32)
            if "levels" in synth:
                arrays["synthetic_patch_mask"] = synth["mask_high"].detach().cpu().numpy()
                arrays["synthetic_feature_map"] = synth["levels"].detach().cpu().numpy()
            for key in ("gray", "blend_mask", "perturbation_norm"):
                if key in synth:
                    arrays[key] = synth[key].detach().cpu().numpy()
            np.savez_compressed(folder / f"slot_{slot}.npz", **arrays)
            write_json(folder / f"slot_{slot}.json", {"source_sample_id": sample["sample_id"], **synth["parameters"]})
            # Fixed grayscale scale and fixed sigmoid scale; no per-image min/max stretch.
            panels = [arrays["source_gray"]]
            if "gray" in arrays:
                panels.append(arrays["gray"])
            panels.append(np.asarray(Image.fromarray((arrays["mask"] * 255).astype(np.uint8)).resize((224, 224), Image.Resampling.NEAREST)) / 255)
            probability = torch.sigmoid(logits[slot:slot+1]).cpu()
            panels.append(F.interpolate(probability, (224, 224), mode="bilinear", align_corners=False)[0, 0].numpy())
            Image.fromarray((np.concatenate(panels, axis=1).clip(0, 1) * 255).astype(np.uint8)).save(folder / f"slot_{slot}.png")

    @torch.no_grad()
    def predict(self, rows, output_dir=None):
        self.model.eval()
        result = []
        dataset = NoveltyDataset(rows)
        if output_dir is not None:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        for i in range(len(dataset)):
            s = dataset[i]
            levels, grid = self.encode(s)
            pred = self.model.decode_levels(levels[None], grid, s["roi"][None].to(self.device), [s["roi_valid"]])
            logits = F.interpolate(pred["dense_logits"], (224, 224), mode="bilinear", align_corners=False)
            # Image score is defined on the same evaluation grid as all metrics.
            image_score, _ = top_roi_score(logits, s["roi"][None].to(self.device), [s["roi_valid"]])
            _, truth = pool_mask_targets([s["mask_high"]], (224, 224))
            item = {**rows[i], "score": logits[0, 0].cpu().numpy(), "gt": truth[0].numpy().astype(bool),
                    "roi": s["roi"].numpy().astype(bool), "roi_valid": s["roi_valid"],
                    "image_score": float(image_score[0])}
            result.append(item)
            if output_dir is not None:
                stem = stable_digest(s["sample_id"])[:20]
                diagnostic_maps = {key: pred[key][0].cpu().numpy() for key in ("map_layer5", "map_layer8", "map_layer11")
                                   if pred[key] is not None}
                np.savez_compressed(Path(output_dir) / (stem + ".npz"), logit=item["score"],
                    probability=logits.sigmoid()[0, 0].cpu().numpy(), gt=item["gt"], roi=item["roi"],
                    sample_id=s["sample_id"], image_score=item["image_score"],
                    dense_anomaly_map=pred["dense_anomaly_map"][0].cpu().numpy(),
                    layer_maps_available=pred["layer_maps_available"], **diagnostic_maps)
        return result

    def save_checkpoint(self, name):
        visual = self.model.encoder.visual_inlayer_bank.state_dict() if self.cfg["train_visual"] else None
        payload = {"version": self.VERSION, "identity": self.identity, "identity_digest": self.identity_digest,
            "head_spec": self.model.head.spec, "head": self.model.head.state_dict(), "visual_adapter": visual,
            "diagnostic_direction": self.model.diagnostic_direction,
            "optimizer": self.optimizer.state_dict(), "scheduler": None, "scheduler_policy": "constant_lr",
            "step": self.step, "best_ap": self.best_ap, "best_step": self.best_step, "counts": self.counts,
            "encoded_images": self.encoded_images, "elapsed_seconds": self.elapsed_seconds,
            "rng_state": rng_state(), "trainable_names": self.model.trainable_names()}
        atomic_torch_save(self.output / name, payload)

    def load_checkpoint(self, path, *, resume=True):
        blob = torch.load(path, map_location="cpu", weights_only=False)
        if blob.get("version") != self.VERSION or blob["identity_digest"] != self.identity_digest:
            raise ValueError("Checkpoint version/config/data/base-weight identity mismatch")
        if stable_digest(blob["identity"]) != blob["identity_digest"]:
            raise ValueError("Checkpoint embedded identity was modified")
        saved_direction = blob.get("diagnostic_direction")
        saved_list = None if saved_direction is None else saved_direction.detach().float().cpu().tolist()
        if stable_digest(saved_list) != self.identity["diagnostic_direction_digest"]:
            raise ValueError("Checkpoint diagnostic direction mismatch")
        if blob["head_spec"] != self.model.head.spec or blob["trainable_names"] != self.model.trainable_names():
            raise ValueError("Checkpoint architecture/freeze policy mismatch")
        self.model.head.load_state_dict(blob["head"], strict=True)
        if self.cfg["train_visual"]:
            self.model.encoder.visual_inlayer_bank.load_state_dict(blob["visual_adapter"], strict=True)
        if resume:
            self.optimizer.load_state_dict(blob["optimizer"])
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)
            self.step, self.best_ap, self.best_step = blob["step"], blob["best_ap"], blob["best_step"]
            self.counts = blob["counts"]
            self.encoded_images, self.elapsed_seconds = blob["encoded_images"], blob["elapsed_seconds"]
            restore_rng(blob["rng_state"])
        self.model.assert_optimizer(self.optimizer)
        return blob

    def run(self, *, resume=False, evaluation_sets=None, stop_after=None):
        evaluation_sets = evaluation_sets or []
        from paclip.evaluation.novelty_protocol import resolve_condition, evaluation_coverage
        source_test = [r for r in self.rows if r["split"] == "test"]
        sets = []
        if source_test:
            sets.append(("源集测试", source_test, resolve_condition(self.rows, source_test, self.cfg["source_condition"])))
        else:
            sets.append(("源集测试", [], "exploratory"))
        for name, records, condition in evaluation_sets:
            sets.append((name, records, resolve_condition(self.rows, records, condition)))
        coverage = evaluation_coverage(sets)
        if resume:
            self.load_checkpoint(self.output / "checkpoint_last.pt")
            # Drop only uncommitted log tail after an interrupted checkpoint interval.
            path = self.output / "training_log.jsonl"
            if path.exists():
                committed = []
                for line in path.read_text(encoding="utf-8").splitlines():
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        break
                    if row["step"] <= self.step:
                        committed.append(row)
                write_rows(path, committed)
        else:
            self.output.mkdir(parents=True, exist_ok=False)
            write_json(self.output / "config.json", self.cfg)
            write_json(self.output / "identity.json", self.identity)
            write_json(self.output / "evaluation_plan.json", {"sets": [{"name": n, "condition": c, "samples": len(rs)} for n, rs, c in sets],
                                                               "four_condition_coverage": coverage})
            write_rows(self.output / "split_manifest.jsonl", self.rows)
            write_rows(self.output / "support_samples.jsonl", self.support_rows)
            write_json(self.output / "trainable_parameters.json", {"names": self.model.trainable_names(),
                "count": sum(p.numel() for p in self.model.parameters() if p.requires_grad)})
        validation = [r for r in self.rows if r["split"] == "validation"]
        if not validation:
            raise ValueError("Source validation required for model and threshold selection")
        started = time.perf_counter()
        base_elapsed = self.elapsed_seconds
        try:
            write_json(self.output / "completion.json", {"status": "training", "step": self.step})
            while self.step < self.cfg["steps"]:
                row = self.train_step()
                self.elapsed_seconds = base_elapsed + time.perf_counter() - started
                row["elapsed_seconds"] = self.elapsed_seconds
                with (self.output / "training_log.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                if self.step == 1 or self.step % self.cfg["log_every"] == 0:
                    print(f"[novelty] step={self.step}/{self.cfg['steps']} loss={row['total']:.5f} counts={row['counts']}", flush=True)
                if self.step % self.cfg["validate_every"] == 0 or self.step == self.cfg["steps"]:
                    items = self.predict(validation)
                    points = fit_operating_points(items, self.cfg["normal_budget"])
                    # Selection only needs ranking, not a repeated full FROC computation.
                    from paclip.evaluation.ranking import binary_ranking_metrics
                    ap = binary_ranking_metrics(np.concatenate([r["score"].ravel() for r in items]),
                        np.concatenate([r["gt"].ravel() for r in items]))["pixel_ap"]
                    if not np.isfinite(ap):
                        raise ValueError("Source validation AP is undefined")
                    guard = {"passed": True, "reason": "not_enabled_AP_selection_only"}
                    if self.guard_reference is not None:
                        from paclip.evaluation.novelty_protocol import _detection, development_guard
                        guard = development_guard(_detection(items, points["dice_threshold"]), self.guard_reference)
                    if guard["passed"] and ap > self.best_ap:
                        self.best_ap, self.best_step = ap, self.step
                        self.save_checkpoint("checkpoint_selected.pt")
                        write_json(self.output / "validation_selection.json", {"step": self.step, "ap": ap,
                            "rule": self.cfg["selection"], "operating_points": points,
                            "guard": guard})
                    write_json(self.output / f"validation_step_{self.step:06d}.json", {"ap": ap, "operating_points": points, "guard": guard})
                if self.step % self.cfg["save_every"] == 0 or self.step == self.cfg["steps"] or (stop_after and self.step >= stop_after):
                    self.save_checkpoint("checkpoint_last.pt")
                if stop_after and self.step >= stop_after and self.step < self.cfg["steps"]:
                    write_json(self.output / "completion.json", {"status": "stopped_for_test", "step": self.step})
                    return
            if self.best_step is None:
                raise RuntimeError("No source-validation checkpoint passed the declared selection guard; final tests were not evaluated")
            self.load_checkpoint(self.output / "checkpoint_selected.pt", resume=False)
            selected_validation = self.predict(validation, self.output / "evaluation" / "source_validation" / "scores")
            points = fit_operating_points(selected_validation, self.cfg["normal_budget"])
            write_json(self.output / "protocol.json", {**points, "identity_digest": self.identity_digest,
                "selected_step": self.best_step, "test_used_for_selection": False, "four_condition_coverage": coverage})
            results = {"source_validation": evaluate(selected_validation, points, include_oracle=False)}
            for name, records, condition in sets:
                if not records:
                    results[name] = {"status": "unavailable", "reason": "No verified samples supplied"}
                    continue
                check = check_condition(self.rows, records, condition)
                destination = self.output / "evaluation" / name
                items = self.predict(records, destination / "scores")
                results[name] = {"status": "complete", "condition": check, **evaluate(items, points)}
                for item in items:
                    # Binary predictions for both SOURCE-selected operating points.
                    np.savez_compressed(destination / "scores" / (stable_digest(item["sample_id"])[:20] + "_binary.npz"),
                        dice=item["score"] >= points["dice_threshold"],
                        normal_budget=item["score"] >= points["normal_budget_threshold"])
                write_json(destination / "metrics.json", results[name])
            write_json(self.output / "results.json", results)
            self.elapsed_seconds = base_elapsed + time.perf_counter() - started
            expected = {k: self.cfg["steps"] * q for k, q in zip(self.counts, self.sampler.quotas)}
            if self.counts != expected:
                raise AssertionError(f"Run-level source counts mismatch: {self.counts} != {expected}")
            write_json(self.output / "completion.json", {"status": "complete", "steps": self.step,
                "selected_step": self.best_step, "counts": self.counts, "elapsed_seconds": self.elapsed_seconds,
                "encoded_images": self.encoded_images, "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                "checkpoint_selected_sha256": file_sha256(self.output / "checkpoint_selected.pt"),
                "results_sha256": file_sha256(self.output / "results.json"), "identity_digest": self.identity_digest})
            return results
        except BaseException as exc:
            write_json(self.output / "completion.json", {"status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                "last_attempted_step": self.step, "error": str(exc),
                "resume_checkpoint_exists": (self.output / "checkpoint_last.pt").exists()})
            raise
