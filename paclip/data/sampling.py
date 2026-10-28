"""E1 lesion-aware batches built exclusively from the selected training support."""

import math

import numpy as np
import torch
from torch.utils.data import Sampler
from paclip.data.supervision import get_supervision_masks


class LesionBalancedBatchSampler(Sampler):
    """50% positive, 25% normal, 25% hard/adjacent negative when available.

    Small batches use the closest feasible integer quotas. Sampling with
    replacement permits a small support set; unavailable groups are explicitly
    recorded in summary rather than fabricated. This is a training sampler only.
    """

    def __init__(self, dataset, batch_size, seed=0):
        self.batch_size = int(batch_size)
        if self.batch_size < 1 or len(dataset) < 1:
            raise ValueError("Sampling requires a nonempty dataset and positive batch size")
        self.seed = int(seed)
        self.epoch = 0
        self.dataset = dataset
        positive, negative, areas, boundary = [], [], {}, set()
        self.sample_ids = []
        boundary_known = 0
        for index in range(len(dataset)):
            item = dataset[index]
            mask = torch.as_tensor(get_supervision_masks(item))
            hard_positive = bool((mask > 0).any())
            (positive if hard_positive else negative).append(index)
            areas[index] = float((mask > 0).float().mean())
            self.sample_ids.append(str(item.get("sample_id", index)))
            boundary_known += bool(item.get("boundary_known", "lesion_boundary" in item))
            if item.get("lesion_boundary", False):
                boundary.add(index)
        self.positive = np.asarray(positive, dtype=np.int64)
        self.negative = np.asarray(negative, dtype=np.int64)
        self.hard = np.asarray([], dtype=np.int64)
        self.normal = self.negative
        self.adjacent = set(getattr(dataset, "adjacent_negative_indices", []))
        if positive and negative:
            # Deterministic provisional groups, refreshed by the model before training.
            self._partition_negatives({index: 0.0 for index in negative})
        weights = np.asarray([(2.0 if i in boundary else 1.0) / math.sqrt(max(areas[i], 1e-6)) for i in positive])
        self.positive_weights = weights / weights.sum() if len(weights) else None
        self.summary = {
            "policy": "e1_50_positive_25_normal_25_hard",
            "positive_slices": len(positive), "negative_slices": len(negative),
            "hard_negative_slices": len(self.hard), "boundary_slices": len(boundary),
            "hard_negative_rule": "adjacent first, then current-model mean top-5%-patch anomaly probability",
            "hard_mining_status": "pending_model_refresh" if negative and positive else "not_applicable",
            "hard_mining_history": [],
            "boundary_known_slices": boundary_known,
            "boundary_priority_status": "active" if boundary else (
                "known_nonboundary_only" if boundary_known else "unavailable_no_verified_slice_context"),
            "positive_priority": "inverse square root lesion area fraction; boundary weight 2",
            "sampling_with_replacement": True,
            "fallback": "missing class: sample available class only" if not positive or not negative else (
                "batch_size < 4: closest feasible integer quotas" if self.batch_size < 4 else None),
            "normal_and_hard_overlap": len(negative) == 1,
        }
        if hasattr(dataset, "pairs"):
            self.summary["training_volume_slices"] = [list(pair) for pair in dataset.pairs]

    def _partition_negatives(self, scores):
        ranked = sorted(self.negative.tolist(), key=lambda i: (i in self.adjacent, scores[i], -i), reverse=True)
        count = max(1, len(ranked) // 2)
        self.hard = np.asarray(ranked[:count], dtype=np.int64)
        hard_set = set(self.hard.tolist())
        self.normal = np.asarray([i for i in self.negative if i not in hard_set], dtype=np.int64)
        if not len(self.normal):
            self.normal = self.negative

    def update_hard_negatives(self, scores, step):
        """Scores must cover exactly the selected support's negative samples."""
        if set(scores) != set(self.negative.tolist()) or not all(math.isfinite(float(v)) for v in scores.values()):
            raise ValueError("Mining requires finite scores for every selected negative, and no other samples")
        if not len(self.negative) or not len(self.positive):
            return
        self._partition_negatives(scores)
        self.summary["hard_mining_status"] = "model_refreshed"
        self.summary["hard_mining_history"].append({
            "before_step": int(step),
            "scores": {self.sample_ids[i]: float(scores[i]) for i in self.negative},
            "hard_sample_ids": [self.sample_ids[i] for i in self.hard],
        })

    def __len__(self):
        return max(1, math.ceil((len(self.positive) + len(self.negative)) / self.batch_size))

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        for batch_index in range(len(self)):
            if not len(self.positive) or not len(self.negative):
                pool = self.positive if len(self.positive) else self.negative
                weights = self.positive_weights if len(self.positive) else None
                yield rng.choice(pool, size=self.batch_size, replace=True, p=weights).tolist()
                continue
            if self.batch_size == 1:
                # Alternate classes even when a pass contains multiple singleton batches.
                ordinal = (self.epoch - 1) * len(self) + batch_index
                pool = self.positive if ordinal % 2 == 0 else self.negative
                yield [int(rng.choice(pool))]
                continue
            n_positive = math.ceil(self.batch_size / 2)
            n_normal = max(1, self.batch_size // 4)
            n_hard = self.batch_size - n_positive - n_normal
            batch = rng.choice(self.positive, n_positive, replace=True, p=self.positive_weights).tolist()
            batch += rng.choice(self.normal, n_normal, replace=True).tolist()
            batch += rng.choice(self.hard, n_hard, replace=True).tolist()
            rng.shuffle(batch)
            yield batch


@torch.no_grad()
def refresh_hard_negatives(model, sampler, anchors, device, step, cached_enc=None, batch_size=16):
    """Mine only selected support negatives; restore every module's train/eval mode."""
    if not len(sampler.positive) or not len(sampler.negative):
        return
    modes = [(module, module.training) for module in model.modules()]
    scores = {}
    try:
        model.eval()
        encoded = cached_enc if cached_enc is not None else model.encode_anchors(anchors)
        indices = sampler.negative.tolist()
        for start in range(0, len(indices), batch_size):
            selected = indices[start:start + batch_size]
            images = torch.stack([sampler.dataset[i]["image"] for i in selected]).to(device)
            logits = model(images, encoded)["patch_logits"].float()
            probability = (logits[:, 1] - logits[:, 0]).sigmoid().flatten(1)
            k = max(1, math.ceil(probability.shape[1] * .05))
            values = probability.topk(k, dim=1).values.mean(1).cpu().tolist()
            scores.update(zip(selected, values))
        sampler.update_hard_negatives(scores, step)
    finally:
        for module, training in modes:
            module.training = training


def add_adjacent_volume_negatives(dataset):
    """Include both positive endpoints and nearest negatives of selected volumes.

    K still counts selected volumes. Validation/test volumes are never consulted.
    The exact added pairs are exported in the sampler's summary.
    """
    import nibabel as nib

    original = list(dataset.pairs)
    occupied = set(original)
    adjacent_indices = []
    boundary_pairs = set()
    for index in sorted({index for index, _ in original}):
        record = dataset.base.records[index]
        if int(record["label"]) != 1 or not record.get("mask_path"):
            continue
        mask = np.asarray(nib.load(record["mask_path"]).dataobj) > 0
        if mask.ndim != 3:
            raise ValueError("Adjacent slice sampling expects a 3D lesion mask")
        positive_z = np.flatnonzero(mask.any(axis=(0, 1)))
        if not len(positive_z):
            continue
        for z in (int(positive_z[0]), int(positive_z[-1])):
            pair = (index, z)
            boundary_pairs.add(pair)
            if pair not in occupied:
                dataset.pairs.append(pair)
                occupied.add(pair)
        for z in (int(positive_z[0]) - 1, int(positive_z[-1]) + 1):
            pair = (index, z)
            if 0 <= z < mask.shape[2] and pair not in occupied:
                adjacent_indices.append(len(dataset.pairs))
                dataset.pairs.append(pair)
                occupied.add(pair)
    dataset.adjacent_negative_indices = adjacent_indices
    dataset.lesion_boundary_pairs = boundary_pairs
    return dataset
