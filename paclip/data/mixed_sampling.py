"""Step-indexed, replayable quotas. Synthetic sources are NORMAL support only."""
import numpy as np

MODES = {"dense_only": (4, 4, 0, 0), "ratio_control": (6, 2, 0, 0),
         "image_synthetic": (4, 2, 2, 0), "feature_synthetic": (4, 2, 0, 2),
         "joint_synthetic": (4, 2, 1, 1)}


def instance_seed(seed, step, slot, stream=0):
    return int(np.random.SeedSequence([int(seed), int(step), int(slot), int(stream)]).generate_state(1)[0])


class MixedSourceSampler:
    def __init__(self, support, mode, batch_size=8, seed=0):
        if mode not in MODES or batch_size < 4 or batch_size % 4:
            raise ValueError("Unknown mode or batch size not a positive multiple of four")
        raw = np.array(MODES[mode]) * batch_size / 8
        if not np.all(raw == raw.astype(int)):
            raise ValueError("Joint synthesis requires a batch size divisible by eight")
        self.quotas = tuple(map(int, raw))
        self.support, self.seed = list(support), int(seed)
        self.normal = [i for i, r in enumerate(support) if r["label"] == 0]
        self.real = [i for i, r in enumerate(support) if r["label"] == 1]
        if not self.normal or not self.real:
            raise ValueError("Both real classes are required; quotas cannot silently degrade")
        if any(r.get("split") != "train" for r in support):
            raise ValueError("Sampler accepts training support only")

    def batch(self, step):
        rng = np.random.default_rng(instance_seed(self.seed, step, 0, 11))
        result = []
        for count, kind, synth in zip(self.quotas, ("normal", "real", "synthetic", "synthetic"),
                                      ("none", "none", "image", "feature")):
            pool = self.real if kind == "real" else self.normal
            for _ in range(count):
                idx = int(rng.choice(pool))
                result.append({"support_index": idx, "source_kind": kind, "synthetic_type": synth,
                               "source_sample_id": self.support[idx]["sample_id"],
                               "synthetic_seed": instance_seed(self.seed, step, len(result), 12)})
        # Counts are exact, including repeated support indices.
        return result
