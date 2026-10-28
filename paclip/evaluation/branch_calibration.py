"""Optional later-stage empirical normal CDF calibration; NOT a lesion probability."""
import numpy as np


class NormalCDF:
    def __init__(self, values, metadata, eps=1e-4):
        self.values = np.sort(np.asarray(values, dtype=np.float64).ravel())
        if not len(self.values) or not np.isfinite(self.values).all() or not 0 < eps < .5:
            raise ValueError("Finite nonempty source normal values and valid eps required")
        if metadata.get("split") != "validation" or metadata.get("normal_only") is not True or not metadata.get("sequence"):
            raise ValueError("Calibration must identify source normal validation and sequence")
        self.eps = eps
        self.metadata = {**metadata, "n_pixels": len(self.values), "tail_resolution": 1 / len(self.values),
                         "eps": eps, "meaning": "normal_percentile_not_lesion_probability"}

    def transform(self, score):
        if not np.isfinite(np.asarray(score)).all():
            raise ValueError("Calibration input must be finite")
        q = np.searchsorted(self.values, np.asarray(score), side="right") / len(self.values)
        q = np.clip(q, self.eps, 1 - self.eps)
        return np.log(q) - np.log1p(-q)

    def save(self, path):
        import json
        np.savez_compressed(path, values=self.values, metadata=json.dumps(self.metadata), eps=self.eps)

    @classmethod
    def load(cls, path):
        import json
        with np.load(path, allow_pickle=False) as blob:
            return cls(blob["values"], json.loads(str(blob["metadata"])), float(blob["eps"]))


def calibrated_fusion(branches, calibrators, weights):
    if set(branches) != set(calibrators) or set(branches) != set(weights):
        raise ValueError("All named branches need their own calibrator and weight")
    if any(w < 0 or not np.isfinite(w) for w in weights.values()) or not np.isclose(sum(weights.values()), 1):
        raise ValueError("Nonnegative fusion weights must sum to one")
    if len({np.asarray(x).shape for x in branches.values()}) != 1:
        raise ValueError("Branch maps must align spatially")
    return sum(weights[k] * calibrators[k].transform(branches[k]) for k in branches)
