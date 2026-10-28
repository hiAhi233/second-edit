"""Only clean, frozen, PRE-projection visual features may be cached."""
from pathlib import Path
import torch
from paclip.data.fewshot import stable_digest, file_sha256
from paclip.implementation_identity import implementation_fingerprint


class CleanFeatureCache:
    VERSION = "clean_visual_levels_v2"

    def __init__(self, root, encoder_sha, image_size=224, layer_indices=(5, 8, 11)):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.identity = {"version": self.VERSION, "encoder_sha256": encoder_sha,
            "image_size": image_size, "layers_zero_based": list(layer_indices),
            "preprocess": "PIL_L_bilinear_gray01_CLIP_mean_std_v1", "grid": [image_size // 16] * 2,
            "features": "L2_normalized_each_level_before_trainable_projection"}
        self.identity["extractor_implementation"] = implementation_fingerprint("features")
        from importlib.metadata import version, PackageNotFoundError
        self.identity["runtime"] = {"torch": torch.__version__}
        for package in ("timm", "open_clip_torch", "Pillow"):
            try:
                self.identity["runtime"][package] = version(package)
            except PackageNotFoundError:
                self.identity["runtime"][package] = "unavailable"
        self.hits = self.misses = 0

    def key(self, sample):
        return stable_digest({**self.identity, "sample_id": sample["sample_id"],
            "image_sha256": sample["image_sha256"], "roi_sha256": sample.get("roi_sha256"),
            "roi_policy": sample["roi_policy"], "roi_valid": sample["roi_valid"]})

    def get(self, sample):
        key = self.key(sample)
        path = self.root / (key + ".pt")
        if not path.exists():
            self.misses += 1
            return None
        blob = torch.load(path, map_location="cpu", weights_only=True)
        x, grid = blob["levels"], tuple(blob["grid"])
        if blob.get("key") != key or x.shape != (len(self.identity["layers_zero_based"]), grid[0] * grid[1], 512) or not torch.isfinite(x).all():
            raise ValueError(f"Corrupt/incompatible clean feature cache: {path}")
        if list(grid) != self.identity["grid"]:
            raise ValueError("Cached visual grid mismatch")
        self.hits += 1
        return x, grid

    def put(self, sample, levels, grid):
        key = self.key(sample)
        path = self.root / (key + ".pt")
        tmp = path.with_suffix(".tmp")
        torch.save({"key": key, "levels": levels.detach().float().cpu(), "grid": list(grid)}, tmp)
        tmp.replace(path)
