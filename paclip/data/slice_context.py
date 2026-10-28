"""可信切片索引：只在病例、序列、划分、方向和层间距一致时建立邻片关系。

源 Brain_AD PNG 当前没有真实 slice_z（返回 -1），因此源域索引明确标记为
不可用；不能根据文件名数字或排序伪造相邻切片。外部 NIfTI 可以用真实 z
轴建立邻接，但仍逐例检查方向、层间距和脑区有效性。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np


def sequence_key(value) -> str:
    text = str(value or "unknown").strip()
    for separator in (",", "(", ";", "/"):
        text = text.split(separator)[0]
    return text.strip() or "unknown"


@dataclass(frozen=True)
class SliceMeta:
    sample_id: str
    case_id: str
    series_id: str
    sequence: str
    slice_z: int
    split: str
    spacing_z: Optional[float] = None
    orientation: Optional[str] = None
    brain_valid: bool = True


@dataclass
class SliceIndex:
    meta: Dict[int, SliceMeta]
    neighbors: Dict[int, Tuple[Optional[int], Optional[int]]]
    reasons: Dict[int, list]
    dataset: str
    available: bool
    summary: dict = field(default_factory=dict)

    def neighbor(self, index: int) -> Tuple[Optional[int], Optional[int]]:
        return self.neighbors.get(int(index), (None, None))

    def pair_available(self, index: int) -> bool:
        previous, following = self.neighbor(index)
        return previous is not None and following is not None

    def fingerprint(self) -> str:
        payload = {
            "dataset": self.dataset,
            "available": self.available,
            "meta": {str(i): asdict(m) for i, m in sorted(self.meta.items())},
            "neighbors": {str(i): list(v) for i, v in sorted(self.neighbors.items())},
            "reasons": {str(i): list(v) for i, v in sorted(self.reasons.items())},
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    @classmethod
    def from_meta(cls, meta: Dict[int, SliceMeta], dataset: str = "synthetic",
                  summary: Optional[dict] = None) -> "SliceIndex":
        neighbors, reasons = _build_neighbors(meta)
        available = any(prev is not None and nxt is not None for prev, nxt in neighbors.values())
        result = cls(meta=dict(meta), neighbors=neighbors, reasons=reasons,
                     dataset=dataset, available=available)
        result.summary = summary or _summary(result)
        return result


def _spacing_compatible(a: SliceMeta, b: SliceMeta) -> bool:
    if a.spacing_z is None or b.spacing_z is None:
        return False
    scale = max(abs(a.spacing_z), abs(b.spacing_z), 1e-6)
    return abs(a.spacing_z - b.spacing_z) <= max(1e-3, .05 * scale)


def _build_neighbors(meta: Dict[int, SliceMeta]):
    groups = {}
    for index, item in meta.items():
        groups.setdefault((item.case_id, item.series_id, item.sequence, item.split), []).append(index)
    neighbors, reasons = {}, {}
    for group in groups.values():
        by_z = {meta[index].slice_z: index for index in group if meta[index].slice_z >= 0}
        for index in group:
            item = meta[index]
            issues = []
            if item.slice_z < 0:
                issues.append("slice_z_unavailable")
            if not item.brain_valid:
                issues.append("empty_brain_slice")
            previous = by_z.get(item.slice_z - 1) if item.slice_z >= 0 else None
            following = by_z.get(item.slice_z + 1) if item.slice_z >= 0 else None
            for label, other in (("prev", previous), ("next", following)):
                if other is None:
                    issues.append(f"{label}_missing")
                    continue
                other_meta = meta[other]
                if other_meta.orientation != item.orientation:
                    issues.append(f"{label}_orientation_mismatch")
                if not _spacing_compatible(item, other_meta):
                    issues.append(f"{label}_spacing_mismatch")
                if not other_meta.brain_valid:
                    issues.append(f"{label}_empty_brain")
            usable = (previous is not None and following is not None
                      and not any(issue.endswith(("missing", "mismatch", "empty_brain"))
                                  or issue == "empty_brain_slice" for issue in issues))
            neighbors[index] = (previous, following) if usable else (None, None)
            reasons[index] = sorted(set(issues))
    return neighbors, reasons


def _summary(index: SliceIndex) -> dict:
    counts = {}
    for issues in index.reasons.values():
        for issue in issues:
            counts[issue] = counts.get(issue, 0) + 1
    return {
        "dataset": index.dataset,
        "available": index.available,
        "n_slices": len(index.meta),
        "n_with_prev": sum(1 for prev, _ in index.neighbors.values() if prev is not None),
        "n_with_next": sum(1 for _, nxt in index.neighbors.values() if nxt is not None),
        "n_with_pair": sum(1 for prev, nxt in index.neighbors.values() if prev is not None and nxt is not None),
        "reason_counts": counts,
        "fingerprint": index.fingerprint(),
    }


def _source_index(dataset, split: str) -> SliceIndex:
    records = getattr(dataset, "records", [])
    meta = {}
    for index, record in enumerate(records):
        slice_z = int(record.get("slice_z", -1) if record.get("slice_z") is not None else -1)
        case_id = str(record.get("case_id") or "")
        meta[index] = SliceMeta(
            sample_id=str(record.get("sample_id", index)),
            case_id=case_id,
            series_id=str(record.get("series_id") or case_id),
            sequence=sequence_key(record.get("sequence") or record.get("modality")),
            slice_z=slice_z,
            split=split,
            spacing_z=record.get("spacing_z"),
            orientation=record.get("orientation"),
            brain_valid=True,
        )
    index = SliceIndex.from_meta(meta, dataset=split)
    if not any(item.slice_z >= 0 for item in meta.values()):
        index.available = False
        for key in index.reasons:
            index.reasons[key] = sorted(set(index.reasons[key] + ["source_png_slice_z_unavailable"]))
        index.neighbors = {key: (None, None) for key in index.neighbors}
        index.summary = _summary(index)
    return index


def _external_index(dataset, split: str) -> SliceIndex:
    protocol = dataset.protocol
    pairs = dataset.pairs
    meta = {}
    valid_brain = {}
    headers = {}
    for case_index, record in enumerate(protocol["cases"]):
        image = nib.as_closest_canonical(nib.load(record["image"]))
        affine = image.affine
        spacing_z = float(nib.affines.voxel_sizes(affine)[2])
        orientation = "".join(nib.aff2axcodes(affine))
        headers[case_index] = (spacing_z, orientation, image.shape[2])
        brain_obj = nib.as_closest_canonical(nib.load(record["brain_mask"]))
        if brain_obj.shape != image.shape or not np.allclose(brain_obj.affine, affine, atol=1e-4):
            raise ValueError(f"{record['case_id']} canonical brain mask is not aligned with the canonical image")
        brain = np.asarray(brain_obj.dataobj) > 0
        for z in range(image.shape[2]):
            valid_brain[(case_index, z)] = bool(brain[:, :, z].any())
    for index, (case_index, z) in enumerate(pairs):
        record = protocol["cases"][case_index]
        spacing_z, orientation, _ = headers[case_index]
        case_id = str(record.get("case_id") or case_index)
        meta[index] = SliceMeta(
            sample_id=f"{case_id}/z{z:04d}",
            case_id=case_id,
            series_id=str(record.get("series_id") or record.get("series") or case_id),
            sequence=sequence_key(record.get("sequence") or protocol.get("sequence")),
            slice_z=int(z),
            split=split,
            spacing_z=spacing_z,
            orientation=orientation,
            brain_valid=valid_brain.get((case_index, z), False),
        )
    index = SliceIndex.from_meta(meta, dataset=split)
    return index


def build_slice_index(dataset, split: str) -> SliceIndex:
    """Build a trusted index; source PNGs deliberately return unavailable."""
    if hasattr(dataset, "protocol") and hasattr(dataset, "pairs"):
        return _external_index(dataset, split)
    return _source_index(dataset, split)
