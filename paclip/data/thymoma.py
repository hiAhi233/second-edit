"""胸腺瘤 ROI 数据加载：3D CT 体积 + 手动勾画的肿瘤 ROI mask。

数据目录结构（本院 233 例胸腺瘤，全异常，均有 ROI）：
    root/<范围>/
        XXX000-origin.nii.gz   平扫图像
        XXX000.nii.gz          平扫 ROI（0/1）
        XXX020-origin.nii.gz   增强图像
        XXX020.nii.gz          增强 ROI（0/1）
    编号：前 3 位为患者编号，后 3 位 000=平扫、020=增强。

无正常（无病灶）病例 → 只用于训练局部对齐分支：
病灶内 patch ↔ 异常锚点、病灶外 patch ↔ 正常锚点（见 losses.local_alignment_loss）。
"""

import glob
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from paclip.data.datasets import _gray2clip
from paclip.data.supervision import mask_fields

_ORIGIN_SUFFIX = "-origin.nii.gz"


def _window_slice(s: np.ndarray, wl: float = 40.0, ww: float = 400.0) -> np.ndarray:
    """CT HU 纵隔窗归一化到 [0,1]：lo=wl-ww/2，hi=wl+ww/2。

    胸腺瘤位于前纵隔，放射科标准纵隔窗 WW=400 / WL=40（文档建议 WW 350-400 / WL 40-50）。
    """
    lo, hi = wl - ww / 2.0, wl + ww / 2.0
    return np.clip((s - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _downsample_mask(m: np.ndarray, grid: int) -> np.ndarray:
    """二值 mask (H,W) → (grid,grid)，每 cell 取区域内最大值（保小病灶）。

    BILINEAR+阈值 会丢失面积小于一个 cell 的小肿瘤，改用分块 max 保真。
    """
    H, W = m.shape
    out = np.zeros((grid, grid), dtype=np.float32)
    for i in range(grid):
        h0, h1 = int(i * H / grid), int((i + 1) * H / grid)
        for j in range(grid):
            w0, w1 = int(j * W / grid), int((j + 1) * W / grid)
            blk = m[h0:h1, w0:w1]
            out[i, j] = float(blk.max()) if blk.size else 0.0
    return out


def scan_roi_volumes(root: str) -> List[Tuple[str, str]]:
    """递归扫描 root 下所有 `*-origin.nii.gz`，配对同名（去 -origin）ROI mask。

    Returns:
        [(img_path, mask_path), ...]，按路径排序。
    """
    vols: List[Tuple[str, str]] = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.endswith(_ORIGIN_SUFFIX):
                base = f[: -len(_ORIGIN_SUFFIX)]  # "003000"
                mask = os.path.join(dirpath, base + ".nii.gz")
                if os.path.isfile(mask):
                    vols.append((os.path.join(dirpath, f), mask))
    return sorted(vols)


def make_splits(
    root: str,
    seed: int = 0,
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> dict:
    """按**患者编号**（文件名前 3 位）切 train/val/test，避免同病例切片泄漏。

    Returns:
        {"train": [(img, mask), ...], "val": [...], "test": [...]}
    """
    vols = scan_roi_volumes(root)
    cases = sorted({os.path.basename(img)[:3] for img, _ in vols})
    rng = np.random.RandomState(seed)
    rng.shuffle(cases)

    n = len(cases)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    train_cases = set(cases[:n_train])
    val_cases = set(cases[n_train : n_train + n_val])
    test_cases = set(cases[n_train + n_val :])

    splits: dict = {}
    for name, case_set in [
        ("train", train_cases),
        ("val", val_cases),
        ("test", test_cases),
    ]:
        splits[name] = [
            (img, m) for img, m in vols if os.path.basename(img)[:3] in case_set
        ]
    return splits


class ThymomaROIDataset(Dataset):
    """胸腺瘤 3D CT 体积 + ROI mask，按病灶切片采样（全异常）。

    每次取一个含病灶的轴向切片（优先小面积和病灶边界层），
    保留原始掩码，训练时按模型输出网格生成监督。
    """

    def __init__(
        self,
        root: str,
        volumes: Optional[List[Tuple[str, str]]] = None,
        image_size: int = 224,
        grid: int = 14,
        window: Tuple[float, float] = (40.0, 400.0),
        topk: int = 8,
    ):
        self.image_size = image_size
        self.grid = grid
        self.wl, self.ww = window
        self.topk = topk
        self.volumes = volumes if volumes is not None else scan_roi_volumes(root)

    def __len__(self) -> int:
        return len(self.volumes)

    def _pick_lesion_slice(self, msk: np.ndarray) -> int:
        """按病灶面积反平方根采样，并提高起止层权重。"""
        sums = msk.sum(axis=(0, 1))  # (D,)
        idxs = np.flatnonzero(sums > 0)
        if len(idxs) == 0:
            return msk.shape[2] // 2
        weights = 1.0 / np.sqrt(sums[idxs])
        weights[[0, -1]] *= 2.0
        return int(np.random.choice(idxs, p=weights / weights.sum()))

    def __getitem__(self, idx: int) -> dict:
        import nibabel as nib

        img_path, mask_path = self.volumes[idx]
        vol = np.asarray(nib.load(img_path).dataobj, dtype=np.float32)  # (H, W, D)
        msk = np.asarray(nib.load(mask_path).dataobj, dtype=np.float32)  # (H, W, D) 0/1

        z = self._pick_lesion_slice(msk)
        s = _window_slice(vol[:, :, z], self.wl, self.ww)
        m = msk[:, :, z]

        gray = Image.fromarray((s * 255).astype(np.uint8)).resize(
            (self.image_size, self.image_size), Image.BILINEAR
        )
        image = _gray2clip(np.asarray(gray, dtype=np.float32) / 255.0)

        return {
            "image": image,
            **mask_fields(m, self.grid),
            "label": torch.tensor(1, dtype=torch.long),
            "path": img_path,
        }


class ThymomaSliceDataset(Dataset):
    """从预提取的 .npz 病灶切片读取（快速训练/评估）。"""

    def __init__(self, npz_dir: str, files: Optional[List[str]] = None, grid: int = 14):
        self.grid = grid
        self.files = (
            files
            if files is not None
            else sorted(glob.glob(os.path.join(npz_dir, "*.npz")))
        )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        d = np.load(self.files[idx])
        x = d["image"].astype(np.float32)  # (224,224) [0,1]
        # Old coarse-only caches cannot recover the original lesion occupancy.
        if "mask_high" in d.files:
            m = d["mask_high"]
        elif "mask224" in d.files:
            m = d["mask224"]
        else:
            raise ValueError(f"{self.files[idx]} 只有粗网格掩码，请重新提取带 mask_high 的切片缓存")
        return {
            "image": _gray2clip(x),
            **mask_fields(m, self.grid),
            "label": torch.tensor(1, dtype=torch.long),
            "path": self.files[idx],
        }


def make_slice_splits(
    npz_dir: str,
    seed: int = 0,
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> dict:
    """按病例号（文件名前 3 位）切分预提取切片，与 make_splits 同 seed 同病例集合。"""
    files = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))
    cases = sorted({os.path.basename(f)[:3] for f in files})
    rng = np.random.RandomState(seed)
    rng.shuffle(cases)

    n = len(cases)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    sets = {
        "train": set(cases[:n_train]),
        "val": set(cases[n_train : n_train + n_val]),
        "test": set(cases[n_train + n_val :]),
    }
    return {
        name: [f for f in files if os.path.basename(f)[:3] in cset]
        for name, cset in sets.items()
    }


def case_ids(files: List[str]) -> set:
    """文件名前 3 位 = 患者编号（见模块 docstring 的编号约定）。"""
    return {os.path.basename(f)[:3] for f in files}


def sample_support(files: List[str], k: Optional[int], seed: int = 0) -> List[str]:
    """从给定划分里按病例抽 k 张切片；k=None 或 k≥len(files) 表示全用。

    按病例抽（而不是按切片）—— 同一病例的切片高度相关，混着抽会让"5 张"实际只
    覆盖 2~3 个病例，把有效样本数虚高。先每病例取 1 张轮转，保证病例覆盖尽量广。

    small-sample 训练与对照实验（paclip.cli.thymoma_ablation.py / paclip.cli.thymoma_roi.py --n-support）共用，
    放在这里而不是各自的脚本里，免得两份实现漂移。
    """
    if k is None or k >= len(files):
        return list(files)
    rng = np.random.RandomState(seed)
    by_case: dict = {}
    for f in files:
        by_case.setdefault(os.path.basename(f)[:3], []).append(f)
    cases = sorted(by_case)
    rng.shuffle(cases)
    picked: List[str] = []
    for rnd in range(max(len(v) for v in by_case.values())):
        for c in cases:
            if len(picked) >= k:
                return picked
            if rnd < len(by_case[c]):
                picked.append(by_case[c][rnd])
    return picked[:k]
