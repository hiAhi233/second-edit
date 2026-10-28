"""解剖 ROI 掩码与异常图门控。

动机：异常图对「哪里是脑组织」一无所知，于是头皮、颅骨、背景都会被判成异常。
实测（多尺度模型，test 820 张）：
    正常切片高分区中 53.5% 落在脑 ROI 之外 —— 这就是热力图「乱指、红一大片」的主因。
    异常切片 ROI 内被判红的像素里，只有 54.7% 是真病灶。

门控的关键陷阱：amap 整体为负（均值约 −0.4），把 ROI 外**置 0 反而高于病灶**，
会把排序彻底搞坏（pxAUROC 0.957 → 0.569）。必须压到全局最小值**以下**。

ROI 由图像强度阈值给出（paclip 的 tissue_mask_from_image 思路），不需要病灶标注，
部署时可用。
"""

from typing import Optional

import numpy as np
import torch

# paclip 用的阈值：0~255 下的 20
DEFAULT_THR = 20.0 / 255.0


def roi_from_gray(gray01: np.ndarray, grid: int = 14,
                  thr: float = DEFAULT_THR, min_patches: int = 5) -> np.ndarray:
    """灰度图（值域 [0,1]）→ (grid, grid) 布尔 ROI。

    用 NEAREST 缩放到 patch 网格，与 paclip 一致：只要该 patch 覆盖的像素里有组织
    就算 ROI 内，宁可多留不可漏掉边缘病灶。

    Args:
        gray01: (H, W) 或 (grid, grid) 的灰度数组，值域 [0, 1]。
        grid: patch 网格边长。
        thr: 组织/背景分界（值域 [0,1] 下默认 0.078）。
        min_patches: ROI 小于该格数时视为退化，返回全 True（不门控）。
    """
    from PIL import Image

    g = np.asarray(gray01, dtype=np.float32)
    if g.max() > 1.0:
        g = g / 255.0
    if g.shape != (grid, grid):
        g = np.asarray(Image.fromarray((g * 255).astype(np.uint8)).resize(
            (grid, grid), Image.NEAREST), dtype=np.float32) / 255.0
    roi = g > thr
    if roi.sum() < min_patches:
        roi = np.ones((grid, grid), dtype=bool)
    return roi


def roi_from_tensor(gray01: torch.Tensor, grid: int = 14,
                    thr: float = DEFAULT_THR) -> torch.Tensor:
    """批量化版本，给 Dataset 用。gray01: (B, H, W) 值域 [0,1] → (B, grid, grid) bool。"""
    import torch.nn.functional as F

    x = gray01.unsqueeze(1).float()
    small = F.interpolate(x, size=(grid, grid), mode="nearest")
    roi = small.squeeze(1) > thr
    # 退化的 ROI 整张放开
    degenerate = roi.flatten(1).sum(dim=1) < 5
    roi[degenerate] = True
    return roi


def gate_by_roi(amap: np.ndarray, roi: np.ndarray,
                floor: Optional[float] = None) -> np.ndarray:
    """把 ROI 外的异常分数压到 floor（默认全局最小值 − 1）。

    压到「全局最小值以下」而不是 0：amap 整体为负，置 0 会让背景排到病灶前面。
    这样 ROI 内的相对排序完全不变，ROI 外则永远不会被任何阈值选中。
    """
    a = np.asarray(amap, dtype=np.float32)
    r = np.asarray(roi, dtype=bool)
    if floor is None:
        floor = float(a.min()) - 1.0
    return np.where(r, a, floor)
