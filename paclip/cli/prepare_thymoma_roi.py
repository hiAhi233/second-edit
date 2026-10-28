"""预处理：把胸腺瘤 ROI 3D 体积的病灶切片一次性提取成 .npz，供训练/评估快速读取。

避免训练时每个 epoch 重复解压 466 个 .nii.gz（约 90s/epoch 的纯 I/O）。
每个体积优先抽取小面积及起止边界病灶切片，存：
    image   (224,224) 纵隔窗灰度 [0,1]
    mask    (14,14)   病灶 mask（分块 max 下采样，训练局部对齐用）
    mask224 (224,224) 病灶 mask（出图画 GT 轮廓用）
    mask_high (H,W) 原始二值掩码，供动态生成 soft/hard 监督
"""

import argparse
import os
import sys

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from paclip.data.thymoma import (
    _downsample_mask,
    _window_slice,
    scan_roi_volumes,
)
from paclip.data.datasets import choose_fixed_zs

ROOT = "data/本院图像和ROI_能共享_2025-11-18整理/全部图像及ROI"
OUT = "thymoma_slices"
TOPK = 8


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=ROOT, help="原始图像与 ROI 所在目录")
    parser.add_argument("--out", default=OUT, help="输出 npz 切片目录")
    parser.add_argument("--top-k", type=int, default=TOPK, help="每个体积最多提取的病灶切片数（优先小病灶和起止边界）")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k 必须为正数")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    os.makedirs(args.out, exist_ok=True)
    vols = scan_roi_volumes(args.root)
    n_slices = 0
    rng = np.random.RandomState(0)
    for img_path, mask_path in vols:
        vol = np.asarray(nib.load(img_path).dataobj, dtype=np.float32)
        msk = np.asarray(nib.load(mask_path).dataobj, dtype=np.float32)
        sums = msk.sum(axis=(0, 1))
        count = min(args.top_k, int((sums > 0).sum()))
        idxs, _ = choose_fixed_zs(msk.shape[2], msk, count, rng, 1) if count else ([], "empty")

        base = os.path.basename(img_path).replace("-origin.nii.gz", "")
        for z in idxs:
            s = _window_slice(vol[:, :, z])
            m = msk[:, :, z]
            gray = Image.fromarray((s * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)
            mask14 = _downsample_mask(m, 14)
            # Max pooling retains small lesions for display/evaluation too.
            native = torch.from_numpy((m > 0).astype(np.float32))[None, None]
            mask224 = F.adaptive_max_pool2d(native, (224, 224))[0, 0].numpy()

            np.savez_compressed(
                os.path.join(args.out, f"{base}_z{z}.npz"),
                image=np.asarray(gray, dtype=np.float32) / 255.0,
                mask=mask14,
                mask224=mask224,
                mask_high=(m > 0).astype(np.uint8),
            )
            n_slices += 1
    print(f"提取切片数: {n_slices}，输出到 {args.out}/")


if __name__ == "__main__":
    main()
