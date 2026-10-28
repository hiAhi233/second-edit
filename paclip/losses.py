"""损失函数（brain MRI 异常检测）。

对应文档"文本侧损失函数"（m 为允许的最大相似度）：

    L_text = Σ_l  max(0, cos(t_a^l, t_n^l) − m)      # normal/abnormal 锚点分离

全局对齐（CLS 图像级判断）：
    L_global = CE( cls_logits, y )                   # normal vs abnormal

局部对齐（病灶内 patch ↔ 异常锚点、病灶外 patch ↔ 正常锚点）：
    L_local  = FocalBCE + SoftDice + 0.1 * Boundary  # patch 病灶面积占比

2026-09-28 新增（见 docx 改动 7 的机制定位）：
    L_text 加下界 m_lo，阻止 cos(t_a,t_n) 被推到 −1（对跖点）；
    L_div  惩罚三层差向量 d_l = t_a^l − t_n^l 的方向对齐。
    原因：anomaly_map_l = f_patch·d_l，融合是 softmax 凸组合，故三个 d_l 平行 ⟺
    融合图恒等于单层图 ⟺ 三层机制形同虚设。原版单边 hinge 会把 cos 推到 −1，
    于是 d_l ≈ 2·t_a^l，d_l 方向被锁成 t_a 的方向，跨层 t_a 相似度直接决定三层 map
    相同（实测训练后 cos(d)=0.96、三层 map 相关 0.994）。两项新损失默认关闭，
    以保持与旧实验的可比性。
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from paclip.data.supervision import pool_mask_targets


def text_separation_loss(
    anchors: Dict[str, Dict[str, torch.Tensor]],
    margin: float = 0.3,
    margin_lo: Optional[float] = None,
) -> torch.Tensor:
    """文本侧分离损失：惩罚每层 normal/abnormal 锚点相似度超出 [margin_lo, margin]。

    margin_lo=None 时退化为文档原版的**单边** hinge relu(cos − margin)：只罚"太像"、
    不罚"太不像"，无下界，优化器会把 cos 一路推到 −1。这是三层塌成一层的第一环
    （机制见 docx 改动 7 第 (3) 条）。给定下界即拆掉这条坍缩路径。
    """
    losses = []
    for lvl in anchors:
        cos = F.cosine_similarity(anchors[lvl]["abnormal"], anchors[lvl]["normal"], dim=0)
        upper = F.relu(cos - margin)
        if margin_lo is None:
            losses.append(upper)
        else:
            losses.append(upper + F.relu(margin_lo - cos))
    if not losses:
        return torch.zeros((), device="cpu")
    return torch.stack(losses).mean()


def d_diversity_loss(
    anchors: Dict[str, Dict[str, torch.Tensor]],
    margin_d: float = 0.7,
) -> torch.Tensor:
    """三层差向量 d_l = t_a^l − t_n^l 的方向多样性。

    anomaly_map 只由 d_l 决定，而融合是 softmax 凸组合，所以三个 d_l 平行 ⟺
    融合图恒等于任一单层图 ⟺ 融合权重成哑参数。**没有任何一项损失要求三层分歧**，
    不显式要求它就不会发生——实测训练后 cos(d) 均值 0.96、三层 map 相关 0.994。
    """
    ds = []
    for lvl in anchors:
        d = anchors[lvl]["abnormal"] - anchors[lvl]["normal"]
        ds.append(F.normalize(d, dim=0, eps=1e-6))
    if len(ds) < 2:
        return torch.zeros((), device=ds[0].device)
    losses = [
        F.relu(F.cosine_similarity(ds[i], ds[j], dim=0) - margin_d)
        for i in range(len(ds))
        for j in range(i + 1, len(ds))
    ]
    return torch.stack(losses).mean()


def global_alignment_loss(cls_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """全局对齐：CLS 对 normal/abnormal 的交叉熵。

    Args:
        cls_logits: (B, 2) 图像级 logits（[normal, abnormal]）。
        labels: (B,) 图像级标签，0=正常，1=异常。
    """
    return F.cross_entropy(cls_logits, labels.long())


def focal_bce(logits, target, valid=None, gamma=2.0, alpha_positive=0.75):
    """Soft-label focal BCE, without potentially unbounded pos_weight."""
    prob = logits.sigmoid()
    positive = -alpha_positive * target * (1 - prob).pow(gamma) * F.logsigmoid(logits)
    negative = -(1 - alpha_positive) * (1 - target) * prob.pow(gamma) * F.logsigmoid(-logits)
    weight = torch.ones_like(target) if valid is None else valid.to(target.dtype)
    return ((positive + negative) * weight).sum() / weight.sum().clamp_min(1)


def soft_dice_loss(prob, target, valid=None, eps=1e-6):
    weight = torch.ones_like(target) if valid is None else valid.to(target.dtype)
    pred, truth = prob * weight, target * weight
    numerator = 2 * (pred * truth).flatten(1).sum(1)
    denominator = pred.flatten(1).sum(1) + truth.flatten(1).sum(1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    active = weight.flatten(1).sum(1) > 0
    return (loss * active).sum() / active.sum().clamp_min(1)


def boundary_loss(prob, target, valid=None):
    """L1 discrepancy of horizontal/vertical soft-mask differences.

    Use only edges whose two endpoints are inside ROI; handles 1x1 grids.
    """
    valid = torch.ones_like(target, dtype=torch.bool) if valid is None else valid.bool()
    total = prob.sum() * 0
    count = prob.new_zeros(())
    for axis in (-2, -1):
        edge_valid = valid.narrow(axis, 1, valid.size(axis) - 1) & valid.narrow(axis, 0, valid.size(axis) - 1)
        error = (torch.diff(prob, dim=axis) - torch.diff(target, dim=axis)).abs()
        total = total + (error * edge_valid).sum()
        count = count + edge_valid.sum()
    return total / count.clamp_min(1)


def local_alignment_loss(
    patch_logits: torch.Tensor,
    masks: torch.Tensor,
    roi: Optional[torch.Tensor] = None,
    gamma: float = 2.0,
    alpha_positive: float = 0.75,
    boundary_weight: float = 0.1,
    recall_weight: float = 0.1,
) -> torch.Tensor:
    """Focal + Dice + boundary supervision, pooled at the actual (h,w).

    masks may be native-resolution tensors or a list with differing shapes.
    Hard-positive recall pressure defaults to 0.1 to implement document 1.1.
    Set recall_weight=0 explicitly for the original three-term ablation.
    """
    if patch_logits.ndim != 4 or patch_logits.shape[1] != 2:
        raise ValueError("patch_logits must have shape (B, 2, h, w)")
    # Float32 keeps sigmoid/log-sigmoid stable under mixed precision.
    logits = patch_logits[:, 1].float() - patch_logits[:, 0].float()
    return binary_local_loss(logits, masks, roi, gamma, alpha_positive, boundary_weight, recall_weight)


def binary_local_loss(logits, masks, roi=None, gamma=2., alpha_positive=.75,
                      boundary_weight=.1, recall_weight=.1):
    """Shared B,H,W binary-logit implementation; legacy aggregation is unchanged.

    The new trainer invokes this per image before source-group averaging.
    """
    if logits.ndim != 3:
        raise ValueError("binary logits must have shape (B,H,W)")
    logits = logits.float()
    target, hard = pool_mask_targets(masks, logits.shape[-2:], logits.device)
    if target.shape[0] != logits.shape[0]:
        raise ValueError("Mask batch size does not match logits")
    valid = None
    if roi is not None:
        valid = F.adaptive_max_pool2d(roi.to(logits.device).float()[:, None], logits.shape[-2:])[:, 0] > 0
    prob = logits.sigmoid()
    loss = focal_bce(logits, target, valid, gamma, alpha_positive)
    loss = loss + soft_dice_loss(prob, target, valid) + boundary_weight * boundary_loss(prob, target, valid)
    if recall_weight:
        selected = hard if valid is None else hard * valid
        loss = loss + recall_weight * ((1 - prob) * selected).sum() / selected.sum().clamp_min(1)
    return loss


def per_level_local_loss(
    patch_logits_per_level: torch.Tensor,
    masks: torch.Tensor,
    roi: Optional[torch.Tensor] = None,
    **local_options,
) -> torch.Tensor:
    """每一层各自的定位损失：要求第 l 层的 patch logits **自己**就能定位。

    动机（docx 改动 11 实测）：d_diversity_loss 是纯"要不一样"的目标，没有任何质量项
    约束它往哪个方向分歧，于是它把三层推成"不一样地差"——只开②时 L1 的 Dice 从 0.6215
    掉到 0.4742。融合（softmax 凸组合）能平均掉一部分独立误差（集成增益 +0.006 → +0.048），
    但补不平单层的损失，净效果仍是掉点。

    只监督融合结果时，单层可以烂得任意 —— 只要加权后是好的。这一项给每层各自加定位
    监督，把②的"不一样"约束在"各自都还准"的基础上。与②在梯度上是对抗的（质量要把
    三层拉向同一张正确的图，多样性要把它们推开），平衡点是这项超参要扫的东西。

    Args:
        patch_logits_per_level: (B, L, 2, h, w) 逐层 patch logits。
        masks: (B, h, w) 病灶掩码（0/1）。
        roi: (B, h, w) 解剖 ROI（布尔），可选。
    """
    n_levels = patch_logits_per_level.shape[1]
    return torch.stack([
        local_alignment_loss(patch_logits_per_level[:, l], masks, roi, **local_options)
        for l in range(n_levels)
    ]).mean()


class TotalLoss(nn.Module):
    """总损失 = w_text·文本分离 + w_global·全局对齐 + w_local·局部对齐
    + w_div·多层多样性 + w_level·逐层定位。

    margin_lo / w_div / w_level 默认 None / 0.0 / 0.0；局部项采用 E1 组合损失。
    """

    def __init__(
        self,
        margin: float = 0.3,
        w_text: float = 1.0,
        w_global: float = 1.0,
        w_local: float = 1.0,
        margin_lo: Optional[float] = None,
        margin_d: float = 0.7,
        w_div: float = 0.0,
        w_level: float = 0.0,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.75,
        boundary_weight: float = 0.1,
        recall_weight: float = 0.1,
    ):
        super().__init__()
        self.margin = margin
        self.w_text = w_text
        self.w_global = w_global
        self.w_local = w_local
        self.margin_lo = margin_lo
        self.margin_d = margin_d
        self.w_div = w_div
        self.w_level = w_level
        self.local_options = dict(gamma=focal_gamma, alpha_positive=focal_alpha,
                                  boundary_weight=boundary_weight, recall_weight=recall_weight)

    def forward(
        self,
        anchors: Dict[str, Dict[str, torch.Tensor]],
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
        roi: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        loss_text = text_separation_loss(anchors, self.margin, self.margin_lo)
        loss_global = global_alignment_loss(outputs["cls_logits"], labels)

        loss_dict = {"text": loss_text, "global": loss_global}

        if masks is not None:
            loss_local = local_alignment_loss(outputs["patch_logits"], masks, roi, **self.local_options)
            loss_dict["local"] = loss_local
        else:
            loss_local = torch.zeros((), device=loss_text.device)

        loss_div = d_diversity_loss(anchors, self.margin_d)
        loss_dict["div"] = loss_div

        # 逐层定位：约束②的分歧方向（见 per_level_local_loss 的 docstring）
        loss_level = torch.zeros((), device=loss_text.device)
        if self.w_level and masks is not None and "patch_logits_per_level" in outputs:
            loss_level = per_level_local_loss(
                outputs["patch_logits_per_level"], masks, roi, **self.local_options
            )
            loss_dict["level"] = loss_level

        total = (
            self.w_text * loss_text
            + self.w_global * loss_global
            + self.w_local * loss_local
            + self.w_div * loss_div
            + self.w_level * loss_level
        )
        loss_dict["total"] = total
        return loss_dict
