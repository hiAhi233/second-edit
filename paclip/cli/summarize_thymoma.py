"""Audit exported held-out predictions, report source subgroups, index heatmaps.

This script never trains, chooses thresholds, or modifies predictions.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from sklearn.metrics import average_precision_score, roc_auc_score


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump_csv(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metrics(rows, name, threshold):
    labels = np.array([r["label"] for r in rows], dtype=bool)
    scores = np.array([r["score_fused"] for r in rows])
    predicted = scores >= threshold
    tp, fp = int((predicted & labels).sum()), int((predicted & ~labels).sum())
    fn, tn = int((~predicted & labels).sum()), int((~predicted & ~labels).sum())
    intersection = sum(r["intersection_pixels"] for r in rows)
    pred_area = sum(r["predicted_pixels"] for r in rows)
    gt_area = sum(r["gt_pixels"] for r in rows)
    positives = [r for r in rows if r["label"]]
    normals = [r for r in rows if not r["label"]]
    both_classes = len(np.unique(labels)) == 2
    return {"group": name, "slices": len(rows), "positive_slices": len(positives),
            "negative_slices": len(normals), "unique_patients": len({r["case_id"] for r in rows}),
            "image_auroc": float(roc_auc_score(labels, scores)) if both_classes else None,
            "image_ap": float(average_precision_score(labels, scores)) if both_classes else None,
            "image_f1": 2 * tp / max(2 * tp + fp + fn, 1), "image_accuracy": (tp + tn) / max(len(rows), 1),
            "sensitivity": tp / len(positives) if positives else None,
            "specificity": tn / len(normals) if normals else None,
            "image_tp": tp, "image_fp": fp, "image_fn": fn, "image_tn": tn,
            "dice_micro_224": 2 * intersection / (pred_area + gt_area) if pred_area + gt_area else 1.0,
            "iou_micro_224": intersection / (pred_area + gt_area - intersection) if pred_area + gt_area - intersection else 1.0,
            "positive_dice_macro_224": float(np.mean([r["dice_224"] for r in positives])) if positives else None,
            "normal_false_positive_pixel_fraction": float(np.mean([r["predicted_pixel_fraction"] for r in normals])) if normals else None,
            "positive_zero_overlap_fraction": float(np.mean([r["intersection_pixels"] == 0 for r in positives])) if positives else None,
            "image_threshold_from_validation": float(threshold)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/thymoma_nlst_training_20261021"))
    args = parser.parse_args()
    run = args.run_dir.resolve()
    cfg, official = read_json(run / "run_config.json"), read_json(run / "metrics.json")
    assert cfg["status"] == "visualized" and cfg["completed_steps"] == 2700
    assert cfg["case_disjoint_verified"] and cfg["task"] == "thymoma"
    assert official["postprocess"]["thresholds"]["image_threshold_source"] == "val"
    assert official["postprocess"]["thresholds"]["pixel_threshold_source"] == "val"
    source = {r["sample_id"]: r for r in (json.loads(line) for line in (args.data_dir / "splits.jsonl").read_text(encoding="utf-8").splitlines())}
    folder = run / "evaluations" / cfg["evaluation_id"]
    predictions = [json.loads(line) for line in (folder / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    threshold = official["postprocess"]["thresholds"]["image_threshold"]
    rows, pixel_scores, pixel_labels = [], [], []
    for record in predictions:
        original = source[record["sample_id"]]
        assert original["split"] == "test" and record["label"] == original["label"]
        assert record["gt_available"] and record["mask_status"] != "unavailable"
        with np.load(folder / "arrays" / record["array"], allow_pickle=False) as arrays:
            pred, gt = arrays["pred_mask"] > 0, arrays["gt_mask"] > 0
            assert pred.shape == gt.shape == (224, 224)
            assert bool(gt.any()) == bool(record["label"])
            assert np.isfinite(arrays["raw_map"]).all() and np.isfinite(arrays["score_map_high"]).all()
            inter, pa, ga = int((pred & gt).sum()), int(pred.sum()), int(gt.sum())
            local_map = arrays["score_map_high"]
            pixel_scores.append(np.asarray(arrays["score_map_refined"], dtype=np.float32).reshape(-1).copy())
            pixel_labels.append(gt.reshape(-1).copy())
            inside = float(local_map[gt].mean()) if ga else None
            outside = float(local_map[~gt].mean())
            peak_yx = np.unravel_index(int(np.argmax(local_map)), local_map.shape)
            peak_in_gt = bool(gt[peak_yx]) if ga else None
        assert bool(record["pred_label"]) == bool(record["score_fused"] >= threshold)
        rows.append({"sample_id": record["sample_id"], "case_id": record["case_id"], "source": original["source"],
                     "label": record["label"], "pred_label": record["pred_label"],
                     "score_global": record["score_global"], "score_local": record["score_local"], "score_fused": record["score_fused"],
                     "intersection_pixels": inter, "predicted_pixels": pa, "gt_pixels": ga,
                     "dice_224": 2 * inter / (pa + ga) if pa + ga else 1.0,
                     "iou_224": inter / (pa + ga - inter) if pa + ga - inter else 1.0,
                     "predicted_pixel_fraction": pa / pred.size,
                     "mean_score_in_lesion": inside, "mean_score_outside_lesion": outside,
                     "heatmap_peak_in_lesion": peak_in_gt,
                     "heatmap_row": "heatmaps/figures/rows/" + record["file_id"] + ".png"})
    assert len(rows) == 585 and len({r["sample_id"] for r in rows}) == 585
    flat_scores, flat_labels = np.concatenate(pixel_scores), np.concatenate(pixel_labels)
    highres = {"pixel_auroc_224": float(roc_auc_score(flat_labels, flat_scores)),
               "pixel_ap_224": float(average_precision_score(flat_labels, flat_scores)),
               "score": "score_map_refined", "annotation": "224x224 original slice annotation", "pixels": int(flat_labels.size)}
    (run / "highres_pixel_metrics.json").write_text(json.dumps(highres, indent=2), encoding="utf-8")
    del pixel_scores, pixel_labels, flat_scores, flat_labels
    private = [r for r in rows if r["source"] != "NLST_screening_negative"]
    external = [r for r in rows if r["source"] != "private_lesion_free_slice"]
    groups = [metrics(rows, "all_test", threshold), metrics(external, "private_lesion_vs_NLST_control", threshold),
              metrics(private, "within_private_slice_level", threshold)]
    for name in ["NLST_screening_negative", "private_lesion_free_slice", "private_lesion"]:
        groups.append(metrics([r for r in rows if r["source"] == name], name, threshold))
    assert np.isclose(groups[0]["image_auroc"], official["postprocess"]["image"]["auroc"])
    assert np.isclose(groups[0]["dice_micro_224"], official["postprocess"]["pixel_highres"]["dice_micro"])
    dump_csv(run / "per_slice_metrics.csv", rows)
    dump_csv(run / "subgroup_metrics.csv", groups)
    (run / "subgroup_metrics.json").write_text(json.dumps(groups, indent=2, ensure_ascii=False), encoding="utf-8")
    positives = sorted((r for r in rows if r["label"]), key=lambda r: (r["dice_224"], r["sample_id"]))
    selection = [("Best positive by Dice", positives[-1]), ("Median positive by Dice", positives[len(positives) // 2]),
                 ("Worst positive by Dice", positives[0])]
    for src in ["NLST_screening_negative", "private_lesion_free_slice"]:
        selection.append(("First sorted " + src, next(r for r in sorted(rows, key=lambda r: r["sample_id"]) if r["source"] == src)))
    panels = []
    for label, row in selection:
        panel = Image.open(run / row["heatmap_row"]).convert("RGB")
        title = Image.new("RGB", (panel.width, panel.height + 28), "black")
        title.paste(panel, (0, 28))
        ImageDraw.Draw(title).text((4, 5), f"{label} | Dice={row['dice_224']:.4f} | score={row['score_fused']:.4f}", fill="white")
        panels.append(title)
    preview = Image.new("RGB", (max(p.width for p in panels), sum(p.height for p in panels)), "black")
    y = 0
    for panel in panels:
        preview.paste(panel, (0, y))
        y += panel.height
    preview.save(run / "representative_heatmaps.jpg", quality=95)
    (run / "representative_selection.json").write_text(json.dumps([dict(selection_reason=reason, **row) for reason, row in selection], indent=2), encoding="utf-8")
    import torch
    checkpoint = torch.load(run / "checkpoint_final.pt", map_location="cpu", weights_only=False)
    assert checkpoint["task"] == "thymoma" and checkpoint["step"] == 2700
    assert checkpoint["config"]["text_inlayer_enabled"] and checkpoint["config"]["visual_inlayer_enabled"]
    state = checkpoint["model_state_dict"]
    learned = {"fusion_logits": state["fusion_weights"].tolist(),
               "fusion_normalized_weights": torch.softmax(state["fusion_weights"], dim=0).tolist(),
               "adapter_parameter_norms": {k: float(v.float().norm()) for k, v in state.items()
                                           if k.startswith(("inlayer_bank.", "visual_inlayer_bank."))},
               "step": checkpoint["step"], "task": checkpoint["task"]}
    (run / "learned_parameters.json").write_text(json.dumps(learned, indent=2), encoding="utf-8")
    del state, checkpoint
    raw_i, raw_p = official["image"], official["pixel"]
    post_i, post_p = official["postprocess"]["image"], official["postprocess"]["pixel_highres"]
    pct = lambda v: "—" if v is None else f"{100 * v:.2f}%"
    text = ["# 胸腺瘤异常检测训练结果", "", "已完成一次 K=5、seed=0、2700 步的文本/视觉层内适配器联合训练。训练目录为用户指定的 PA-clip新设计2/paclip，复用其中脑 MRI 的训练器、模型和损失。", "",
            "## 训练与划分", "", "- BiomedCLIP ViT-B/16 冻结骨干；学习文本层 8/9/10/11、视觉层 5/8/11 的 attn/ffn 适配器以及三层提示词融合权重。",
            "- 支持集共 5 张、5 位患者：NLST 正常 2 张、私有胸腺瘤病灶 3 张；不是每类 5 张。AdamW，lr=1e-4，weight_decay=1e-5，有效 batch=5。",
            "- 私有 CT 原有 47 人测试名单保持不变，其余 186 人分为 163 人训练候选池和 23 人验证。公共 NLST 对照 16 人按 8/4/4 人划分；所有患者在训练/验证/测试之间互斥。",
            "- 测试 585 张：病灶 365 张（47 人），NLST 阴性 32 张（4 人），私有病灶外切片 188 张（47 人，与病灶切片属于同一批测试患者）。后者仅表示该切片无标注病灶，不是健康受试者。",
            "- 验证集 305 张：181 张病灶、32 张 NLST 阴性、92 张私有病灶外切片。图像和像素阈值均只由验证集确定；测试集不参与阈值选择。",
            "- CT 统一到 LPS 轴向显示，WL=40、WW=400、224×224。每位私有患者最多取病灶面积最大的 8 张，因此结果针对选定切片，不能当作全体积或患者诊断性能。",
            "- 原有 14×14 双线性掩码加载会让部分小病灶消失，训练候选池排除 401 张这类阳性切片，保留 127 位私有患者的 846 张阳性及 8 位 NLST 受试者的 64 张阴性，再按既有 seed=0 规则抽取 5 张；验证/测试不作该筛选。", "",
            "三层文本提示融合的最终 softmax 权重依次为 " + ", ".join(f"{x:.6f}" for x in learned["fusion_normalized_weights"]) + "，来自实际最终 checkpoint；参数审计见 `learned_parameters.json`。", "",
            "## 完整测试集指标", "", "| 评估方式 | 图像 AUROC | 图像 AP | 图像 F1 | 像素 AUROC | Dice | IoU |", "|---|---:|---:|---:|---:|---:|---:|",
            f"| 原始输出，14×14 掩码 | {pct(raw_i['auroc'])} | {pct(raw_i['ap'])} | {pct(raw_i['f1'])} | {pct(raw_p['pixel_auroc'])} | {pct(raw_p['dice'])} | {pct(raw_p['iou'])} |",
            f"| 既有后处理，224×224 掩码 | {pct(post_i['auroc'])} | {pct(post_i['ap'])} | {pct(post_i['f1'])} | {pct(highres['pixel_auroc_224'])} | {pct(post_p['dice_micro'])} | {pct(post_p['iou_micro'])} |", "",
            f"224×224 连续细化分数图的像素 AP 为 {pct(highres['pixel_ap_224'])}，对应 `highres_pixel_metrics.json`。", "",
            "原始像素指标使用训练器既有 14×14 标签；验证集有 37/181 张、测试集有 89/365 张阳性切片的标签在此分辨率下消失，因此原始像素指标不能反映全部小病灶。后处理 Dice/IoU 按保留病灶的 224×224 标注计算微平均，更适合核对定位。两者分辨率不同，应分别报告。", "",
            "## 数据来源分组（同一验证阈值）", "", "| 分组 | 阳性/阴性切片 | 图像 AUROC | 敏感度 | 特异度 | Dice 微平均 224 | 阳性 Dice 宏平均 224 |", "|---|---:|---:|---:|---:|---:|---:|"]
    labels = {"all_test": "完整测试集", "private_lesion_vs_NLST_control": "私有病灶 vs NLST 阴性", "within_private_slice_level": "私有病灶 vs 私有病灶外切片"}
    for g in groups[:3]:
        text.append(f"| {labels[g['group']]} | {g['positive_slices']}/{g['negative_slices']} | {pct(g['image_auroc'])} | {pct(g['sensitivity'])} | {pct(g['specificity'])} | {pct(g['dice_micro_224'])} | {pct(g['positive_dice_macro_224'])} |")
    text += ["", "## 本轮判断", "",
             f"模型已学到部分病灶，预览中的大病灶可以得到较好的热区，但仍有 {pct(groups[0]['positive_zero_overlap_fraction'])} 的阳性切片预测掩码与标注完全没有重叠。完整测试集图像 AUROC 为 {pct(groups[0]['image_auroc'])}，当前验证阈值下特异度仅 {pct(groups[0]['specificity'])}，定位和误报问题都尚未解决。",
             f"私有病灶与公共 NLST 阴性之间的图像 AUROC 达到 {pct(groups[1]['image_auroc'])}，但在私有来源内部只有 {pct(groups[2]['image_auroc'])}；该落差提示来源域差异、解剖切片位置等因素可能参与判别，不能只报告前一组较高的结果，也不能由本轮断言加入公共正常样本有效。私有病灶外切片不是健康患者，这也限制了对‘正常’的解释。",
             "完整测试集阳性比例较高，全部判为阳性的 F1 基线约为 76.84%；本轮 F1 78.07% 不能单独解释为分类效果良好。像素 AUROC 受大量背景像素影响，也应与像素 AP、Dice 和失败热图一起阅读。", ""]
    positive_rows = [r for r in rows if r["label"]]
    peak_hit = float(np.mean([r["heatmap_peak_in_lesion"] for r in positive_rows]))
    contrast = float(np.mean([r["mean_score_in_lesion"] - r["mean_score_outside_lesion"] for r in positive_rows]))
    text += ["", f"病灶切片中，热图最高点落在标注内的比例为 {pct(peak_hit)}；病灶内外平均局部分数差为 {contrast:.6f}。这两项仅用于定位核查。", "",
             "## 结果文件", "", "- `checkpoint_final.pt`：最终权重与优化器记录。", "- `metrics.json` / `thresholds.json`：训练器原始指标及验证阈值。", "- `subgroup_metrics.csv` / `.json`：来源分组指标；`per_slice_metrics.csv`：全部切片分数、面积、Dice、热图路径。",
             "- `heatmaps/figures/overview_*.png`：全部测试切片的分页四联图；`heatmaps/figures/rows/*.png`：单切片四联图。原图/热图/热图+绿色GT/红色预测+绿色GT。", "- `representative_heatmaps.jpg`：按 Dice 选出的最好、中位、最差阳性，以及两种阴性来源的首个排序切片；同时提供 `representative_selection.json` 公开选择规则。",
             "- 热图使用固定 0–1 色阶，没有逐图拉伸。预测数组和全部原始分数见 `evaluations/`。", "",
             "## 解释范围", "", "这是单随机种子、仅 5 张支持图像的先导实验。NLST 为筛查阴性胸部 CT 对照，不等于经过胸腺专科确诊的健康胸腺；只有 4 位公共测试对照，且公共/私有数据在年龄、设备、采集条件等方面可能存在差异。应结合私有来源内的分组结果判断是否利用了数据来源差异。未做‘有/无正常训练’对照，不能据此断言加入正常样本带来多少提升。", ""]
    (run / "训练结果说明.md").write_text("\n".join(text), encoding="utf-8")
    print(json.dumps({"run_dir": str(run), "groups": groups[:3], "peak_in_lesion": peak_hit, "inside_outside_contrast": contrast}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
