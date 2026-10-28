"""Index existing heatmaps and assemble additional patient-diverse contact sheets."""
import csv
import json
from pathlib import Path
from paclip.paths import PROJECT_ROOT

import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEFAULT_RUN_DIR = PROJECT_ROOT / "runs/thymoma_nlst_20261021/tv_joint_k5_s0_74ab80954001dcac"


def sheet(root, font, small, path, selected, title, note):
    panels = []
    for row in selected:
        original = Image.open(root / row["heatmap_row"]).convert("RGB")
        panel = Image.new("RGB", (original.width, original.height + 30), "black")
        panel.paste(original, (0, 30))
        ImageDraw.Draw(panel).text((6, 5), f"{row['case_id']} | 来源={row['source']} | Dice={float(row['dice_224']):.3f}", font=font, fill="white")
        panels.append(panel)
    canvas = Image.new("RGB", (panels[0].width, 72 + sum(p.height for p in panels)), "black")
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 5), title, font=font, fill="white")
    draw.text((6, 30), note, font=small, fill="#cccccc")
    draw.text((6, 50), "列：原图 / 固定色阶热图 / 热图+绿色真实ROI / 红色预测掩码+绿色真实ROI", font=small, fill="#cccccc")
    y = 72
    for panel in panels:
        canvas.paste(panel, (0, y))
        y += panel.height
    canvas.save(root / path, quality=95)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR, help="包含 per_slice_metrics.csv 和 heatmaps 的已有实验目录")
    args = parser.parse_args()
    root = args.run_dir.resolve()
    with (root / "per_slice_metrics.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 15)
    small = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 13)

    
    by_case = {}
    for row in rows:
        if row["source"] == "private_lesion":
            by_case.setdefault(row["case_id"], []).append(row)
    representatives = []
    for case, samples in sorted(by_case.items()):
        ordered = sorted(samples, key=lambda r: (float(r["dice_224"]), r["sample_id"]))
        representatives.append(ordered[len(ordered) // 2])
    representatives.sort(key=lambda r: (float(r["dice_224"]), r["case_id"]))
    chosen = [representatives[int(i)] for i in np.linspace(0, len(representatives) - 1, 18).round().astype(int)]
    selections = {}
    for page in range(3):
        selected = [chosen[i] for i in [2*page, 2*page+1, 6+2*page, 7+2*page, 12+2*page, 13+2*page]]
        name = f"更多病灶病例_{page+1:02d}.jpg"
        sheet(root, font, small, name, selected, f"更多病灶病例 {page+1}/3：每页6位不同患者", "每人取Dice排序的中位切片；从47位患者中按代表切片Dice均匀取18人，覆盖较差、中等和较好效果。")
        selections[name] = selected

    controls = []
    for case in sorted({r["case_id"] for r in rows if r["source"] == "NLST_screening_negative"}):
        samples = sorted([r for r in rows if r["case_id"] == case], key=lambda r: r["sample_id"])
        controls.append(samples[len(samples) // 2])
    private = sorted([r for r in rows if r["source"] == "private_lesion_free_slice"], key=lambda r: float(r["score_fused"]))
    controls += [private[len(private)//2], private[-1]]
    name = "更多阴性对照_01.jpg"
    sheet(root, font, small, name, controls, "更多阴性对照：4位NLST受试者，以及2张私有目标ROI外切片", "私有ROI外切片来自胸腺瘤患者，不能当作健康受试者；其中包含不同解剖层面，仅供误报诊断。")
    selections[name] = controls
    (root / "更多图集选择记录.json").write_text(json.dumps(selections, ensure_ascii=False, indent=2), encoding="utf-8")

    pages = sorted((root / "heatmaps/figures").glob("overview_*.png"))
    if len(list((root / "heatmaps/figures/rows").glob("*.png"))) != len(rows):
        raise ValueError("Heatmap row count does not match per_slice_metrics.csv")
    lines = ["# 全部热图索引", "", f"本次训练已生成{len(rows)}张测试切片的四联图，共{len(pages)}页。以下全部来自同一个最终checkpoint，未修改预测或热图色阶。", "",
             "## 更多不同患者预览", "", "病灶预览共18位不同患者；每人选其Dice排序的中位切片，再按代表切片Dice均匀取样，覆盖成功和失败情况。不是只选效果最好的图。", ""]
    for name in selections:
        lines.append(f"- [{name}]({(root/name).as_posix()})")
    lines += ["", "绿色：真实ROI；红色：预测掩码。热图色阶固定0–1。Dice使用原预测与224×224真实ROI计算。", "",
              f"## 原始完整{len(pages)}页", "", "以下按照原始测试顺序，每页最多8张。同一患者的相邻切片可能连续出现。", ""]
    for index, path in enumerate(pages):
        patients = ", ".join(dict.fromkeys(r["case_id"] for r in rows[index*8:(index+1)*8]))
        lines.append(f"- [第{index+1:02d}页 · {patients}]({path.as_posix()})")
    lines += ["", "## 单切片索引", "", "| 切片 | 来源 | Dice | 四联图 |", "|---|---|---:|---|"]
    for row in rows:
        lines.append(f"| {row['sample_id']} | {row['source']} | {float(row['dice_224']):.3f} | [查看]({(root/row['heatmap_row']).as_posix()}) |")
    (root / "全部热图索引.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({"full_pages":len(pages),"slice_heatmaps":len(rows),"additional_previews":list(selections)},ensure_ascii=False))


if __name__ == "__main__":
    main()
