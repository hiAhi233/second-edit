"""Collect the third-step B1/B2/C1/C2/D1/D2 metrics into one table."""

import argparse
import csv
import json
from pathlib import Path

from paclip.engine.trainer import atomic_write_json


CONFIGS = ("B1", "C1", "B2", "C2", "D1", "D2")


def main(argv=None):
    project = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=str(project / 'runs' / 'brain_mri_E3_20261025'))
    parser.add_argument('--baseline-root', default=None,
                        help='Read completed B1/B2 here when the continuation uses a separate root')
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    baseline_root = Path(args.baseline_root).resolve() if args.baseline_root else root
    rows = []
    for config in CONFIGS:
        source_root = baseline_root if config in ('B1', 'B2') else root
        result_path = source_root / config / "results.json"
        protocol_path = source_root / config / "protocol.json"
        completion_path = source_root / config / "completion.json"
        if not all(path.exists() for path in (result_path, protocol_path, completion_path)):
            continue
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if protocol.get("status") != "complete" or completion.get("status") != "complete":
            continue
        results = json.loads(result_path.read_text(encoding="utf-8"))
        for dataset, metrics in results.items():
            volume = metrics.get("volume_detection", {})
            rows.append({
                "config": config,
                "dataset": dataset,
                "base": protocol.get("base"),
                "context": protocol.get("context"),
                "consistency": protocol.get("consistency"),
                "selected_beta": protocol.get("selected_beta"),
                "pixel_ap": metrics.get("pixel_ap"),
                "pixel_auroc": metrics.get("pixel_auroc"),
                "dice_micro": metrics.get("dice_micro"),
                "lesion_recall": metrics.get("lesion_recall"),
                "small_lesion_recall": metrics.get("small_lesion_recall"),
                "normal_false_positive_area_ratio": metrics.get("normal_false_positive_area_ratio"),
                "normal_slice_false_positive_components": metrics.get("normal_slice_false_positive_components"),
                "volume_lesion_recall": volume.get("lesion_recall"),
                "volume_fp_per_case": volume.get("false_positives_per_case"),
                "slice_context_available": (metrics.get("slice_context") or {}).get("available"),
                "context_group_fallback_slices": metrics.get("context_group_fallback_slices"),
            })
    if not rows:
        raise SystemExit("No E3 results found; run run_brain_e3.py first")
    columns = list(rows[0])
    with (root / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)
    lines = ["# 第三步 B1/B2/C1/C2/D1/D2 汇总", "",
             "| 配置 | 数据 | β | AP | Dice | 小病灶召回 | 正常 FP 面积 | 3D 召回 | 3D FP/例 | 源邻片可用 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    def number(value, spec):
        return "" if value is None else format(value, spec)
    for row in rows:
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            row["config"], row["dataset"], row["selected_beta"],
            number(row["pixel_ap"], ".4f"), number(row["dice_micro"], ".4f"),
            number(row["small_lesion_recall"], ".4f"),
            number(row["normal_false_positive_area_ratio"], ".6f"),
            number(row["volume_lesion_recall"], ".4f"),
            number(row["volume_fp_per_case"], ".2f"),
            row["slice_context_available"]))
    lines += ["", "源 PNG 没有真实 slice_z，D1/D2 在源测试上应等于 C1/C2；2.5D 仅在 WMH/ATLAS 上生效。"
              "所有数值来自固定源验证阈值；未在目标测试上重新选参。"]
    (root / "第三步效果对比.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    atomic_write_json(str(root / "report_artifacts.json"), {"report": str(root / "第三步效果对比.md"),
                                                             "comparison": str(root / "comparison.csv")})
    print(f"[report] {root / '第三步效果对比.md'}", flush=True)


if __name__ == "__main__":
    main()
