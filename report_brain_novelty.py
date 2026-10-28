"""Retain every metric, failures and paired seeds; never hide a negative result."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from paclip.data.patient_manifest import write_json

NAMES = {"first_step": "患者独立第一步基准", "dense_only": "多层密集定位头", "ratio_control": "密集头取样比例对照",
         "image_synthetic": "密集头加图像伪异常", "feature_synthetic": "密集头加特征伪异常",
         "joint_synthetic": "密集头加两类伪异常"}


def flatten(obj, prefix=""):
    output = {}
    for k, v in obj.items():
        if k == "per_patient":
            continue  # Preserved in raw JSON, not repeated as thousands of summary columns.
        name = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            output.update(flatten(v, name))
        elif v is None or isinstance(v, (int, float)) and not isinstance(v, bool):
            output[name] = v
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs, lines = [], ["# 未知病灶定位实验结果", "", "完整原始指标随每组 results.json 保留；缺失与失败不作为零分。", ""]
    for path in sorted(args.root.rglob("completion.json")):
        if not (path.parent / "config.json").exists():
            continue
        status = json.loads(path.read_text(encoding="utf-8"))
        config = json.loads((path.parent / "config.json").read_text(encoding="utf-8"))
        name = NAMES.get(config["mode"], config["mode"])
        name += f"；排序权重 {config['lambda_rank']}；视觉适配器{'训练' if config['train_visual'] else '冻结'}"
        lines += [f"## {name}，种子 {config['seed']}", "", f"状态：{status['status']}；目录：`{path.parent}`。", ""]
        if status["status"] != "complete":
            lines += [f"原因：{status.get('error', '运行尚未完成')}。", ""]
            continue
        results = json.loads((path.parent / "results.json").read_text(encoding="utf-8"))
        identity = json.loads((path.parent / "identity.json").read_text(encoding="utf-8"))
        for condition, metrics in results.items():
            lines += [f"### {condition}", ""]
            if metrics.get("status") == "unavailable":
                lines += [metrics["reason"], ""]
                continue
            flat = flatten(metrics)
            lines += ["| 指标（分母见原始结果） | 数值 |", "|---|---:|"]
            lines += [f"| {k} | {format(v, '.8g') if v is not None else '不可用'} |" for k, v in flat.items()]
            lines += ["", "FROC 逐阈值记录、不可用体积原因及原始预测见该组 `results.json` 和 `evaluation/`。", ""]
            runs.append({"method": name, "mode": config["mode"], "seed": config["seed"],
                         "condition": condition, "metrics": flat, "identity": identity, "config": config})
    lines += ["## 多种子与成对差值", "", "以下统计仅合并相同划分、支持集和评估条件；只有一组种子时不报告标准差。", ""]
    grouped = defaultdict(list)
    for r in runs:
        key = (r["method"], r["condition"], r["identity"]["split_digest"], r["identity"]["support_digest"])
        grouped[key].append(r)
    summary = []
    for key, values in grouped.items():
        if len({r["seed"] for r in values}) != len(values):
            raise ValueError(f"Duplicate seed runs for {key}; select one before summarizing")
        from paclip.data.fewshot import stable_digest
        settings = {stable_digest({k: v for k, v in r["config"].items() if k != "seed"}) for r in values}
        if len(settings) > 1:
            raise ValueError(f"Cannot combine seeds with different training settings: {key}")
        allkeys = sorted(set.intersection(*(set(r["metrics"]) for r in values)))
        stats = {}
        for k in allkeys:
            available = [r["metrics"][k] for r in values if r["metrics"][k] is not None]
            stats[k] = {"mean": float(np.mean(available)) if available else None,
                        "std": float(np.std(available, ddof=1)) if len(available) > 1 else None,
                        "valid_seeds": len(available)}
        summary.append({"method": key[0], "condition": key[1], "seeds": [r["seed"] for r in values], "statistics": stats})
        lines += [f"### {key[0]} / {key[1]}", "", "| 指标 | 均值 | 标准差 |", "|---|---:|---:|"]
        lines += [f"| {k} | {format(v['mean'], '.8g') if v['mean'] is not None else '不可用'} | {v['std'] if v['std'] is not None else '不足两个有效种子'} |" for k, v in stats.items()]
        lines += [""]
    paired = []
    for r in runs:
        controls = {"first_step": [], "dense_only": ["first_step"], "ratio_control": ["dense_only"],
                    "image_synthetic": ["dense_only", "ratio_control"], "feature_synthetic": ["dense_only", "ratio_control"],
                    "joint_synthetic": ["dense_only", "ratio_control", "image_synthetic", "feature_synthetic"]}[r["mode"]]
        for control in controls:
            candidates = [b for b in runs if b["mode"] == control and b["seed"] == r["seed"] and b["condition"] == r["condition"]
                and all(b["identity"][k] == r["identity"][k] for k in ("split_digest", "support_digest", "base_sha256"))]
            if len(candidates) != 1:
                continue
            base = candidates[0]
            diffs = {k: r["metrics"][k] - base["metrics"][k] if r["metrics"][k] is not None and base["metrics"][k] is not None else None
                     for k in r["metrics"] if k in base["metrics"]}
            paired.append({"method": r["method"], "control": NAMES[control], "condition": r["condition"], "seed": r["seed"], "difference": diffs})
            lines += [f"### 相对{NAMES[control]}：{r['method']} / {r['condition']} / 种子 {r['seed']}", "", "| 指标 | 成对差值 |", "|---|---:|"]
            lines += [f"| {k} | {format(v, '+.8g') if v is not None else '不可用'} |" for k, v in diffs.items()]
            lines += ["", "AP、召回、Dice 与误报分别判断方向；三种子一致不能替代患者级不确定性评估。", ""]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    write_json(args.output.with_suffix(".json"), {"runs": runs, "summary": summary, "paired_differences": paired})


if __name__ == "__main__":
    main()
