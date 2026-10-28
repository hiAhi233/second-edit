"""Unified command line: python -m paclip <command> [options]."""
from __future__ import annotations

import argparse
import importlib
import sys


COMMANDS = {
    "train": ("paclip.engine.trainer", "单次训练、续训或独立评估"),
    "brain": ("paclip.cli.brain", "脑 MRI 多 K、多随机种子实验"),
    "brain-memory": ("paclip.cli.brain_memory", "冻结 E1 并建立多层正常原型，验证 E2 定位效果"),
    "thymoma": ("paclip.cli.thymoma_ad", "胸腺瘤与 NLST 正常样本的固定 K=5 实验"),
    "thymoma-roi": ("paclip.cli.thymoma_roi", "仅 ROI 切片的历史训练协议"),
    "thymoma-ablation": ("paclip.cli.thymoma_ablation", "胸腺瘤适配器消融实验"),
    "prepare-brain": ("paclip.cli.prepare_brain", "整理脑 MRI 切片与划分"),
    "prepare-thymoma": ("paclip.cli.prepare_thymoma_ad", "构建胸腺瘤与正常 CT 数据"),
    "prepare-thymoma-roi": ("paclip.cli.prepare_thymoma_roi", "提取胸腺瘤 ROI 切片"),
    "heatmaps": ("paclip.cli.heatmaps", "读取权重或预测文件生成热图"),
    "summarize-thymoma": ("paclip.cli.summarize_thymoma", "汇总胸腺瘤测试指标"),
    "gallery-thymoma": ("paclip.cli.gallery_thymoma", "为已有胸腺瘤热图生成索引"),
    "diagnose-text": ("paclip.diagnostics.text_cosine", "分析提示词层级的方向相似度"),
    "diagnose-thymoma": ("paclip.diagnostics.analyze_thymoma", "诊断胸腺瘤预测"),
    "smoke-brain": ("paclip.diagnostics.smoke_brain", "CUDA 脑 MRI 训练流程检查"),
    "smoke-text": ("paclip.diagnostics.smoke_text", "CUDA 文本适配器检查"),
    "smoke-visual": ("paclip.diagnostics.smoke_visual", "CUDA 视觉适配器检查"),
}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="python -m paclip", description="PA-CLIP2 医学影像小样本异常检测",
        epilog="查看子命令参数：python -m paclip <command> --help",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("command", nargs="?", choices=COMMANDS, help="\n".join(
        f"{name}: {description}" for name, (_, description) in COMMANDS.items()))
    if not argv or argv[0] in {"-h", "--help"}:
        parser.print_help()
        return 0
    command = parser.parse_args(argv[:1]).command
    module_name, description = COMMANDS[command]
    # Smoke commands take no arguments; their help must never start CUDA work.
    if command.startswith("smoke-"):
        smoke_parser = argparse.ArgumentParser(prog=f"python -m paclip {command}", description=description)
        smoke_parser.parse_args(argv[1:])
    previous = sys.argv
    try:
        sys.argv = [f"python -m paclip {command}", *argv[1:]]
        result = importlib.import_module(module_name).main()
        # Schedulers return result records; only explicit integer return values
        # are process exit codes.
        return result if isinstance(result, int) else 0
    finally:
        sys.argv = previous


if __name__ == "__main__":
    raise SystemExit(main())
