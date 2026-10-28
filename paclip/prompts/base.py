"""Prompt value objects, tokenizer utilities and checkpoint-version checks."""
from dataclasses import dataclass
from typing import Dict, List, Sequence
import torch


@dataclass
class ThreeLevelPrompts:
    """三层提示词模板，每层 normal / abnormal 各一组（可多条，做均值池化）。"""

    normal: Dict[str, List[str]]
    abnormal: Dict[str, List[str]]

    @property
    def levels(self) -> List[str]:
        return list(self.normal.keys())

    def all_texts(self) -> List[str]:
        texts: List[str] = []
        for lvl in self.levels:
            texts.extend(self.normal[lvl])
            texts.extend(self.abnormal[lvl])
        return texts


def resolve_prompt_set(requested, saved, registry, default, *, loading=False):
    """新实验用默认版本；加载权重继承版本，缺元数据时必须明确指定。"""
    if loading and not saved and requested is None:
        raise ValueError("checkpoint 未记录 prompt_set，请显式指定 --prompt-set 为该权重训练时的版本")
    if saved and saved not in registry:
        raise ValueError(f"checkpoint 的提示词 {saved!r} 不属于当前任务")
    if requested is not None and saved and requested != saved:
        raise ValueError(f"提示词不一致：checkpoint={saved}，--prompt-set={requested}；不能按原实验续训/评估")
    name = requested or saved or default
    if name not in registry:
        raise ValueError(f"未知提示词版本：{name!r}")
    return name


def build_text_anchors(
    texts: Sequence[str],
    tokenizer,
    max_length: int = 256,
) -> Dict[str, torch.Tensor]:
    """tokenize 字符串列表，返回 input_ids / attention_mask（CPU 长整型）。"""
    enc = tokenizer(
        list(texts),
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
