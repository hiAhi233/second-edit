"""层内瓶颈残差适配器（Houlsby 式）+ 按器官/病分的适配器库。

背景（docx 改动 16 第 (2) 条自查）：原来的 `ResidualTextAdapter` 套在文本塔**输出端**
（512 维 CLIP 空间），实测它**不是"小调整"** —— 残差模长是原嵌入的 1.2~1.7 倍、
cos(h,t') 只有 0.26~0.49（夹角 60°~75°），反推原始残差是输入的 15~20 倍。它把文本嵌入
近乎整个换掉了；而唯一监督是融合后的那一个异常图，于是它用这个自由度把三层 d 压平了
（cos(d) 0.23→0.96）。

本模块改成**插进冻结基座内部**的形式：在文本塔最后若干层的 attention 之后 / FFN 之后
各插一个瓶颈残差模块。每个器官/病一套，互不干扰。

已探明的硬约束（动手前查实，勿偏离）：
  - 实际加载的是 site-packages 的 open_clip 3.3.0；文本塔是 **HF BertModel**
  - 层列表路径：`clip.text.transformer.encoder.layer`，12 × BertLayer
  - **塔内 hidden = 768**；512 是 `clip.text.proj`（768→640→512）之后的维度
  - `BertLayer.forward` 返回**裸 tensor**；`BertAttention.forward` 返回**二元组**
    `(hidden, attn_weights)` —— 包装时**必须原样保留第二项**，否则
    `BertLayer.forward` 里的 `out, _ = self.attention(...)` 会炸
  - **不可用包装类替换 layer 对象** —— HF 的输出捕获按 `isinstance(parent, BertLayer)`
    匹配，换类会静默废掉 `output_hidden_states`。所以这里只猴补丁 `forward` 属性
  - 两个包装都必须**逐字转发 `**kwargs`**（`BertLayer.forward` 会收到 `position_ids`）
  - 未开梯度检查点（`set_grad_checkpointing` 从未调用），所以 hook 不会被重算破坏
"""

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_LAYERS: Tuple[int, ...] = (8, 9, 10, 11)   # "最后若干 Transformer 层"
DEFAULT_POSITIONS: Tuple[str, ...] = ("attn", "ffn")  # Houlsby 的两个插入点


# ---------------------------------------------------------------------- #
# 单个适配器
# ---------------------------------------------------------------------- #
class InLayerBottleneckAdapter(nn.Module):
    """Houlsby 式瓶颈残差：`h + λ · W_up(GELU(W_down(h)))`。

    `W_up` 零初始化 → step 0 严格恒等，训练从冻结基座原样出发。
    **不加 LayerNorm**：塔内每层自己已有 LayerNorm，再叠一个会改掉 HF 的数值路径，
    且会让"恒等起点"不再严格成立。
    """

    def __init__(self, d_model: int = 768, bottleneck: int = 64,
                 lambda_t: float = 0.1, lambda_t_learnable: bool = True):
        super().__init__()
        self.d_model = d_model
        self.bottleneck = bottleneck
        self.down = nn.Linear(d_model, bottleneck, bias=True)
        self.up = nn.Linear(bottleneck, d_model, bias=True)

        if lambda_t_learnable:
            self.lambda_t = nn.Parameter(torch.tensor(float(lambda_t)))
        else:
            self.register_buffer("lambda_t", torch.tensor(float(lambda_t)))

        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)     # ← 恒等起点
        nn.init.zeros_(self.up.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (..., d_model) —— 逐 token 施加，形状不变。"""
        return h + self.lambda_t * self.up(F.gelu(self.down(h)))


# ---------------------------------------------------------------------- #
# 按器官/病分的适配器库
# ---------------------------------------------------------------------- #
class OrganAdapterBank(nn.Module):
    """每个器官/病一套独立的层内适配器，互不干扰。

    设计要点：
      - 键 = 器官/病名（`"thymoma"`、将来 `"brain"` …）。加新器官 = 加一个键。
      - **只有 active 那个键的参数 `requires_grad=True`** —— 小样本只训自己那套，
        也保证换器官时旧的那套不被污染。
      - 槽位按 `(层, 位置)` 展开成扁平列表，`get()` 在**调用时**动态查表，
        所以 `set_active()` 换器官不需要重装 hook。
    """

    def __init__(self, organs: Sequence[str], layers: Sequence[int] = DEFAULT_LAYERS,
                 positions: Sequence[str] = DEFAULT_POSITIONS,
                 d_model: int = 768, bottleneck: int = 64, lambda_t: float = 0.1):
        super().__init__()
        if not organs:
            raise ValueError("OrganAdapterBank 至少要有一个器官/病名")
        self.layers: List[int] = [int(x) for x in layers]
        self.positions: Tuple[str, ...] = tuple(positions)
        self.n_slot = len(self.layers) * len(self.positions)

        self.adapters = nn.ModuleDict({
            str(organ): nn.ModuleList([
                InLayerBottleneckAdapter(d_model, bottleneck, lambda_t)
                for _ in range(self.n_slot)
            ])
            for organ in organs
        })
        self._active: Optional[str] = None
        self.set_active(str(organs[0]))

    # ---- 槽位 ---- #
    def _slot(self, layer_idx: int, pos: str) -> int:
        return self.layers.index(int(layer_idx)) * len(self.positions) + self.positions.index(pos)

    def get(self, layer_idx: int, pos: str) -> Optional[nn.Module]:
        """取当前 active 器官在 (层, 位置) 上的适配器；无 active 则 None。"""
        if self._active is None:
            return None
        return self.adapters[self._active][self._slot(layer_idx, pos)]

    # ---- 激活切换 ---- #
    @property
    def active(self) -> Optional[str]:
        return self._active

    def organ_names(self) -> List[str]:
        return list(self.adapters.keys())

    def set_active(self, organ: Optional[str]) -> None:
        """切到某个器官：只有它的参数可训，其余全部冻住。"""
        if organ is not None and organ not in self.adapters:
            raise KeyError(f"未知器官 {organ!r}；已注册：{self.organ_names()}")
        self._active = organ
        for name, mods in self.adapters.items():
            for p in mods.parameters():
                p.requires_grad = (name == organ)

    # ---- 每个器官单独存取（互不覆盖）---- #
    def state_dict_for(self, organ: str) -> Dict[str, torch.Tensor]:
        return {k: v for k, v in self.adapters[organ].state_dict().items()}

    def load_state_dict_for(self, organ: str, sd: Dict[str, torch.Tensor]) -> None:
        self.adapters[organ].load_state_dict(sd)

    def n_trainable(self) -> int:
        return sum(p.numel() for name, mods in self.adapters.items()
                   if name == self._active for p in mods.parameters())


# ---------------------------------------------------------------------- #
# 安装
# ---------------------------------------------------------------------- #
def encoder_layers(clip) -> nn.ModuleList:
    """定位 HF 文本塔的 BertLayer 列表。"""
    lay = getattr(getattr(getattr(getattr(clip, "text", None), "transformer", None),
                          "encoder", None), "layer", None)
    if lay is None:
        raise AttributeError(
            "找不到文本塔层列表（期望 clip.text.transformer.encoder.layer）："
            "该基座的文本塔可能不是 HF BertModel，本模块的猴补丁方式不适用。")
    return lay


def _patch_forward(module: nn.Module, bank: "OrganAdapterBank",
                   layer_idx: int, pos: str, hits: Dict[str, int],
                   key: str, tuple_out: bool) -> None:
    """把 module.forward 换成"先跑原实现、再施加适配器"的包装。

    适配器在**调用时**从 bank 动态查（而非安装时绑定），这样 `set_active()` 切器官
    不需要重装 hook。`tuple_out=True` 用于 attention —— 必须保留返回元组的其余项。
    """
    orig = getattr(module, "forward")

    if tuple_out:
        def wrapped(*args, **kwargs):
            out = orig(*args, **kwargs)
            ad = bank.get(layer_idx, pos)
            if ad is None:
                return out
            hits[key] = hits.get(key, 0) + 1
            return (ad(out[0]),) + tuple(out[1:])
    else:
        def wrapped(*args, **kwargs):
            out = orig(*args, **kwargs)
            ad = bank.get(layer_idx, pos)
            if ad is None:
                return out
            hits[key] = hits.get(key, 0) + 1
            return ad(out)

    setattr(module, "forward", wrapped)


def install_inlayer_adapters(clip, bank: "OrganAdapterBank",
                             layers: Optional[Sequence[int]] = None,
                             positions: Optional[Sequence[str]] = None) -> Dict[str, int]:
    """把 bank 的适配器挂到文本塔的指定层上（猴补丁 forward）。

    **必须在 `for p in clip.parameters(): p.requires_grad = False` 之后调用** ——
    否则新插的模块会被一起冻掉。

    Returns:
        触发计数字典 {挂点名: 次数}，用于验证 hook 真的在跑。
    """
    layers = list(layers if layers is not None else bank.layers)
    positions = tuple(positions if positions is not None else bank.positions)
    lay = encoder_layers(clip)
    hits: Dict[str, int] = {}

    for li in layers:
        layer = lay[li]
        if "ffn" in positions:
            _patch_forward(layer, bank, li, "ffn", hits, f"L{li}.ffn", tuple_out=False)
        if "attn" in positions:
            # 包 attention 而不是 layer.attention 之后的张量：这里能拿到 post-LayerNorm 的
            # attention 输出，且不必动 BertLayer.forward 的分块逻辑
            _patch_forward(layer.attention, bank, li, "attn", hits, f"L{li}.attn", tuple_out=True)
    return hits


def adapter_param_report(bank: "OrganAdapterBank") -> str:
    """一行说明：注册了哪些器官、当前激活谁、各有多少参数。"""
    parts = []
    for name, mods in bank.adapters.items():
        n = sum(p.numel() for p in mods.parameters())
        mark = " ← active" if name == bank.active else ""
        parts.append(f"{name}:{n}{mark}")
    return f"[inlayer] 器官={bank.organ_names()} 槽位={bank.n_slot} | " + "  ".join(parts)
