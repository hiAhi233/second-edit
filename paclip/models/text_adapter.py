"""残差文本 Adapter。

文档原文：
    因为 BiomedCLIP 的文本编码器是冻结的，它输出的文本嵌入无法精准区分
    上一步定义的多个医学属性锚点，而且也不能直接解冻整个文本编码器，
    所以要引入一个残差适配器。

    保持主干冻结，在输出端或最后若干 Transformer 层加入轻量瓶颈残差适配器：
        只训练 W_down、W_up，
        残差比例 λ_t 建议设置初始 0.05-0.10，
        bottleneck 维度可以从 64、128、256 做消融。

实现：给定冻结文本编码器输出的文本嵌入 h ∈ R^d，

    r = W_up( ReLU( W_down( LayerNorm(h) ) ) )   # 瓶颈残差
    t = h + λ_t * r                               # 残差连接

仅 W_down ∈ R^{b×d}、W_up ∈ R^{d×b}（以及可选的 λ_t）参与训练。
"""

from typing import Optional

import torch
import torch.nn as nn


class ResidualTextAdapter(nn.Module):
    """冻结主干之上的轻量瓶颈残差适配器。"""

    def __init__(
        self,
        d_model: int = 768,
        bottleneck: int = 128,
        lambda_t: float = 0.05,
        lambda_t_learnable: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.bottleneck = bottleneck

        # 只训练 W_down / W_up
        self.ln = nn.LayerNorm(d_model)
        self.down = nn.Linear(d_model, bottleneck, bias=True)   # W_down
        self.up = nn.Linear(bottleneck, d_model, bias=True)     # W_up
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # 残差比例 λ_t：初值 0.05~0.10，可选可学习
        if lambda_t_learnable:
            self.lambda_t = nn.Parameter(torch.tensor(float(lambda_t)))
        else:
            self.register_buffer("lambda_t", torch.tensor(float(lambda_t)))

        # 零初始化 up 层，使初始残差为 0，训练更稳定
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """对冻结文本嵌入施加残差适配。

        Args:
            h: 文本嵌入，形状 (..., d_model)。

        Returns:
            适配后的文本嵌入，形状与输入一致。
        """
        residual = self.up(self.dropout(self.act(self.down(self.ln(h)))))
        return h + self.lambda_t * residual
