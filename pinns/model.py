"""最小 MLP：输入归一化 + 若干全连接层。

只负责网络本身：结构、归一化、Xavier 初始化与 forward。
不涉及数据读取、采样、PDE、损失、优化器或绘图。

约定：输入 ``xt`` 形状 ``[N, 2]``，输出形状 ``[N, 2]``；
``xt[:, 0] = x``、``xt[:, 1] = t``，输出列为 ``(Re(q), Im(q))``。
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

# 可用的激活函数，按配置中的名称选择。
ACTIVATIONS: dict[str, type[nn.Module]] = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
}


class MLP(nn.Module):
    """全连接网络，输入先线性映射到 ``[0, 1]``。

    归一化边界 ``lb/ub`` 注册为 buffer，随模型迁移设备并进入 ``state_dict``，
    因此重载 ``model.pt`` 时不需要额外保存区域信息。
    """

    def __init__(
        self,
        in_dim: int = 2,
        out_dim: int = 2,
        hidden_layers: int = 10,
        width: int = 40,
        activation: str = "tanh",
        lb: Sequence[float] = (-5.0, -0.5),
        ub: Sequence[float] = (5.0, 0.5),
    ) -> None:
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(f"不支持的激活函数 {activation!r}，可选 {sorted(ACTIVATIONS)}")

        self.register_buffer("lb", torch.as_tensor(lb, dtype=torch.float32))
        self.register_buffer("ub", torch.as_tensor(ub, dtype=torch.float32))

        layers: list[nn.Module] = []
        n_in = in_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(n_in, width))
            layers.append(ACTIVATIONS[activation]())
            n_in = width
        layers.append(nn.Linear(n_in, out_dim))
        self.net = nn.Sequential(*layers)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Xavier uniform 权重 + 零偏置，保持旧版初始化方式。"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, xt: torch.Tensor) -> torch.Tensor:
        """输入 ``[N, 2]`` 的 ``(x, t)``，返回 ``[N, 2]`` 的 ``(u, v)``。"""
        xt = (xt - self.lb) / (self.ub - self.lb)
        return self.net(xt)
