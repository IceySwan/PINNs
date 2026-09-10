"""NLS 单孤子：i q_t + q_xx + 2 |q|^2 q = 0。

本文件是具体方程数学逻辑的唯一位置：

- PDE 残差；
- IC 损失；
- 周期值边界损失。

Trainer 只依赖 ``compute_losses(model, train_data)``，返回 ``ic``、``bc``、``pde``
三个标量。归约方式沿用旧版：每个实分量先对样本求 MSE，再把实部、虚部相加。
"""

from __future__ import annotations

from typing import Any

import torch
from torch.autograd import grad


def _mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """单个实分量的均方误差。"""
    return torch.mean((pred - target) ** 2)


def residual(model: torch.nn.Module, xt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """PDE 残差 ``(f_u, f_v)``。

    令 ``q = u + i v``，则 ``i q_t + q_xx + 2|q|^2 q = 0`` 的虚部为

        f_u = u_t + v_xx + 2 (u^2 + v^2) v，

    负实部为

        f_v = v_t - u_xx - 2 (u^2 + v^2) u。
    """
    xt = xt.clone().requires_grad_(True)
    uv = model(xt)
    u, v = uv[:, 0:1], uv[:, 1:2]

    u_grad = grad(u.sum(), xt, create_graph=True)[0]
    v_grad = grad(v.sum(), xt, create_graph=True)[0]
    u_x, u_t = u_grad[:, 0:1], u_grad[:, 1:2]
    v_x, v_t = v_grad[:, 0:1], v_grad[:, 1:2]
    u_xx = grad(u_x.sum(), xt, create_graph=True)[0][:, 0:1]
    v_xx = grad(v_x.sum(), xt, create_graph=True)[0][:, 0:1]

    amp2 = u ** 2 + v ** 2
    f_u = u_t + v_xx + 2.0 * amp2 * v
    f_v = v_t - u_xx - 2.0 * amp2 * u
    return f_u, f_v


def initial_loss(model: torch.nn.Module, data: Any) -> torch.Tensor:
    """初值损失：参考解在 ``t = t[0]`` 的复数值作为目标。"""
    uv_pred = model(data.x_ic)
    return _mse(uv_pred[:, 0:1], data.uv_ic[:, 0:1]) + _mse(
        uv_pred[:, 1:2], data.uv_ic[:, 1:2]
    )


def boundary_loss(model: torch.nn.Module, data: Any) -> torch.Tensor:
    """周期值边界损失：左右边界共享同一组时刻，约束 u、v 分别相等。"""
    uv_left = model(data.x_bc_left)
    uv_right = model(data.x_bc_right)
    return _mse(uv_left[:, 0:1], uv_right[:, 0:1]) + _mse(
        uv_left[:, 1:2], uv_right[:, 1:2]
    )


def pde_loss(model: torch.nn.Module, data: Any) -> torch.Tensor:
    """方程残差损失，对实部、虚部分别取均方后相加。"""
    f_u, f_v = residual(model, data.x_pde)
    return torch.mean(f_u ** 2) + torch.mean(f_v ** 2)


def compute_losses(model: torch.nn.Module, train_data: Any) -> dict[str, torch.Tensor]:
    """Trainer 使用的唯一接口，返回 ``{"ic", "bc", "pde"}`` 三个标量。"""
    return {
        "ic": initial_loss(model, train_data),
        "bc": boundary_loss(model, train_data),
        "pde": pde_loss(model, train_data),
    }
