"""参考数据读取、训练点构造与结果导出。

本模块集中处理所有与数组有关的职责：

- 读取 MATLAB 生成的参考解；
- 构造 IC 目标值、左右边界配对坐标和 PDE 配点；
- 把参考网格转成模型输入并做全网格预测；
- 计算相对 L2 误差、保存 ``prediction.mat``。

约定（见 reflector-v3.md 第 7、8 节）：

- MATLAB 的 ``X, T, q`` 为 ``[Nt, Nx]``，Python 统一使用 ``order="C"``；
- 复数场只保存 ``q_ref``、``q_pred``、``abs_error``，实部虚部在绘图时提取。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io
import torch
from scipy.stats import qmc

REFERENCE_KEYS = ("x", "t", "X", "T", "q")


@dataclass
class Reference:
    """MATLAB 生成的参考网格与参考解。"""

    x: np.ndarray  # [Nx]
    t: np.ndarray  # [Nt]
    X: np.ndarray  # [Nt, Nx]
    T: np.ndarray  # [Nt, Nx]
    q: np.ndarray  # [Nt, Nx]，复数

    @property
    def lb(self) -> np.ndarray:
        """计算区域下界 [x_min, t_min]，由参考网格推导，避免另设区域常量。"""
        return np.array([self.x[0], self.t[0]], dtype=np.float64)

    @property
    def ub(self) -> np.ndarray:
        """计算区域上界 [x_max, t_max]。"""
        return np.array([self.x[-1], self.t[-1]], dtype=np.float64)


@dataclass
class TrainingData:
    """训练用的坐标张量与 IC 目标值。"""

    x_ic: torch.Tensor        # [N_ic, 2]
    uv_ic: torch.Tensor       # [N_ic, 2]
    x_bc_left: torch.Tensor   # [N_bc, 2]
    x_bc_right: torch.Tensor  # [N_bc, 2]
    x_pde: torch.Tensor       # [N_pde, 2]
    meta: dict[str, Any] = field(default_factory=dict)


def load_reference(path: str | Path) -> Reference:
    """读取参考 MAT，校验字段与形状后返回 :class:`Reference`。"""
    path = Path(path)
    data = scipy.io.loadmat(path)
    missing = [key for key in REFERENCE_KEYS if key not in data]
    if missing:
        raise KeyError(f"{path} 缺少变量 {missing}")

    ref = Reference(
        x=np.asarray(data["x"], dtype=np.float64).ravel(),
        t=np.asarray(data["t"], dtype=np.float64).ravel(),
        X=np.asarray(data["X"], dtype=np.float64),
        T=np.asarray(data["T"], dtype=np.float64),
        q=np.asarray(data["q"]).astype(np.complex128),
    )

    if ref.X.shape != ref.T.shape or ref.X.shape != ref.q.shape:
        raise ValueError(f"X/T/q 形状不一致：{ref.X.shape} {ref.T.shape} {ref.q.shape}")
    n_t, n_x = ref.X.shape
    if ref.x.size != n_x or ref.t.size != n_t:
        raise ValueError(f"x/t 与网格不匹配：x {ref.x.size}, t {ref.t.size}, grid {n_t}x{n_x}")
    return ref


def build_training_data(
    ref: Reference,
    config: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> TrainingData:
    """构造 IC、周期值边界与 PDE 配点。

    IC 的 x 坐标从参考网格抽取，目标值直接取参考解初始时刻所在行，
    Python 不重复实现解析解。左右边界共享同一组时刻。
    """
    sampling = config["sampling"]
    n_ic = int(sampling["n_ic"])
    n_bc = int(sampling["n_bc"])
    n_f = int(sampling["n_f"])
    seed = int(sampling["seed"])
    lb, ub = ref.lb, ref.ub

    # IC / BC 沿用旧版抽样方式：均匀有放回地抽取网格点。
    np.random.seed(seed)
    ic_idx = np.random.choice(ref.x.size, size=n_ic, replace=True)
    bc_idx = np.random.choice(ref.t.size, size=n_bc, replace=True)

    x_ic = ref.x[ic_idx][:, None]
    t_ic = np.full((n_ic, 1), ref.t[0])
    x_ic_coord = np.hstack([x_ic, t_ic])
    q_ic = ref.q[0, ic_idx]
    uv_ic = np.column_stack([q_ic.real, q_ic.imag])

    t_bc = ref.t[bc_idx]
    x_bc_left = np.column_stack([np.full(n_bc, lb[0]), t_bc])
    x_bc_right = np.column_stack([np.full(n_bc, ub[0]), t_bc])

    # 独立 LHS 配点；采样器由 pyDOE 换成 SciPy，区域、数量与 seed 保持不变。
    sampler = qmc.LatinHypercube(d=2, seed=seed)
    x_f = qmc.scale(sampler.random(n=n_f), lb, ub)

    # PDE 配点沿用旧版的拼接规则。
    x_pde = np.vstack([x_ic_coord, x_bc_left, x_bc_right, x_f])

    def to_tensor(array: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(array, dtype=dtype, device=device)

    meta = {
        "n_ic": n_ic,
        "n_bc": n_bc,
        "n_f": n_f,
        "n_pde": int(x_pde.shape[0]),
        "seed": seed,
        "sampler": "scipy.stats.qmc.LatinHypercube(d=2)",
        "lb": lb.tolist(),
        "ub": ub.tolist(),
        "ic_time": float(ref.t[0]),
    }

    return TrainingData(
        x_ic=to_tensor(x_ic_coord),
        uv_ic=to_tensor(uv_ic),
        x_bc_left=to_tensor(x_bc_left),
        x_bc_right=to_tensor(x_bc_right),
        x_pde=to_tensor(x_pde),
        meta=meta,
    )


def grid_input(ref: Reference, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """参考网格展平后的模型输入 ``[Nt*Nx, 2]``，与 MATLAB 的 ``[Nt, Nx]`` 一一对应。"""
    xt = np.column_stack([ref.X.ravel(order="C"), ref.T.ravel(order="C")])
    return torch.as_tensor(xt, dtype=dtype, device=device)


def predict_grid(
    model: torch.nn.Module,
    ref: Reference,
    device: torch.device,
    dtype: torch.dtype,
) -> np.ndarray:
    """在完整参考网格上预测复数场 ``q = u + i v``，形状与参考解一致。"""
    model.eval()
    with torch.no_grad():
        raw = model(grid_input(ref, device, dtype)).detach().cpu().numpy()
    return (raw[:, 0] + 1j * raw[:, 1]).reshape(ref.X.shape, order="C")


def relative_l2(q_pred: np.ndarray, q_ref: np.ndarray) -> tuple[float, float]:
    """返回 (复数相对 L2, 幅值相对 L2)，两者都在参考网格上计算。"""
    complex_l2 = float(np.linalg.norm(q_pred - q_ref) / np.linalg.norm(q_ref))
    mag_l2 = float(
        np.linalg.norm(np.abs(q_pred) - np.abs(q_ref)) / np.linalg.norm(np.abs(q_ref))
    )
    return complex_l2, mag_l2


def save_prediction(path: str | Path, ref: Reference, q_pred: np.ndarray) -> np.ndarray:
    """保存 ``prediction.mat``，字段与 reflector-v3.md 第 8 节一致。"""
    abs_error = np.abs(q_pred - ref.q)
    scipy.io.savemat(
        Path(path),
        {
            "x": ref.x,
            "t": ref.t,
            "X": ref.X,
            "T": ref.T,
            "q_ref": ref.q,
            "q_pred": q_pred,
            "abs_error": abs_error,
        },
    )
    return abs_error


def sha256_file(path: str | Path) -> str:
    """参考数据文件的摘要，写入日志用于复现实验。"""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
