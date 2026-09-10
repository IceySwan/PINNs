"""Python 诊断图：训练损失与误差密度。

科研结果图（heatmap、3D、切片）由 MATLAB 的 ``plot_results.m`` 生成。
本模块只画两张图：

- ``plot_loss``：total / ic / bc / pde 随优化次数的变化；
- ``plot_error_density``：``|q_pred - q_ref|`` 的分布。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")  # 无显示环境下直接写文件

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator  # noqa: E402


def plot_loss(
    history: Sequence[dict[str, Any]],
    path: str | Path,
    title: str = "Training loss",
    series: Sequence[str] = ("total", "ic", "bc", "pde"),
) -> None:
    """绘制训练损失。

    x 轴使用全局评估序号（Adam 与 L-BFGS 连续编号），并在两个阶段之间画分界线。

    ``series`` 控制绘制哪些损失曲线；默认四条全画。只需总损失时传
    ``series=("total",)``，例如单独保存的 ``total_loss.png``。
    """
    if not history:
        raise ValueError("history 为空，无法绘图")

    x = np.arange(1, len(history) + 1)

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for key in series:
        ax.semilogy(x, np.array([row[key] for row in history]), linewidth=1.2, label=key)

    # y 轴按实际 loss 设置：对数刻度，主刻度落在 10 的整数次幂上，
    # 标签用普通文本 "1e-05"（不用 mathtext 上标，避免小字号下难以辨认）。
    # ylim 严格取数据覆盖的整十倍程，顶部不再留空余刻度。
    values = np.concatenate(
        [np.asarray([row[key] for row in history], dtype=float) for key in series]
    )
    positive = values[values > 0]
    if positive.size:
        ax.set_ylim(
            10.0 ** np.floor(np.log10(positive.min())),
            10.0 ** np.ceil(np.log10(positive.max())),
        )
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0e}"))

    # 阶段分界线：最后一条 adam 记录之后进入 L-BFGS。
    stages = [row["stage"] for row in history]
    if "lbfgs" in stages:
        boundary = stages.index("lbfgs") + 1
        ax.axvline(boundary, color="0.4", linestyle="--", linewidth=1.0)
        ax.annotate(
            "L-BFGS",
            xy=(boundary, ax.get_ylim()[1]),
            xytext=(4, -12),
            textcoords="offset points",
            fontsize=9,
            color="0.4",
        )

    ax.set_xlabel("evaluation")
    ax.set_ylabel("loss (log scale)")
    ax.set_title(title)
    # log 轴的次级刻度（2,3,...,9）会画出一排密集细线，这里只保留主网格线。
    ax.grid(True, which="major", alpha=0.3)
    ax.grid(False, which="minor")
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(path), dpi=300)
    plt.close(fig)


def plot_error_density(
    values: np.ndarray,
    path: str | Path,
    bins: int = 200,
    signed: bool = False,
    title: str | None = None,
) -> None:
    """绘制误差分布。

    ``signed=False``：输入 ``|q_pred - q_ref|``，横轴非负，标注均值与 99 分位。

    ``signed=True``：输入带符号误差，例如幅值误差 ``|q_pred| - |q_ref|``
    （也可传入 ``Re(q_pred - q_ref)`` 或 ``Im(q_pred - q_ref)``）。
    横轴以 0 为中心对称分箱，并标出 0、均值与标准差，用于判断预测偏高还是偏低。
    """
    values = np.asarray(values, dtype=np.float64).ravel()
    mean = float(values.mean())
    std = float(values.std())

    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    if signed:
        span = float(np.abs(values).max())
        edges = np.linspace(-span, span, bins + 1) if span > 0 else bins
        ax.hist(values, bins=edges, density=True, color="tab:blue", alpha=0.75)
        ax.axvline(0.0, color="0.4", linewidth=1.0)
        ax.axvline(mean, color="tab:red", linewidth=1.2, label=f"mean = {mean:+.3e}")
        ax.axvline(mean + std, color="tab:orange", linestyle="--", linewidth=1.0,
                   label=f"mean ± std = {std:.3e}")
        ax.axvline(mean - std, color="tab:orange", linestyle="--", linewidth=1.0)
        ax.set_xlabel("signed error")
        ax.set_title(title or "Signed error density")
    else:
        p99 = float(np.quantile(values, 0.99))
        ax.hist(values, bins=bins, density=True, color="tab:blue", alpha=0.75)
        ax.axvline(mean, color="tab:red", linewidth=1.2, label=f"mean = {mean:.3e}")
        ax.axvline(p99, color="tab:orange", linestyle="--", linewidth=1.2, label=f"p99 = {p99:.3e}")
        ax.set_xlabel("|q_pred - q_ref|")
        ax.set_title(title or "Absolute error density")

    ax.set_ylabel("density")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(path), dpi=300)
    plt.close(fig)
