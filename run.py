"""统一入口。

用法：

    uv run python run.py nls_one_soliton --tag baseline

流程：读取 case 配置 → 读取 MATLAB 参考数据 → 构造训练点 → 训练（Adam + L-BFGS）
→ 全网格预测 → 保存模型、预测、历史、配置快照与诊断图。

本文件只负责串联流程，不包含任何方程数学。
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tomllib

from pinns.data import (
    build_training_data,
    load_reference,
    predict_grid,
    relative_l2,
    save_prediction,
    sha256_file,
)
from pinns.model import MLP
from pinns.plotting import plot_error_density, plot_loss
from pinns.trainer import Trainer

REPO = Path(__file__).resolve().parent
LOGGER = logging.getLogger("pinns")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行一个 PINNs case")
    parser.add_argument("case", help="cases/ 下的 case 名称，例如 nls_one_soliton")
    parser.add_argument("--tag", default=None, help="实验 tag；默认取配置中的 [experiment].tag")
    return parser.parse_args()


def force_utf8_streams() -> None:
    """Windows 控制台默认代码页可能无法编码中文，统一把标准输出与错误切到 UTF-8。"""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def setup_logging(log_path: Path) -> None:
    """日志同时写文件和终端。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )


def load_config(path: Path) -> dict[str, Any]:
    with open(path, "rb") as handle:
        return tomllib.load(handle)


def load_problem(path: Path) -> Any:
    """按路径动态导入 case 的 problem.py。"""
    spec = importlib.util.spec_from_file_location(f"case_{path.parent.name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git_commit() -> str:
    """当前 Git commit，用于实验记录。"""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def build_model(config: dict[str, Any], ref: Any, device: torch.device, dtype: torch.dtype) -> MLP:
    """按配置和参考网格边界构造网络。"""
    model_cfg = config["model"]
    model = MLP(
        in_dim=2,
        out_dim=2,
        hidden_layers=int(model_cfg["hidden_layers"]),
        width=int(model_cfg["width"]),
        activation=str(model_cfg["activation"]),
        lb=ref.lb,
        ub=ref.ub,
    )
    return model.to(device=device, dtype=dtype)


def main() -> None:
    force_utf8_streams()
    args = parse_args()
    case_dir = REPO / "cases" / args.case
    if not case_dir.is_dir():
        sys.exit(f"未找到 case 目录：{case_dir}")

    config = load_config(case_dir / "config.toml")
    tag = args.tag or str(config["experiment"]["tag"])
    run_dir = REPO / "runs" / args.case / tag
    if run_dir.exists():
        sys.exit(f"实验目录已存在，请更换 tag：{run_dir}")
    run_dir.mkdir(parents=True)

    setup_logging(run_dir / "log.txt")
    start = time.perf_counter()

    # ---- 环境与实验记录 ----
    dtype = torch.float64 if config["model"]["dtype"] == "float64" else torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False  # TF32 影响 float32 matmul，关闭以保持数值一致
    torch.backends.cudnn.allow_tf32 = False

    LOGGER.info("case=%s tag=%s", args.case, tag)
    LOGGER.info("git=%s python=%s torch=%s", git_commit(), sys.version.split()[0], torch.__version__)
    lock_path = REPO / "uv.lock"
    if lock_path.exists():
        LOGGER.info("uv.lock sha256=%s", sha256_file(lock_path)[:16])
    LOGGER.info("device=%s dtype=%s", device, dtype)
    if torch.cuda.is_available():
        LOGGER.info("gpu=%s cuda=%s", torch.cuda.get_device_name(0), torch.version.cuda)

    seed = int(config["sampling"]["seed"])
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ---- 参考数据 ----
    data_path = REPO / config["data"]["file"]
    ref = load_reference(data_path)
    LOGGER.info("reference=%s sha256=%s", config["data"]["file"], sha256_file(data_path))
    LOGGER.info(
        "grid: X %s  x [%g, %g] (%d)  t [%g, %g] (%d)",
        ref.X.shape,
        ref.x[0],
        ref.x[-1],
        ref.x.size,
        ref.t[0],
        ref.t[-1],
        ref.t.size,
    )
    LOGGER.info("domain from reference: lb=%s ub=%s", ref.lb.tolist(), ref.ub.tolist())

    train_data = build_training_data(ref, config, device, dtype)
    LOGGER.info("sampling: %s", train_data.meta)

    # ---- 网络与方程 ----
    model = build_model(config, ref, device, dtype)
    problem = load_problem(case_dir / "problem.py")
    LOGGER.info(
        "model: %d 隐藏层 x %d 节点, 激活=%s, 参数量=%d",
        config["model"]["hidden_layers"],
        config["model"]["width"],
        config["model"]["activation"],
        sum(p.numel() for p in model.parameters()),
    )

    # ---- 训练 ----
    trainer = Trainer(model=model, problem=problem, train_data=train_data, config=config)
    trainer.fit()
    LOGGER.info(
        "timings: adam %.1f s, lbfgs %.1f s",
        trainer.timings["adam_seconds"],
        trainer.timings["lbfgs_seconds"],
    )

    # ---- 预测与输出 ----
    q_pred = predict_grid(model, ref, device, dtype)
    l2_complex, l2_mag = relative_l2(q_pred, ref.q)
    abs_error = save_prediction(run_dir / "prediction.mat", ref, q_pred)
    trainer.save_model(run_dir / "model.pt")
    trainer.write_history(run_dir / "history.csv")
    shutil.copyfile(case_dir / "config.toml", run_dir / "config.toml")
    plot_loss(trainer.history, run_dir / "loss.png")
    plot_loss(
        trainer.history,
        run_dir / "total_loss.png",
        title="Total loss",
        series=("total",),
    )

    # 误差分布：abs 为 |q_pred - q_ref|；signed 为幅值误差 |q_pred| - |q_ref|，
    # 正值表示预测幅值偏大。若只想保留带符号版本，删掉下面第一行即可。
    signed_error = np.abs(q_pred) - np.abs(ref.q)
    plot_error_density(abs_error, run_dir / "error_density.png")
    plot_error_density(signed_error, run_dir / "error_density_signed.png", signed=True)

    LOGGER.info("relative L2 (complex)   = %.6e", l2_complex)
    LOGGER.info("relative L2 (magnitude) = %.6e", l2_mag)
    LOGGER.info(
        "abs error: max=%.3e  mean=%.3e  p99=%.3e",
        float(abs_error.max()),
        float(abs_error.mean()),
        float(np.quantile(abs_error, 0.99)),
    )
    LOGGER.info("outputs: %s", run_dir)
    LOGGER.info("total seconds: %.1f", time.perf_counter() - start)


if __name__ == "__main__":
    main()
