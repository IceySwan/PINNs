"""Adam / L-BFGS 训练循环与训练历史。

Trainer 只通过 ``problem.compute_losses(model, train_data)`` 取得损失，
按配置加权后驱动两个优化器：

- Adam 使用普通训练循环；
- L-BFGS 使用独立 closure，并在结束后记录诊断信息。

训练过程中只记录已 detach 的轻量数值，不做整网格预测、相对 L2 或绘图。
"""

from __future__ import annotations

import csv
import logging
import time
from pathlib import Path
from typing import Any

import torch

LOGGER = logging.getLogger("pinns")

HISTORY_FIELDS = ("stage", "evaluation", "total", "ic", "bc", "pde")


class Trainer:
    """驱动一个 case 的完整训练过程。"""

    def __init__(
        self,
        model: torch.nn.Module,
        problem: Any,
        train_data: Any,
        config: dict[str, Any],
    ) -> None:
        self.model = model
        self.problem = problem
        self.train_data = train_data

        loss_cfg = config["loss"]
        self.weights = {
            "ic": float(loss_cfg["w_ic"]),
            "bc": float(loss_cfg["w_bc"]),
            "pde": float(loss_cfg["w_pde"]),
        }

        adam_cfg = config["adam"]
        self.adam_steps = int(adam_cfg["steps"])
        self.adam = torch.optim.Adam(model.parameters(), lr=float(adam_cfg["lr"]))

        lbfgs_cfg = config["lbfgs"]
        self.lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=float(lbfgs_cfg["lr"]),
            max_iter=int(lbfgs_cfg["max_iter"]),
            max_eval=int(lbfgs_cfg["max_eval"]),
            history_size=int(lbfgs_cfg["history_size"]),
            tolerance_grad=float(lbfgs_cfg["tolerance_grad"]),
            tolerance_change=float(lbfgs_cfg["tolerance_change"]),
            line_search_fn=str(lbfgs_cfg["line_search_fn"]),
        )

        self.log_every = int(config["logging"]["every"])
        self.history: list[dict[str, Any]] = []
        self.timings: dict[str, float] = {}
        self.lbfgs_diagnostics: dict[str, Any] = {}
        self.final_losses: dict[str, float] = {}

    # ------------------------------------------------------------------
    # 基础操作
    # ------------------------------------------------------------------
    def _weighted_total(self, losses: dict[str, torch.Tensor]) -> torch.Tensor:
        """按配置加权：total = w_ic*ic + w_bc*bc + w_pde*pde。"""
        total = torch.zeros((), device=losses["pde"].device, dtype=losses["pde"].dtype)
        for key, weight in self.weights.items():
            total = total + weight * losses[key]
        return total

    def _evaluate(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """一次前向 + 反向，返回 (加权总损失, 各分项损失)。"""
        self.model.zero_grad(set_to_none=True)
        losses = self.problem.compute_losses(self.model, self.train_data)
        total = self._weighted_total(losses)
        total.backward()
        return total, losses

    def _max_abs_grad(self) -> float:
        """当前参数上的最大梯度绝对值，用于判断一阶最优条件。"""
        grads = [
            param.grad.detach().reshape(-1)
            for param in self.model.parameters()
            if param.grad is not None
        ]
        if not grads:
            return 0.0
        return float(torch.cat(grads).abs().max())

    def _record(
        self,
        stage: str,
        evaluation: int,
        total: torch.Tensor,
        losses: dict[str, torch.Tensor],
    ) -> None:
        self.history.append(
            {
                "stage": stage,
                "evaluation": evaluation,
                "total": float(total.detach()),
                "ic": float(losses["ic"].detach()),
                "bc": float(losses["bc"].detach()),
                "pde": float(losses["pde"].detach()),
            }
        )

    def _log_progress(self, label: str, evaluation: int) -> None:
        """输出已经记录的损失数值，频率由 logging.every 控制。"""
        entry = self.history[-1]
        LOGGER.info(
            "%s %6d  total_loss=%.6e  ic=%.3e  bc=%.3e  pde=%.3e",
            label,
            evaluation,
            entry["total"],
            entry["ic"],
            entry["bc"],
            entry["pde"],
        )

    # ------------------------------------------------------------------
    # 两个训练阶段
    # ------------------------------------------------------------------
    def _fit_adam(self) -> None:
        LOGGER.info(
            "Adam 阶段：%d 次更新，lr=%g", self.adam_steps, self.adam.param_groups[0]["lr"]
        )
        start = time.perf_counter()
        for step in range(1, self.adam_steps + 1):
            total, losses = self._evaluate()
            self.adam.step()
            self._record("adam", step, total, losses)

            if step % self.log_every == 0 or step == self.adam_steps:
                self._log_progress("adam update", step)
        self.timings["adam_seconds"] = time.perf_counter() - start

    def _fit_lbfgs(self) -> None:
        """L-BFGS 阶段。

        closure 只做损失计算、反向传播和轻量记录；整网格预测、相对 L2、
        绘图和文件保存都放在训练结束之后。
        """
        LOGGER.info("L-BFGS 阶段：max_iter=%d，max_eval=%d", *self._lbfgs_limits())
        calls = 0
        grad_before = None
        total_before = None

        def closure() -> torch.Tensor:
            nonlocal calls, grad_before, total_before
            calls += 1
            total, losses = self._evaluate()
            if grad_before is None:
                grad_before = self._max_abs_grad()
                total_before = float(total.detach())
            self._record("lbfgs", calls, total, losses)
            if calls % self.log_every == 0:
                self._log_progress("lbfgs evaluation", calls)
            return total

        start = time.perf_counter()
        self.lbfgs.step(closure)
        self.timings["lbfgs_seconds"] = time.perf_counter() - start

        # 在最终权重上重新计算损失和梯度，作为该阶段的最终结果。
        total, losses = self._evaluate()
        self.final_losses = {
            "total": float(total.detach()),
            **{key: float(value.detach()) for key, value in losses.items()},
        }
        grad_after = self._max_abs_grad()

        state = self.lbfgs.state[self.lbfgs._params[0]]
        self.lbfgs_diagnostics = {
            "closure_calls": calls,
            "n_iter": int(state.get("n_iter", 0)),
            "func_evals": int(state.get("func_evals", 0)),
            "grad_max_before": grad_before,
            "grad_max_after": grad_after,
            "total_before": total_before,
            "total_after": self.final_losses["total"],
            "reason": self._lbfgs_reason(state),
        }
        LOGGER.info("L-BFGS 诊断：%s", self.lbfgs_diagnostics)
        LOGGER.info(
            "L-BFGS 结束  evaluation=%d  total_loss=%.6e  ic=%.3e  bc=%.3e  pde=%.3e",
            calls,
            self.final_losses["total"],
            self.final_losses["ic"],
            self.final_losses["bc"],
            self.final_losses["pde"],
        )

    def _lbfgs_limits(self) -> tuple[int, int]:
        group = self.lbfgs.param_groups[0]
        return int(group["max_iter"]), int(group["max_eval"])

    def _lbfgs_reason(self, state: dict[str, Any]) -> str:
        """按 torch LBFGS 的停止条件判断退出原因。"""
        max_iter, max_eval = self._lbfgs_limits()
        n_iter = int(state.get("n_iter", 0))
        func_evals = int(state.get("func_evals", 0))
        if n_iter == 0:
            return "初始最优条件（max|grad| <= tolerance_grad）"
        if func_evals >= max_eval:
            return "达到 max_eval"
        if n_iter >= max_iter:
            return "达到 max_iter"
        return "停止判据（线搜索 / 步长 / 损失变化）"

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    def fit(self) -> list[dict[str, Any]]:
        """依次执行 Adam 与 L-BFGS，返回训练历史。"""
        self._fit_adam()
        self._fit_lbfgs()
        return self.history

    def write_history(self, path: str | Path) -> None:
        """保存 ``history.csv``，字段：stage,evaluation,total,ic,bc,pde。"""
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
            writer.writeheader()
            writer.writerows(self.history)

    def save_model(self, path: str | Path) -> None:
        """保存含网络权重与 ``lb/ub`` buffer 的 ``state_dict``。"""
        torch.save(self.model.state_dict(), path)
