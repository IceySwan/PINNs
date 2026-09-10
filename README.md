# PINNs

个人科研计算仓库：用 PINNs 求解一维时空可积方程。当前案例为 NLS 单孤子。

## 工作流

```text
给定方程
   ↓
MATLAB 生成参考网格与参考解（cases/<case>/matlab/generate.m）
   ↓
Python 构造训练点、训练、预测、导出（run.py）
   ↓
Python 诊断图（loss、error density）
   ↓
MATLAB 科研绘图（cases/<case>/matlab/plot_results.m）
```

MATLAB 是参考解的唯一来源，Python 不重复实现解析解。

## 环境

使用 uv 管理，Python 3.11 + PyTorch 2.5.1 CUDA 12.4。

```powershell
uv sync                     # 安装依赖
uv run python run.py nls_one_soliton --tag baseline
```

MATLAB R2024a 用于生成参考数据与结果绘图。

## 目录

```text
PINNs/
├── cases/
│   └── nls_one_soliton/
│       ├── config.toml           # 数据、采样、网络、优化器配置
│       ├── problem.py            # 方程、IC/BC 损失（唯一数学位置）
│       └── matlab/
│           ├── generate.m        # 生成参考数据
│           └── plot_results.m    # 科研绘图
├── pinns/
│   ├── model.py                  # MLP + 输入归一化
│   ├── data.py                   # 参考数据、训练点、预测、导出
│   ├── trainer.py                # Adam / L-BFGS 与训练历史
│   └── plotting.py               # loss、error density
├── data/                         # 参考 MAT
├── runs/<case>/<tag>/            # 每次实验的模型、预测、历史、图片、日志
└── run.py                        # 统一入口
```

## 一次实验

1. 生成参考数据（MATLAB）：

   ```powershell
   matlab -batch "run('cases/nls_one_soliton/matlab/generate.m')"
   ```

2. 训练与预测（Python）：

   ```powershell
   uv run python run.py nls_one_soliton --tag baseline
   ```

   输出到 `runs/nls_one_soliton/baseline/`：

   ```text
   config.toml      本次实验配置快照
   model.pt         网络权重与 lb/ub
   prediction.mat   x, t, X, T, q_ref, q_pred, abs_error
   history.csv      stage,evaluation,total,ic,bc,pde
   loss.png         Python 诊断图
   total_loss.png   加权总损失曲线
   error_density.png
   error_density_signed.png
   log.txt          环境、采样、损失、相对 L2、L-BFGS 诊断
   ```

   `total_loss = 0.5*ic + 0.25*bc + pde`，权重由 `[loss]` 配置。
   默认每 1000 次输出损失到终端和日志：Adam 按更新次数计数，L-BFGS 按 closure 评估次数计数，频率由 `[logging].every` 控制。
   L-BFGS 结束后，在最终权重上重新计算 `total_loss`、各分项损失和梯度；`history.csv` 保留每次训练评估的记录。

3. 科研绘图（MATLAB）：

   ```powershell
   matlab -batch "addpath('cases/nls_one_soliton/matlab'); plot_results('runs/nls_one_soliton/baseline')"
   ```

   输出到 `runs/nls_one_soliton/baseline/figures/`：|q| 热图、复数绝对误差热图
   `heatmap_abs_error.png`（`|q_pred-q_ref|`）、带符号幅值误差热图
   `heatmap_error.png`（`|q_pred|-|q_ref|`）、实部/虚部热图、3D 曲面、固定时刻与固定空间切片。

## 增加新方程

复制 `cases/nls_one_soliton/` 为新 case，改写 `problem.py` 中的方程与损失，
在 `matlab/generate.m` 中生成对应参考数据，调整 `config.toml` 即可。
`pinns/` 公共内核保持不变。

## 约定与方案

- 方案与验收标准见 [reflector-v3.md](reflector-v3.md)。
- 协作约定见 [AGENTS.md](AGENTS.md)。
- 旧实现的复核记录见 [reflector-v3-review.md](reflector-v3-review.md)。
- `legacy` 分支保留重构前的代码（提交 `5502f7d`）；`codex/refactor-pinns` 分支维护重构后的实现。
