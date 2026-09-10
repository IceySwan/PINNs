"""PINNs 公共内核。

模块划分：

- ``model``    神经网络（MLP + 输入归一化）
- ``data``     参考数据读取、训练点构造、预测与结果导出
- ``trainer``  Adam / L-BFGS 训练循环与历史记录
- ``plotting`` Python 诊断图（loss、error density）

具体方程的数学逻辑放在 ``cases/<case>/problem.py``。各模块按需显式导入，
例如 ``from pinns.model import MLP``。
"""
