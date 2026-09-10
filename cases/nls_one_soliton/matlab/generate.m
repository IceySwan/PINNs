% 生成 nls_one_soliton case 的参考数据。
%
% 方程：  i q_t + q_xx + 2 |q|^2 q = 0
% 单孤子：q(x,t) = 2 sech(2 (x + 4t)) exp(i (-2x + 1))
%
% MATLAB 是参考解的唯一来源，Python 不重复实现该公式。
% 输出 data/nls_one_soliton.mat，字段 x, t, X, T, q，其中 X, T, q 为 [Nt, Nx]。

clear; clc;

% ---------------- 计算区域与分辨率 ----------------
x0 = -5.0;   x1 = 5.0;   n_x = 513;    % 空间区间与点数
t0 = -0.5;   t1 = 0.5;   n_t = 401;    % 时间区间与点数

x = linspace(x0, x1, n_x);
t = linspace(t0, t1, n_t);
[X, T] = meshgrid(x, t);               % [Nt, Nx]

% ---------------- 参考解 ----------------
q = 2 .* sech(2 .* (X + 4 .* T)) .* exp(-2i .* X + 1i);

% ---------------- 自检 ----------------
% 孤子峰值应出现在 x = -4t，幅值为 2。
[peak, peak_col] = max(abs(q(n_t / 2 + 0.5, :)));
fprintf('grid      : x %d points on [%g, %g], t %d points on [%g, %g]\n', ...
    n_x, x0, x1, n_t, t0, t1);
fprintf('|q| range : [%.6g, %.6g]\n', min(abs(q(:))), max(abs(q(:))));
fprintf('peak      : |q| = %.6g at x = %.6g (expected 2 at x = 0)\n', ...
    peak, x(peak_col));

% ---------------- 保存 ----------------
% 输出路径按脚本位置解析：cases/<case>/matlab/generate.m -> 仓库根目录/data。
repo_root = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
out_dir = fullfile(repo_root, 'data');
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end
out_file = fullfile(out_dir, 'nls_one_soliton.mat');
save(out_file, 'x', 't', 'X', 'T', 'q', '-v7');
fprintf('saved     : %s\n', out_file);
