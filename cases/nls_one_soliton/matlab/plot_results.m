function plot_results(run_dir)
%PLOT_RESULTS  为一次实验生成科研图（heatmap、3D、切片）。
%
%   plot_results                    % 使用 <repo>/runs/nls_one_soliton/baseline
%   plot_results(run_dir)           % 使用指定实验目录（需含 prediction.mat）
%
% 输入 prediction.mat 的字段：x, t, X, T, q_ref, q_pred, abs_error。
% 输出写入 <run_dir>/figures/ 下的 PNG 文件。
%
% 用法示例（PowerShell）：
%   matlab -batch "addpath('cases/nls_one_soliton/matlab'); plot_results('runs/nls_one_soliton/baseline')"

if nargin < 1 || isempty(run_dir)
    here = fileparts(mfilename('fullpath'));            % cases/<case>/matlab
    repo_root = fileparts(fileparts(fileparts(here)));  % 仓库根目录
    run_dir = fullfile(repo_root, 'runs', 'nls_one_soliton', 'baseline');
end

mat_file = fullfile(run_dir, 'prediction.mat');
if ~isfile(mat_file)
    error('找不到 %s，请先运行 run.py 生成预测。', mat_file);
end
fig_dir = fullfile(run_dir, 'figures');
if ~isfolder(fig_dir)
    mkdir(fig_dir);
end

% ---------------- 读取数据 ----------------
data = load(mat_file);
X = data.X;  T = data.T;
x = data.x(:);  t = data.t(:);
q_ref = data.q_ref;  q_pred = data.q_pred;  err = data.abs_error;

mag_ref = abs(q_ref);   mag_pred = abs(q_pred);
re_ref = real(q_ref);   im_ref = imag(q_ref);
re_pred = real(q_pred); im_pred = imag(q_pred);

% 与 Python 侧一致的相对 L2，作为交叉检查
l2_complex = norm(q_pred(:) - q_ref(:)) / norm(q_ref(:));
l2_mag = norm(mag_pred(:) - mag_ref(:)) / norm(mag_ref(:));
fprintf('run_dir            : %s\n', run_dir);
fprintf('relative L2 complex: %.6e\n', l2_complex);
fprintf('relative L2 mag    : %.6e\n', l2_mag);
fprintf('max abs error      : %.6e\n', max(err(:)));

% ---------------- 1. |q| heatmap：参考 vs 预测 ----------------
fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 1100 450]);
tiledlayout(fig, 1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

nexttile;
contourf(X, T, mag_ref, 40, 'LineColor', 'none');
colorbar; colormap(gca, 'parula');
xlabel('x'); ylabel('t'); title('|q_{ref}|'); axis tight; yticks(-0.5:0.25:0.5);

nexttile;
contourf(X, T, mag_pred, 40, 'LineColor', 'none');
colorbar; colormap(gca, 'parula');
xlabel('x'); ylabel('t'); title('|q_{pred}|'); axis tight; yticks(-0.5:0.25:0.5);

exportgraphics(fig, fullfile(fig_dir, 'heatmap_magnitude.png'), 'Resolution', 300);
close(fig);

% ---------------- 2. 复数绝对误差 heatmap ----------------
% abs_error = |q_pred - q_ref|，同时反映复数场的幅值与相位偏差。
fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 700 450]);
contourf(X, T, err, 40, 'LineColor', 'none');
colorbar; colormap(gca, 'parula'); clim([0, max(err(:))]);
xlabel('x'); ylabel('t'); title('|q_{pred} - q_{ref}|'); axis tight; yticks(-0.5:0.25:0.5);
exportgraphics(fig, fullfile(fig_dir, 'heatmap_abs_error.png'), 'Resolution', 300);
close(fig);

% ---------------- 3. 带符号幅值误差 heatmap ----------------
% 数据为 |q_pred| - |q_ref|，正值表示预测幅值偏高。
fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 700 450]);
contourf(X, T, abs(q_pred) - abs(q_ref), 40, 'LineColor', 'none');
colorbar; colormap(gca, 'parula');
xlabel('x'); ylabel('t'); title('|q_{pred}| - |q_{ref}|'); axis tight; yticks(-0.5:0.25:0.5);
exportgraphics(fig, fullfile(fig_dir, 'heatmap_error.png'), 'Resolution', 300);
close(fig);

% ---------------- 4. 实部 / 虚部 heatmap ----------------
fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 1100 800]);
tiledlayout(fig, 2, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

nexttile; contourf(X, T, re_ref, 40, 'LineColor', 'none'); colorbar;
colormap(gca, 'parula'); xlabel('x'); ylabel('t'); title('Re(q_{ref})'); axis tight; yticks(-0.5:0.25:0.5);
nexttile; contourf(X, T, re_pred, 40, 'LineColor', 'none'); colorbar;
colormap(gca, 'parula'); xlabel('x'); ylabel('t'); title('Re(q_{pred})'); axis tight; yticks(-0.5:0.25:0.5);
nexttile; contourf(X, T, im_ref, 40, 'LineColor', 'none'); colorbar;
colormap(gca, 'parula'); xlabel('x'); ylabel('t'); title('Im(q_{ref})'); axis tight; yticks(-0.5:0.25:0.5);
nexttile; contourf(X, T, im_pred, 40, 'LineColor', 'none'); colorbar;
colormap(gca, 'parula'); xlabel('x'); ylabel('t'); title('Im(q_{pred})'); axis tight; yticks(-0.5:0.25:0.5);

exportgraphics(fig, fullfile(fig_dir, 'heatmap_real_imag.png'), 'Resolution', 300);
close(fig);

% ---------------- 5. 3D 曲面：参考 vs 预测 ----------------
fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 1100 450]);
tiledlayout(fig, 1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

nexttile;
surf(X, T, mag_ref, 'EdgeColor', 'none'); shading interp; colormap(gca, 'parula');
xlabel('x'); ylabel('t'); zlabel('|q|'); title('3D |q_{ref}|');
view(-45, 30); axis tight; colorbar;

nexttile;
surf(X, T, mag_pred, 'EdgeColor', 'none'); shading interp; colormap(gca, 'parula');
xlabel('x'); ylabel('t'); zlabel('|q|'); title('3D |q_{pred}|');
view(-45, 30); axis tight; colorbar;

exportgraphics(fig, fullfile(fig_dir, 'surface_magnitude.png'), 'Resolution', 300);
close(fig);

% ---------------- 6. 固定时刻切片 ----------------
slice_t = [-0.25, 0.0, 0.25];
fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 1200 400]);
tiledlayout(fig, 1, numel(slice_t), 'Padding', 'compact', 'TileSpacing', 'compact');
for k = 1:numel(slice_t)
    [~, idx] = min(abs(t - slice_t(k)));
    nexttile;
    plot(x, mag_ref(idx, :), 'b-', 'LineWidth', 1.2); hold on;
    plot(x, mag_pred(idx, :), 'r--', 'LineWidth', 1.2);
    grid on; xlabel('x'); ylabel('|q|');
    title(sprintf('|q| at t = %.2f', t(idx)));
    legend('ref', 'pred', 'Location', 'best');
end
exportgraphics(fig, fullfile(fig_dir, 'slice_time.png'), 'Resolution', 300);
close(fig);

% ---------------- 7. 固定空间位置切片 ----------------
[~, x0_idx] = min(abs(x));
fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 700 450]);
plot(t, mag_ref(:, x0_idx), 'b-', 'LineWidth', 1.2); hold on;
plot(t, mag_pred(:, x0_idx), 'r--', 'LineWidth', 1.2);
grid on; xlabel('t'); ylabel('|q|');
title(sprintf('|q| at x = %.2f', x(x0_idx)));
legend('ref', 'pred', 'Location', 'best');
exportgraphics(fig, fullfile(fig_dir, 'slice_space.png'), 'Resolution', 300);
close(fig);

fprintf('figures saved to   : %s\n', fig_dir);
end
