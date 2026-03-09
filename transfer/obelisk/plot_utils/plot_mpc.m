%% plot_mpc_debug.m - Plot MPC solution log over time
% Set csv_path below, then run the script.
clear; clc; close all;

csv_path = '/home/wcompton/repos/robot_rl/transfer/obelisk/ctrl_logs/mpc/2026-03-01_01-40-11/mpc_log.csv';  % Set path here, or leave empty to use most recent log
% csv_path = '/home/wcompton/repos/robot_rl/transfer/obelisk/ctrl_logs/mpc/2026-03-01_01-52-24/mpc_log.csv';  % Set path here, or leave empty to use most recent log
% csv_path = '/home/wcompton/repos/robot_rl/transfer/obelisk/ctrl_logs/mpc/2026-03-01_01-56-28/mpc_log.csv';  % Set path here, or leave empty to use most recent log
% csv_path = '/home/wcompton/repos/robot_rl/transfer/obelisk/ctrl_logs/mpc/2026-03-01_02-26-49/mpc_log.csv';  % Set path here, or leave empty to use most recent log

set(groot, ...
    'defaultFigureColor', 'w', ...
    'defaultAxesColor', 'w', ...
    'defaultAxesXColor', 'k', ...
    'defaultAxesYColor', 'k', ...
    'defaultTextColor', 'k', ...
    'defaultLegendColor', 'w', ...
    'defaultLegendTextColor', 'k', ...
    'defaultLegendEdgeColor', 'k');

obs = [2.8, 2.55, 0.5, 0.5, 0;
       10.7, -0.45, 1.3, 2.4, 0.0;
       14.5, 1.55, 2.4, 1.3, 0.0;
       13, -2.25, 15, 1., 0.;
       13, 5.75, 15, 1., 0];

if isempty(csv_path)
    root = '/home/wc/repos/robot_rl/transfer/obelisk';
    log_base = fullfile(root, 'ctrl_logs', 'mpc');
    d = dir(log_base);
    d = d([d.isdir] & ~startsWith({d.name}, '.'));
    % Sort folder names (YYYY-MM-DD_HH-MM-SS) lexicographically = chronologically
    [~, idx] = sort({d.name});
    d = d(idx);
    if isempty(d)
        error('No MPC log folders found in %s', log_base);
    end
    latest_dir = fullfile(log_base, d(end).name);
    csv_path = fullfile(latest_dir, 'mpc_log.csv');
    fprintf('Using latest log: %s\n', csv_path);
end

T = readtable(csv_path);
T = T(1:end-5, :);
headers = T.Properties.VariableNames;
data = table2array(T);

n_steps = size(data, 1);

% Auto-detect horizon length from z columns (z_0_0 .. z_N_0)
z_x_cols = headers(startsWith(headers, 'z_') & endsWith(headers, '_0'));
z_x_cols = z_x_cols(~startsWith(z_x_cols, 'z_ic') & ~startsWith(z_x_cols, 'z_g'));
N = numel(z_x_cols) - 1;
fprintf('Detected horizon N = %d, %d MPC steps\n', N, n_steps);

%% Obstacle definitions (hardcoded from problems.py)
% Each row: [cx, cy, rx, ry, yaw]
% % "right" problem
% obs = [2.8, 2.55, 0.5, 0.5, 0;
%        10.7, -0.45, 1.3, 2.4, 0.0;
%        20.7, 3.95, 0.8, 1.9, 0.0;
%        13, -2.25, 15, 1.0, 0.;
%        13, 5.75, 15, 1, 0];

obs = [2.8, 2.55, 0.5, 0.5, 0;
       10.7, -0.45, 1.3, 2.4, 0.0;
       14.5, 1.55, 2.4, 1.3, 0.0;
       13, -2.25, 15, 1., 0.;
       13, 5.75, 15, 1., 0];
% "gap" problem
% obs = [4.0, 0.0, 2.0, 3.0, 0.2;
%        3.0, 6.0, 2.0, 1.0, -0.2];
% "right_wide" problem
% obs = [1.0, 1.0, 0.5, 0.5, 0.0;
%        1.25, -1.25, 0.5, 0.5, 0.0];
% No obstacles
% obs = zeros(0, 5);

%% Extract column indices
time_idx = find(strcmp(headers, 'time'));
t = data(:, time_idx);
t = t - t(1);  % relative time

ic_idx = [find(strcmp(headers, 'z_ic_0')), ...
          find(strcmp(headers, 'z_ic_1')), ...
          find(strcmp(headers, 'z_ic_2'))];
g_idx  = [find(strcmp(headers, 'z_g_0')), ...
          find(strcmp(headers, 'z_g_1')), ...
          find(strcmp(headers, 'z_g_2'))];
success_idx = find(strcmp(headers, 'success'));

% States: z_k_j for k=0..N, j=0,1,2
z_idx = zeros(N+1, 3);
for k = 0:N
    for j = 0:2
        z_idx(k+1, j+1) = find(strcmp(headers, sprintf('z_%d_%d', k, j)));
    end
end

% Controls: v_k_j for k=0..N-1, j=0,1,2
v_idx = zeros(N, 3);
for k = 0:N-1
    for j = 0:2
        v_idx(k+1, j+1) = find(strcmp(headers, sprintf('v_%d_%d', k, j)));
    end
end

ic_data = data(:, ic_idx);       % n_steps x 3
goal_data = data(:, g_idx);      % n_steps x 3
success_data = data(:, success_idx);

% Applied control (first control in horizon)
v0_idx = v_idx(1, :);
v0_data = data(:, v0_idx);  % n_steps x 3

%% Figure 1: XY plan evolution over time
deadzone = 3;  % goal region radius [m]

figure('Name', 'MPC Plans Over Time', 'Position', [100 100 700 600]);
hold on;
% Color trajectory by commanded vx
cmd_vx = v0_data(:,1);
cmap_traj = parula(256);
vx_min = min(cmd_vx); vx_max = max(cmd_vx);
if vx_max == vx_min; vx_max = vx_min + 1; end  % avoid division by zero
% Plot MPC plans (light gray)
for i = 1:n_steps
    z_i = data(i, z_idx(:));
    z_i = reshape(z_i, N+1, 3);
    plot(z_i(:,1), z_i(:,2), '-', 'Color', [0.7 0.7 0.7 0.3], 'HandleVisibility', 'off');
end
% Robot path colored by commanded vx
for i = 1:n_steps-1
    ci = round((cmd_vx(i) - vx_min) / (vx_max - vx_min) * 255) + 1;
    ci = max(1, min(256, ci));
    plot(ic_data(i:i+1, 1), ic_data(i:i+1, 2), '-', 'Color', cmap_traj(ci,:), 'LineWidth', 2, ...
        'HandleVisibility', 'off');
end
% Start marker
plot(ic_data(1,1), ic_data(1,2), 'gs', 'MarkerSize', 12, 'MarkerFaceColor', 'g', 'HandleVisibility', 'off');
% Goal region circle
th_goal = linspace(0, 2*pi, 100);
gx = goal_data(1,1) + deadzone * cos(th_goal);
gy = goal_data(1,2) + deadzone * sin(th_goal);
fill(gx, gy, [0.2 0.8 0.2], 'FaceAlpha', 0.15, 'EdgeColor', 'g', 'LineStyle', '--', ...
    'LineWidth', 1.5, 'HandleVisibility', 'off');
plot(goal_data(1,1), goal_data(1,2), 'g+', 'MarkerSize', 10, 'LineWidth', 1.5, 'HandleVisibility', 'off');
% Heading arrows
arr_len = 0.3;
quiver(ic_data(1,1), ic_data(1,2), arr_len*cos(ic_data(1,3)), arr_len*sin(ic_data(1,3)), 0, ...
    'g', 'MaxHeadSize', 0.8, 'HandleVisibility', 'off');
% Mark failed solves
fail_idx = find(~success_data);
if ~isempty(fail_idx)
    plot(ic_data(fail_idx, 1), ic_data(fail_idx, 2), 'rx', 'MarkerSize', 8, ...
        'HandleVisibility', 'off');
end
% Plot obstacle ellipses
for oi = 1:size(obs, 1)
    [ex, ey] = ellipse_pts(obs(oi,1), obs(oi,2), obs(oi,3), obs(oi,4), obs(oi,5));
    fill(ex, ey, [1 0.2 0.2], 'FaceAlpha', 0.3, 'EdgeColor', 'r', ...
        'HandleVisibility', 'off');
end
xlabel('x'); ylabel('y');
title(sprintf('MPC Plans (%d steps)', n_steps));
axis equal; grid on;
% Compute tight limits from trajectory, goal region, and obstacles
all_x = [ic_data(:,1); goal_data(1,1) + deadzone; goal_data(1,1) - deadzone];
all_y = [ic_data(:,2); goal_data(1,2) + deadzone; goal_data(1,2) - deadzone];
for oi = 1:size(obs, 1)
    [ex_tmp, ey_tmp] = ellipse_pts(obs(oi,1), obs(oi,2), obs(oi,3), obs(oi,4), obs(oi,5));
    all_x = [all_x; ex_tmp(:)]; %#ok<AGROW>
    all_y = [all_y; ey_tmp(:)]; %#ok<AGROW>
end
pad = 1.0;
x_range = [min(all_x) - pad, max(all_x) + pad];
y_range = [min(all_y) - pad, max(all_y) + pad];
% Expand to 2.5:1 aspect ratio (keeping axis equal)
dx = diff(x_range); dy = diff(y_range);
if dx / dy < 2.5
    % Need wider x range
    new_dx = 2.5 * dy;
    x_mid = mean(x_range);
    x_range = [x_mid - new_dx/2, x_mid + new_dx/2];
else
    % Need taller y range
    new_dy = dx / 2.5;
    y_mid = mean(y_range);
    y_range = [y_mid - new_dy/2, y_mid + new_dy/2];
end
xlim(x_range); ylim(y_range);
% Set figure size to match 2.5:1
set(gcf, 'Position', [100 100 1200 480]);
colormap(parula(256));
clim([vx_min vx_max]);
cb = colorbar; cb.Label.String = 'Commanded v_x (m/s)';
hold off;

%% Figure 1b: XY plan rotated CCW 90deg (plot -y as horizontal, x as vertical)
figure('Name', 'MPC Plans Over Time (Rotated)', 'Position', [100 100 480 1200]);
hold on;
% Plot MPC plans (light gray)
for i = 1:n_steps
    z_i = data(i, z_idx(:));
    z_i = reshape(z_i, N+1, 3);
    plot(-z_i(:,2), z_i(:,1), '-', 'Color', [0.7 0.7 0.7 0.3], 'HandleVisibility', 'off');
end
% Robot path colored by commanded vx
for i = 1:n_steps-1
    ci = round((cmd_vx(i) - vx_min) / (vx_max - vx_min) * 255) + 1;
    ci = max(1, min(256, ci));
    plot(-ic_data(i:i+1, 2), ic_data(i:i+1, 1), '-', 'Color', cmap_traj(ci,:), 'LineWidth', 2, ...
        'HandleVisibility', 'off');
end
% Start marker
plot(-ic_data(1,2), ic_data(1,1), 'gs', 'MarkerSize', 12, 'MarkerFaceColor', 'g', 'HandleVisibility', 'off');
% Goal region circle
gx_r = -goal_data(1,2) + deadzone * cos(th_goal);
gy_r = goal_data(1,1) + deadzone * sin(th_goal);
fill(gx_r, gy_r, [0.2 0.8 0.2], 'FaceAlpha', 0.15, 'EdgeColor', 'g', 'LineStyle', '--', ...
    'LineWidth', 1.5, 'HandleVisibility', 'off');
plot(-goal_data(1,2), goal_data(1,1), 'g+', 'MarkerSize', 10, 'LineWidth', 1.5, 'HandleVisibility', 'off');
% Heading arrow (rotated: dx'=-dy, dy'=dx)
quiver(-ic_data(1,2), ic_data(1,1), -arr_len*sin(ic_data(1,3)), arr_len*cos(ic_data(1,3)), 0, ...
    'g', 'MaxHeadSize', 0.8, 'HandleVisibility', 'off');
% Mark failed solves
if ~isempty(fail_idx)
    plot(-ic_data(fail_idx, 2), ic_data(fail_idx, 1), 'rx', 'MarkerSize', 8, ...
        'HandleVisibility', 'off');
end
% Plot obstacle ellipses
for oi = 1:size(obs, 1)
    [ex, ey] = ellipse_pts(obs(oi,1), obs(oi,2), obs(oi,3), obs(oi,4), obs(oi,5));
    fill(-ey, ex, [1 0.2 0.2], 'FaceAlpha', 0.3, 'EdgeColor', 'r', ...
        'HandleVisibility', 'off');
end
xlabel('-y'); ylabel('x');
title(sprintf('MPC Plans (%d steps) [Rotated]', n_steps));
axis equal; grid on;
% Rotated limits: horizontal = -y_range (flipped), vertical = x_range
xlim([-y_range(2), -y_range(1)]); ylim(x_range);
% Set figure size to 1:2.5 (tall)
set(gcf, 'Position', [100 100 480 1200]);
colormap(parula(256));
clim([vx_min vx_max]);
cb = colorbar('Location', 'southoutside'); cb.Label.String = 'Commanded v_x (m/s)';
hold off;

%% Figure 2: State vs time
figure('Name', 'MPC State vs Time', 'Position', [100 100 700 500]);

subplot(3,1,1);
plot(t, ic_data(:,1), 'b-');
hold on; yline(goal_data(1,1), 'r--', 'Goal');
ylabel('x'); grid on; title('Robot State vs Time');

subplot(3,1,2);
plot(t, ic_data(:,2), 'b-');
hold on; yline(goal_data(1,2), 'r--', 'Goal');
ylabel('y'); grid on;

subplot(3,1,3);
plot(t, rad2deg(ic_data(:,3)), 'b-');
hold on; yline(rad2deg(goal_data(1,3)), 'r--', 'Goal');
ylabel('\theta (deg)'); xlabel('Time (s)'); grid on;

%% Figure 3: Applied control (v_0) vs time
figure('Name', 'MPC Applied Control vs Time', 'Position', [100 100 700 500]);

subplot(3,1,1);
plot(t, v0_data(:,1), 'r-');
ylabel('v_x'); grid on; title('Applied Control vs Time');

subplot(3,1,2);
plot(t, v0_data(:,2), 'r-');
ylabel('v_y'); grid on;

subplot(3,1,3);
plot(t, v0_data(:,3), 'r-');
ylabel('\omega_z'); xlabel('Time (s)'); grid on;

%% Figure 4: Interactive plan browser (arrow keys to navigate)
fig4 = figure('Name', 'MPC Plan Browser', 'Position', [100 100 800 600]);

k_z = 0:N;
k_v = 0:N-1;

% Store all data needed by the callback in the figure's UserData
ud.data = data;
ud.z_idx = z_idx;
ud.v_idx = v_idx;
ud.ic_data = ic_data;
ud.goal_data = goal_data;
ud.success_data = success_data;
ud.t = t;
ud.N = N;
ud.n_steps = n_steps;
ud.k_z = k_z;
ud.k_v = k_v;
ud.step = n_steps;  % start at last step
fig4.UserData = ud;

draw_plan(fig4);
fig4.KeyPressFcn = @on_key_press;

%% Summary
n_fail = sum(~success_data);
fprintf('Plotted %d MPC steps (N=%d), %d failed solves (%.1f%%)\n', ...
    n_steps, N, n_fail, 100*n_fail/n_steps);
fprintf('Use left/right arrow keys in Figure 4 to browse plans.\n');

%% ---- Helper functions ----

function on_key_press(fig, evt)
    ud = fig.UserData;
    switch evt.Key
        case 'rightarrow'
            ud.step = min(ud.step + 1, ud.n_steps);
        case 'leftarrow'
            ud.step = max(ud.step - 1, 1);
        case 'home'
            ud.step = 1;
        case 'end'
            ud.step = ud.n_steps;
        otherwise
            return;
    end
    fig.UserData = ud;
    draw_plan(fig);
end

function draw_plan(fig)
    ud = fig.UserData;
    i = ud.step;

    z_i = ud.data(i, ud.z_idx(:));
    z_i = reshape(z_i, ud.N+1, 3);
    v_i = ud.data(i, ud.v_idx(:));
    v_i = reshape(v_i, ud.N, 3);
    ic_i = ud.ic_data(i, :);
    g_i = ud.goal_data(i, :);
    ok = ud.success_data(i);

    if ok
        status_str = 'converged';
    else
        status_str = 'FAILED';
    end
    title_str = sprintf('Step %d / %d  (t = %.2f s)  [%s]', i, ud.n_steps, ud.t(i), status_str);

    subplot(3,2,1);
    cla; plot(ud.k_z, z_i(:,1), 'b-o', 'MarkerSize', 3);
    hold on; yline(ic_i(1), 'g--', 'IC'); yline(g_i(1), 'r--', 'Goal');
    ylabel('x'); grid on; title(title_str);

    subplot(3,2,3);
    cla; plot(ud.k_z, z_i(:,2), 'b-o', 'MarkerSize', 3);
    hold on; yline(ic_i(2), 'g--', 'IC'); yline(g_i(2), 'r--', 'Goal');
    ylabel('y'); grid on;

    subplot(3,2,5);
    cla; plot(ud.k_z, rad2deg(z_i(:,3)), 'b-o', 'MarkerSize', 3);
    hold on; yline(rad2deg(ic_i(3)), 'g--', 'IC'); yline(rad2deg(g_i(3)), 'r--', 'Goal');
    ylabel('\theta (deg)'); xlabel('Horizon step k'); grid on;

    subplot(3,2,2);
    cla; plot(ud.k_v, v_i(:,1), 'r-o', 'MarkerSize', 3);
    ylabel('v_x'); grid on; title('Controls');

    subplot(3,2,4);
    cla; plot(ud.k_v, v_i(:,2), 'r-o', 'MarkerSize', 3);
    ylabel('v_y'); grid on;

    subplot(3,2,6);
    cla; plot(ud.k_v, v_i(:,3), 'r-o', 'MarkerSize', 3);
    ylabel('\omega_z'); xlabel('Horizon step k'); grid on;

    drawnow;
end

function [ex, ey] = ellipse_pts(cx, cy, rx, ry, yaw)
% Generate points for a rotated ellipse.
    th = linspace(0, 2*pi, 100);
    px = rx * cos(th);
    py = ry * sin(th);
    R = [cos(yaw), -sin(yaw); sin(yaw), cos(yaw)];
    pts = R * [px; py];
    ex = pts(1,:) + cx;
    ey = pts(2,:) + cy;
end