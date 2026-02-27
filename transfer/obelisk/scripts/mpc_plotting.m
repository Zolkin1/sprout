%% plot_mpc_debug.m - Plot MPC solution log over time
% Set csv_path below, then run the script.
clear; clc; close all;

csv_path = '';  % Set path here, or leave empty to use most recent log

if isempty(csv_path)
    root = getenv('ROBOT_RL_ROOT');
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
headers = T.Properties.VariableNames;
data = table2array(T);

n_steps = size(data, 1);

% Auto-detect horizon length from z columns (z_0_0 .. z_N_0)
z_x_cols = headers(startsWith(headers, 'z_') & endsWith(headers, '_0'));
z_x_cols = z_x_cols(~startsWith(z_x_cols, 'z_ic') & ~startsWith(z_x_cols, 'z_g'));
N = numel(z_x_cols) - 1;
fprintf('Detected horizon N = %d, %d MPC steps\n', N, n_steps);

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

%% Figure 1: XY plan evolution over time
figure('Name', 'MPC Plans Over Time', 'Position', [100 100 700 600]);
hold on;
cmap = parula(n_steps);
for i = 1:n_steps
    z_i = data(i, z_idx(:));
    z_i = reshape(z_i, N+1, 3);
    plot(z_i(:,1), z_i(:,2), '-', 'Color', [cmap(i,:) 0.3], 'LineWidth', 0.5);
end
% Robot path (actual trajectory)
plot(ic_data(:,1), ic_data(:,2), 'k-', 'LineWidth', 2, 'DisplayName', 'Robot path');
% Start and goal
plot(ic_data(1,1), ic_data(1,2), 'gs', 'MarkerSize', 12, 'MarkerFaceColor', 'g', 'DisplayName', 'Start');
plot(goal_data(1,1), goal_data(1,2), 'rp', 'MarkerSize', 14, 'MarkerFaceColor', 'r', 'DisplayName', 'Goal');
% Heading arrows
arr_len = 0.3;
quiver(ic_data(1,1), ic_data(1,2), arr_len*cos(ic_data(1,3)), arr_len*sin(ic_data(1,3)), 0, ...
    'g', 'LineWidth', 2, 'MaxHeadSize', 0.8, 'HandleVisibility', 'off');
quiver(goal_data(1,1), goal_data(1,2), arr_len*cos(goal_data(1,3)), arr_len*sin(goal_data(1,3)), 0, ...
    'r', 'LineWidth', 2, 'MaxHeadSize', 0.8, 'HandleVisibility', 'off');
% Mark failed solves
fail_idx = find(~success_data);
if ~isempty(fail_idx)
    plot(ic_data(fail_idx, 1), ic_data(fail_idx, 2), 'rx', 'MarkerSize', 8, ...
        'LineWidth', 2, 'DisplayName', 'Failed solve');
end
xlabel('x'); ylabel('y');
title(sprintf('MPC Plans (%d steps)', n_steps));
axis equal; grid on;
cb = colorbar; cb.Label.String = 'Time (s)';
colormap(parula(n_steps));
clim([t(1) t(end)]);
legend('Location', 'best');
hold off;

%% Figure 2: State vs time
figure('Name', 'MPC State vs Time', 'Position', [100 100 700 500]);

subplot(3,1,1);
plot(t, ic_data(:,1), 'b-', 'LineWidth', 1.5);
hold on; yline(goal_data(1,1), 'r--', 'Goal');
ylabel('x'); grid on; title('Robot State vs Time');

subplot(3,1,2);
plot(t, ic_data(:,2), 'b-', 'LineWidth', 1.5);
hold on; yline(goal_data(1,2), 'r--', 'Goal');
ylabel('y'); grid on;

subplot(3,1,3);
plot(t, rad2deg(ic_data(:,3)), 'b-', 'LineWidth', 1.5);
hold on; yline(rad2deg(goal_data(1,3)), 'r--', 'Goal');
ylabel('\theta (deg)'); xlabel('Time (s)'); grid on;

%% Figure 3: Applied control (v_0) vs time
figure('Name', 'MPC Applied Control vs Time', 'Position', [100 100 700 500]);

% v_0 at each step = first control in the horizon
v0_idx = v_idx(1, :);
v0_data = data(:, v0_idx);  % n_steps x 3

subplot(3,1,1);
plot(t, v0_data(:,1), 'r-', 'LineWidth', 1.5);
ylabel('v_x'); grid on; title('Applied Control vs Time');

subplot(3,1,2);
plot(t, v0_data(:,2), 'r-', 'LineWidth', 1.5);
ylabel('v_y'); grid on;

subplot(3,1,3);
plot(t, v0_data(:,3), 'r-', 'LineWidth', 1.5);
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
    cla; plot(ud.k_z, z_i(:,1), 'b-o', 'LineWidth', 1.5, 'MarkerSize', 3);
    hold on; yline(ic_i(1), 'g--', 'IC'); yline(g_i(1), 'r--', 'Goal');
    ylabel('x'); grid on; title(title_str);

    subplot(3,2,3);
    cla; plot(ud.k_z, z_i(:,2), 'b-o', 'LineWidth', 1.5, 'MarkerSize', 3);
    hold on; yline(ic_i(2), 'g--', 'IC'); yline(g_i(2), 'r--', 'Goal');
    ylabel('y'); grid on;

    subplot(3,2,5);
    cla; plot(ud.k_z, rad2deg(z_i(:,3)), 'b-o', 'LineWidth', 1.5, 'MarkerSize', 3);
    hold on; yline(rad2deg(ic_i(3)), 'g--', 'IC'); yline(rad2deg(g_i(3)), 'r--', 'Goal');
    ylabel('\theta (deg)'); xlabel('Horizon step k'); grid on;

    subplot(3,2,2);
    cla; plot(ud.k_v, v_i(:,1), 'r-o', 'LineWidth', 1.5, 'MarkerSize', 3);
    ylabel('v_x'); grid on; title('Controls');

    subplot(3,2,4);
    cla; plot(ud.k_v, v_i(:,2), 'r-o', 'LineWidth', 1.5, 'MarkerSize', 3);
    ylabel('v_y'); grid on;

    subplot(3,2,6);
    cla; plot(ud.k_v, v_i(:,3), 'r-o', 'LineWidth', 1.5, 'MarkerSize', 3);
    ylabel('\omega_z'); xlabel('Horizon step k'); grid on;

    drawnow;
end