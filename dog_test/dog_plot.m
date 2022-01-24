clear all
close all
clc
dbstop if error
load limbSeq.mat

addpath(genpath('../../Dynamic_3D_Reconstruction/'))
load('../Vardata/dog2.mat')
%limbSeq  = [[0, 1]; [1, 2]; [2, 3]; [0, 4]; [4, 5]; [5, 6]; [0, 7]; [7, 8]; [8, 9]; [9, 10]; [8, 14]; [14, 15]; [15, 16]; [8, 11]; [11, 12]; [12, 13]] +1;
X = pickle_data{2};
AnimatePlot(X(:,30:end), limbSeq_plot)

joints = 43;
colors = hsv(length(limbSeq_plot));
hold on
for f = 30:30:120
points_temp = reshape(X(:,f),[3,joints]);
plot3(points_temp(1,:) , points_temp(2,:),  points_temp(3,:), '.', 'MarkerSize', 10, 'MarkerEdgeColor', 'k');
num_lines = size(limbSeq_plot,1);
lines_x = reshape(points_temp(1,limbSeq_plot), [num_lines,2]);
lines_y = reshape(points_temp(2,limbSeq_plot), [num_lines,2]);
lines_z = reshape(points_temp(3,limbSeq_plot), [num_lines,2]);
lines_plot = cell(num_lines,1);
%trajectory = cell(njoints,1);
for p = 1:num_lines
line(lines_x(p,:), lines_y(p,:), lines_z(p,:), 'Color', colors(p,:), 'LineWidth', 5);
end
end