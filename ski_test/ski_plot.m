clear all
close all
clc
dbstop if error

addpath(genpath('../../Dynamic_3D_Reconstruction/'))
load('../Vardata/ski_init.mat')
limbSeq  = [[0, 1]; [1, 2]; [2, 3]; [0, 4]; [4, 5]; [5, 6]; [0, 7]; [7, 8]; [8, 9]; [9, 10]; [8, 14]; [14, 15]; [15, 16]; [8, 11]; [11, 12]; [12, 13]] +1;
X = pickle_data;%pickle_data{2};
AnimatePlot(X, limbSeq)