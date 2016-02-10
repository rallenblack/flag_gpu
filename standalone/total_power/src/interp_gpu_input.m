% read in data going into GPU
clearvars;
close all;

% FILE = fopen('int8_tra_in.out', 'r');
FILE = fopen('int8_GPUin.out', 'r');
[R, count] = fscanf(FILE, '%g\n');
fclose(FILE);
tmp = find(R ~= 0);
tmp(1:15)