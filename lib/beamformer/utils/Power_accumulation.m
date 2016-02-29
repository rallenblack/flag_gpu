close all;
clearvars;

% Constants
N_bin = 50;
N_time = 25;
% N_time = 100;
N_beam = 7;

% Number of samples in power_data files
N_data = N_beam*N_bin*N_time;

% Load data
% power_acc = fopen('power_data.data', 'r');
power_acc = fopen('output.out', 'r');
[data, count] = fread(power_acc, N_data, 'float');
fclose(power_acc);

power_accumulation = reshape(data, N_bin, N_time, N_beam);
power_accumulation = permute(power_accumulation, [1, 2, 3]);
% power_accumulation = reshape(data, N_time, N_bin, N_beam);
% power_accumulation = permute(power_accumulation, [2, 1, 3]);
% power_accumulation = reshape(data, N_bin, N_beam, N_time);
% power_accumulation = permute(power_accumulation, [1, 3, 2]);
% power_accumulation = reshape(data, N_beam, N_time, N_bin);
% power_accumulation = permute(power_accumulation, [3, 2, 1]);
% power_accumulation = reshape(data, N_beam, N_bin, N_time);
% power_accumulation = permute(power_accumulation, [2, 3, 1]);
% power_accumulation = reshape(data, N_time, N_beam, N_bin);
% power_accumulation = permute(power_accumulation, [3, 1, 2]);


% Plot first frequency bin
figure();
imagesc(1:N_beam, 1:N_time, squeeze(power_accumulation(1,:,:)));
set(gca, 'ydir', 'normal');
xlabel('Beam Index');
ylabel('STI Index');
title('Frequency Bin 1');
colormap('jet');

% Plot second frequency bin
figure();
imagesc(1:N_beam, 1:N_time, squeeze(power_accumulation(2,:,:)));
set(gca, 'ydir', 'normal');
xlabel('Beam Index');
ylabel('STI Index');
title('Frequency Bin 2');
colormap('jet');

% Plot third frequency bin
figure();
imagesc(1:N_beam, 1:N_time, squeeze(power_accumulation(3,:,:)));
set(gca, 'ydir', 'normal');
xlabel('Beam Index');
ylabel('STI Index');
title('Frequency Bin 3');
colormap('jet');

% Plot last frequency bin
figure();
imagesc(1:N_beam, 1:N_time, squeeze(power_accumulation(N_bin,:,:)));
set(gca, 'ydir', 'normal');
xlabel('Beam Index');
ylabel('Time Sample Index');
title(['Frequency Bin ', num2str(N_bin)]);
colormap('jet');