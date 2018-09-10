close all;
clearvars;

% Constants
N_bin = 25; %50;
N_time = 100; %25;
%25 for previous beamformer with 25 STI windows, 40 samples each.
% N_time = 100;
N_beam = 7;
N_pol = 4;

% Number of samples in power_data files
% N_data = N_beam*N_bin*N_time;
N_data = N_pol*N_beam*N_bin*N_time;

% Load data
% power_acc = fopen('power_data.data', 'r');
% power_acc = fopen('power_mcnt_0.out', 'r');
power_acc = fopen('/home/groups/flag/beamformer_0_mcnt_0.out', 'r');
[data, count] = fread(power_acc, N_data, 'float');
fclose(power_acc);

% data2 = zeros(N_data/4,1);
data2 = zeros(N_beam, N_bin, N_time);
data = reshape(data,N_beam, N_pol, N_bin, N_time);

% for l = 1:N_time
%     for m = 1: N_bin
%         for n = 1:N_beam
% %             data2(n,m,l) = data(n,1,m,l); % Test for x polarization
% %             data2(n,m,l) = data(n,2,m,l); % Test for y polarization
% %             data2(n,m,l) = data(n,3,m,l); % Test for real cross polarization
%             data2(n,m,l) = data(n,4,m,l); % Test for imag cross polarization
%         end
%     end
% end

% for pol = 1:N_pol

%     power_accumulation = reshape(squeeze(data(:,pol,:,:)), N_beam, N_bin, N_time);
%     power_accumulation = permute(power_accumulation, [3, 2, 1]);

%     figure();
%     for i = 1:N_bin
%         
%         subplot(5,5,i);
%         imagesc(1:N_beam, 1:N_time, 10*log10(abs(squeeze(power_accumulation(:,i,:)))));
%         set(gca, 'ydir', 'normal');
%         xlabel('Beam Index');
%         ylabel('STI Index');
%         title(['Frequency Bin ', num2str(i)]);
%     end
% end


power_acc_x = reshape(squeeze(data(:,1,:,:)), N_beam, N_bin, N_time);
power_acc_x = permute(power_acc_x, [3, 2, 1]);

power_acc_y = reshape(squeeze(data(:,2,:,:)), N_beam, N_bin, N_time);
power_acc_y = permute(power_acc_y, [3, 2, 1]);

power_acc_xyr = reshape(squeeze(data(:,3,:,:)), N_beam, N_bin, N_time);
power_acc_xyr = permute(power_acc_xyr, [3, 2, 1]);

power_acc_xyi = reshape(squeeze(data(:,4,:,:)), N_beam, N_bin, N_time);
power_acc_xyi = permute(power_acc_xyi, [3, 2, 1]);

% Plot first frequency bin
figure();
imagesc(1:N_beam, 1:N_time, 10*log10(abs(squeeze(power_acc_x(:,1,:)))));
set(gca, 'ydir', 'normal');
xlabel('Beam Index');
ylabel('STI Index');
title('X Frequency Bin 1');

figure();
imagesc(1:N_beam, 1:N_time, 10*log10(abs(squeeze(power_acc_y(:,1,:)))));
set(gca, 'ydir', 'normal');
xlabel('Beam Index');
ylabel('STI Index');
title('Y Frequency Bin 1');

figure();
imagesc(1:N_beam, 1:N_time, 10*log10(abs(squeeze(power_acc_xyr(:,1,:)))));
set(gca, 'ydir', 'normal');
xlabel('Beam Index');
ylabel('STI Index');
title('XY R Frequency Bin 1');

figure();
imagesc(1:N_beam, 1:N_time, 10*log10(abs(squeeze(power_acc_xyi(:,1,:)))));
set(gca, 'ydir', 'normal');
xlabel('Beam Index');
ylabel('STI Index');
title('XY I Frequency Bin 1');

% Plot last frequency bin
figure();
imagesc(1:N_beam, 1:N_time, 10*log10(abs(squeeze(power_acc_x(:,N_bin,:)))));
set(gca, 'ydir', 'normal');
xlabel('Beam Index');
ylabel('Time Sample Index');
title(['X Frequency Bin ', num2str(N_bin)]);

figure();
imagesc(1:N_beam, 1:N_time, 10*log10(abs(squeeze(power_acc_y(:,N_bin,:)))));
set(gca, 'ydir', 'normal');
xlabel('Beam Index');
ylabel('Time Sample Index');
title(['Y Frequency Bin ', num2str(N_bin)]);

figure();
imagesc(1:N_beam, 1:N_time, 10*log10(abs(squeeze(power_acc_xyr(:,N_bin,:)))));
set(gca, 'ydir', 'normal');
xlabel('Beam Index');
ylabel('STI Index');
title(['XY R Frequency Bin ', num2str(N_bin)]);

figure();
imagesc(1:N_beam, 1:N_time, 10*log10(abs(squeeze(power_acc_xyi(:,N_bin,:)))));
set(gca, 'ydir', 'normal');
xlabel('Beam Index');
ylabel('STI Index');
title(['XY I Frequency Bin ', num2str(N_bin)]);



% % Plot first frequency bin
% figure();
% imagesc(1:N_beam, 1:N_time, 10*log10(squeeze(power_accumulation(:,1,:))));
% set(gca, 'ydir', 'normal');
% xlabel('Beam Index');
% ylabel('STI Index');
% title('Frequency Bin 1');
% % colormap('jet');
% 
% % Plot second frequency bin
% figure();
% imagesc(1:N_beam, 1:N_time, 10*log10(squeeze(power_accumulation(:,2,:))));
% set(gca, 'ydir', 'normal');
% xlabel('Beam Index');
% ylabel('STI Index');
% title('Frequency Bin 2');
% % colormap('jet');
% 
% % Plot third frequency bin
% figure();
% imagesc(1:N_beam, 1:N_time, 10*log10(squeeze(power_accumulation(3,:,:))));
% set(gca, 'ydir', 'normal');
% xlabel('Beam Index');
% ylabel('STI Index');
% title('Frequency Bin 3');
% % colormap('jet');
% 
% % Plot last frequency bin
% figure();
% imagesc(1:N_beam, 1:N_time, 10*log10(squeeze(power_accumulation(N_bin,:,:))));
% set(gca, 'ydir', 'normal');
% xlabel('Beam Index');
% ylabel('Time Sample Index');
% title(['Frequency Bin ', num2str(N_bin)]);
% % colormap('jet');