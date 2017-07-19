% Simple script that extracts the covariance matrices for the entire band
% and computes Y-factors

close all;
clearvars;

% System parameters
save_dir = '/lustre/projects/flag/AGBT16B_400_01/BF'; % '/lustre/projects/flag/TMP/BF'; %'/lustre/gbtdata/TGBT16A_508_01/TMP/BF';

% May 19th, 2017 - 01
% Quant gain = 10
% hot_tstamp = '2017_05_20_01:39:52';
% cold_tstamp = '2017_05_20_01:46:47';
% Thot = 290;
% Tcold = 7.5;

% May 20th, 2017 - 01
% Quant gain = 10
% hot_tstamp = '2017_05_21_03:06:05';
% cold_tstamp = '2017_05_21_03:10:22';
% Thot = 290;
% Tcold = 7.5;

% May 20th, 2017 - 02
% Quant gain = 10
% hot_tstamp = '2017_05_21_03:16:27';
% cold_tstamp = '2017_05_21_03:12:45';
% Thot = 290;
% Tcold = 7.5;

% May 20th, 2017 - 03
% Quant gain = 10
% hot_tstamp = '2017_05_21_03:19:47';
% cold_tstamp = '2017_05_21_03:24:53';
% Thot = 290;
% Tcold = 7.5;

% May 20th, 2017 - 04
% Quant gain = 20
% hot_tstamp = '2017_05_21_03:32:31';
% cold_tstamp = '2017_05_21_03:28:16';
% Thot = 290;
% Tcold = 7.5;

% May 21st, 2017 - 01
% Quant gain = 10
% hot_tstamp = '2017_05_22_01:02:42';
% cold_tstamp = '2017_05_22_01:06:18';
% Thot = 290;
% Tcold = 7.5;

% May 21st, 2017 - 02
% Quant gain = 10
% hot_tstamp = '2017_05_22_01:09:49';
% cold_tstamp = '2017_05_22_01:13:27';
% Thot = 290;
% Tcold = 7.5;

% May 21st, 2017 - 03
% Quant gain = 20
% hot_tstamp = '2017_05_22_01:40:27';
% cold_tstamp = '2017_05_22_01:36:57';
% Thot = 290;
% Tcold = 7.5;

% May 21st, 2017 - 03
% Quant gain = 40
% hot_tstamp = '2017_05_22_01:43:10';
% cold_tstamp = '2017_05_22_01:46:57';
% Thot = 290;
% Tcold = 7.5;

% May 22nd, 2017 - 03
% Quant gain = 10
% hot_tstamp = '2017_05_23_02:57:22';
% cold_tstamp = '2017_05_23_03:05:01';
% Thot = 290;
% Tcold = 7.5;

% % May 22nd, 2017 - 04  %%%%%% Not accurate %%%%%%
% % Quant gain = 10
% hot_tstamp = '2017_05_24_20:19:16';
% cold_tstamp = '2017_05_23_03:55:51';
% Thot = 290;
% Tcold = 7.5;

% May 24th, 2017 - 05  %%%%%% GBT test %%%%%%
% Quant gain = 10
hot_tstamp = '2017_05_24_00:46:46';
cold_tstamp = '2017_05_24_00:47:53';
Thot = 290;
Tcold = 7.5;

% % Generate filenames
% Hot_file = sprintf('/home/groups/flag/mat/Hot_cov_%s.mat', hot_tstamp);
% Cold_file = sprintf('/home/groups/flag/mat/Cold_cov_%s.mat', cold_tstamp);

% Get HOT data
% if exist(Hot_file, 'file')
    [R_hot,  dmjd_hot] = aggregate_banks(save_dir, hot_tstamp);
    %save(Hot_file, 'R_hot');
% else
%     load(Hot_file);
% end

% Get COLD data
% if exist(Cold_file, 'file')
    [R_cold, dmjd_cold] = aggregate_banks(save_dir, cold_tstamp);
    %save(Cold_file, 'R_cold');
% else
%     load(Cold_file);
% end

for b = 1:size(R_hot,3)
    [v(:,b),lambda] = eigs(R_hot(:,:,b), R_cold(:,:,b), 1);
    w(:,b) = R_cold(:,:,b)\v(:,b);
    w(:,b) = w/(w(:,b)'*v(:,b));
end

weight_dir = '/home/groups/flag';
weight_num = 1;
w = w(:);
weight_file = sprintf('%s/%d_weights.in', weight_dir, weight_num);
save(weight_file,'w');

% Extract parameters
Nele = size(R_hot, 1);
Nbins = size(R_hot, 3);

% Create data structures for outputs
% P_hot  = zeros(good_elements, Nbins);
% P_cold = zeros(good_elements, Nbins);
P_hot  = zeros(Nele, Nbins);
P_cold = zeros(Nele, Nbins);

% Repeat for every frequency bin
for b = 1:Nbins
%     P_hot(:,b)  = real(diag(R_hot(good_elements,good_elements,b)));
%     P_cold(:,b) = real(diag(R_cold(good_elements,good_elements,b)));
    P_hot(:,b)  = real(diag(R_hot(:,:,b)));
    P_cold(:,b) = real(diag(R_cold(:,:,b)));
end

P_hot(22,:) = NaN;
P_hot(34,:) = NaN;
P_hot(40,:) = NaN;
P_cold(22,:) = NaN;
P_cold(34,:) = NaN;
P_cold(40,:) = NaN;

Y = P_hot(1,:)./P_cold(1,:);
Tsys = (Thot - Tcold*Y)./(Y-1);

R_hot_int = mean(R_hot, 3);
R_cold_int = mean(R_cold,3);

figure(1);
subplot(121);
imagesc(10*log10(squeeze(abs(R_hot_int))));
xlabel('No. of elements');
ylabel('No. of elements');
axis square
subplot(122);
imagesc(10*log10(squeeze(abs(R_cold_int))));
xlabel('No. of elements');
ylabel('No. of elements');
axis square

figure(2);
f1 = 5;
imagesc(10*log10(squeeze(abs(R_hot(:,:,f1)))));
% title(sprintf('On covariance %d', num2str(f1)));
xlabel('No. of elements');
ylabel('No. of elements');

figure(3);
imagesc(10*log10(squeeze(abs(R_cold(:,:,f1)))));
% title(sprintf('Off covariance %d', num2str(f1)));
xlabel('No. of elements');
ylabel('No. of elements');

figure(4);
f2 = 6;
imagesc(10*log10(squeeze(abs(R_hot(:,:,f2)))));
% title(sprintf('On covariance %d', num2str(f2)));
xlabel('No. of elements');
ylabel('No. of elements');

figure(5);
imagesc(10*log10(squeeze(abs(R_cold(:,:,f2)))));
% title(sprintf('Off covariance %d', num2str(f2)));
xlabel('No. of elements');
ylabel('No. of elements');

freqs = (-249:250)*303.24e-3;
figure(6);
plot(freqs,10*log10(P_hot).');
xlabel('Relative Frequency to LO (MHz)');
ylabel('Power (dB)');
grid on;

% % Look for obviously bad frequencies
% Tsys_threshold = 200;
% good_freqs = [];
% for b = 1:Nbins
%     val = mean(abs(Tsys(:,b)));
%     if val <= Tsys_threshold
%         good_freqs = [good_freqs, b];
%     end
% end
% freqs = 1:Nbins;
% bad_freqs = freqs;
% bad_freqs(good_freqs) = [];
% 
% % Look for obviously bad elements
% Tsys_threshold = 200;
% good_ele = [];
% for e = 1:Ngood_ele
%     val = mean(Tsys(e,good_freqs));
%     if val <= Tsys_threshold && val >= 0
%         good_ele = [good_ele, e];
%     end
% end
%     
% Tsys(:,bad_freqs) = NaN;

Tsys_thresh = 100;
Tsys(Tsys>Tsys_thresh) = NaN;
Tsys(Tsys<=0) = NaN;

%% Plot spectrum per element
freqs = (-249:250)*303.24e-3;
figure(7);
plot(freqs, Tsys.');
title(sprintf('OTF Tsys, %s', hot_tstamp), 'Interpreter', 'none');
xlabel('Relative Frequency to LO (MHz)');
ylabel('T_s_y_s (K)');
grid on;

% print(gcf, '-dpng', sprintf('fig/OTF_Tsys_%s.png', hot_tstamp));
