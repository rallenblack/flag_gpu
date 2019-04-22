% This is the main script used to process the output of the gPFB library.
% The gPFB library output is read in and then appropriately parsed to
% seperate the individual bins of the PFB.

% In all the example data I have here, along with what this script was
% mostly created for, was to show the filter response from samplining it by
% sweeping a complex exponential tone across frequency. See notes made on
% this procedure in my research journal.

clearvars;
% %% Load info about data
% f_info = fopen('../../code/pfb/inputdata/genInfo_s_256_c_1_e_1_d_0125.dat');
% 
% info = fread(f_info, 4, 'int');
% info_freq = fread(f_info, 2, 'double');
% bin_axis = fread(f_info, 'double');
% 
% INFO.elements = info(1);
% INFO.coarseChannels = info(2);
% INFO.timeSamples = info(3);
% INFO.n_complexSamples = info(4);
% 
% INFO.sampleRate = info_freq(1);
% INFO.n_freqSamples = info_freq(2);
% fclose(f_info);

%% Setup global parameters
Nsamps = 4000; % INFO.n_complexSamples;
coarseCh = 5; % INFO.coarseChannels;
numEl = 64; % INFO.elements;

frequency_sample_decimation = 8;
fs = 303; %INFO.sampleRate;
nfft = 32;
ntaps = 8;
subbands = coarseCh*numEl; % 5 channels processed * 64 el = 320 subbands

%% Load data
f = fopen('outfile.dat', 'r');
output = fread(f, 'float32');  
fclose(f);

%% Process and order output
%windows = floor((N - nfft*ntaps)/nfft);
windows = Nsamps/nfft;

size_slice = coarseCh*(numEl*2)*nfft;

rows = length(output)/size_slice; % This is not the same size as windows and need to figure out why.

CH = zeros(numEl*coarseCh, nfft);
XX = zeros(windows, nfft);
% XX = zeros(windows, numEl, nfft);
for i = 1:windows
    slice = output((i-1)*size_slice+1:i*size_slice);
    tmp = reshape(slice, [coarseCh*(2*numEl) nfft]);
    
    
    for k = 1:coarseCh
        ch_slice = tmp((k-1)*numEl*2+1:k*numEl*2,:);
        el_re = ch_slice(1:2:end,:);
        el_im = ch_slice(2:2:end,:);
        el_spectra = (el_re + 1j*el_im)/nfft; % fftshift(el_re + 1j*el_im)/nfft; % normalize and shift since the gpu doesnt do that
        XX(i,:) = el_spectra(3,:); % save the spectra to plot filter response.
        
        CH((k-1)*numEl+1:k*numEl,:) = CH((k-1)*numEl+1:k*numEl,:) + abs(el_spectra).^2;
        
    end
end

% time average CH
CH = CH/windows;

%% Plot element and channel
el_idx = 1; % [1 64]
ch_idx = 1; % [1 5]

faxis = 0:fs/nfft:fs-1/nfft;
el_data = CH((ch_idx-1)*numEl+1:ch_idx*numEl,:);
el = el_data(el_idx, :);

figure(1); clf;
subplot(321);
% plot(faxis, 10*log10(el+0.1)); grid on;
% plot(faxis, (el+0.1)); grid on;
plot(faxis, (abs([flip(el(1:length(el)/2)) el((length(el)/2)+1:end)]))+0.1); grid on;
xlim([min(faxis), max(faxis)]);
% ylim([-5, 80]);
xlabel('Frequency (MHz)');
ylabel('Magnitude (arb. units)');
title('Coarse Channel 1 - Processed output');
set(gca, 'xtick', [0:14]*20 + 5);

for i = 2:coarseCh
    ch_idx = i;
    el_data = CH((ch_idx-1)*numEl+1:ch_idx*numEl,:);
    el = el_data(el_idx, :);
    subplot(3,2,i);
%     plot(faxis, 10*log10(abs(el))+0.1); grid on;
%     plot(faxis, (abs(el))+0.1); grid on;
    plot(faxis, (abs([flip(el(1:length(el)/2)) el((length(el)/2)+1:end)]))+0.1); grid on;
    xlim([min(faxis), max(faxis)]);
    xlabel('Frequency (kHz)');
    ylabel('Magnitude (arb. units)');
%     ylim([-5, 80]);
    title(['Coarse channel ' num2str(i)]);
    set(gca, 'xtick', [0:14]*20 + 5);
end

XX_freq_samples = XX(1:frequency_sample_decimation:end,:);
figure();
% plot(20*log10(abs(XX(:,20))));
% plot(20*log10(abs([XX(:,1); XX(:,2); XX(:,3); XX(:,4); XX(:,5); XX(:,6); XX(:,7); XX(:,8); XX(:,9); XX(:,10); ...
%     XX(:,11); XX(:,12); XX(:,13); XX(:,14); XX(:,15); XX(:,16); XX(:,17); XX(:,18); XX(:,19); XX(:,20); ...
%     XX(:,21); XX(:,22); XX(:,23); XX(:,24); XX(:,25); XX(:,26); XX(:,27); XX(:,28); XX(:,29); XX(:,30); XX(:,31); XX(:,32)])));
% plot(20*log10(abs([XX_freq_samples(:,1); XX_freq_samples(:,2); XX_freq_samples(:,3); XX_freq_samples(:,4); ...
%     XX_freq_samples(:,5); XX_freq_samples(:,6); XX_freq_samples(:,7); XX_freq_samples(:,8); XX_freq_samples(:,9); ...
%     XX_freq_samples(:,10); XX_freq_samples(:,11); XX_freq_samples(:,12); XX_freq_samples(:,13); XX_freq_samples(:,14); ...
%     XX_freq_samples(:,15); XX_freq_samples(:,16); XX_freq_samples(:,17); XX_freq_samples(:,18); XX_freq_samples(:,19); ... 
%     XX_freq_samples(:,20); XX_freq_samples(:,21); XX_freq_samples(:,22); XX_freq_samples(:,23); XX_freq_samples(:,24); ...
%     XX_freq_samples(:,25); XX_freq_samples(:,26); XX_freq_samples(:,27); XX_freq_samples(:,28); XX_freq_samples(:,29); ...
%     XX_freq_samples(:,30); XX_freq_samples(:,31); XX_freq_samples(:,32)])));
XX_sti = abs([XX_freq_samples(:,1); XX_freq_samples(:,2); XX_freq_samples(:,3); XX_freq_samples(:,4); ...
    XX_freq_samples(:,5);  XX_freq_samples(:,6);  XX_freq_samples(:,7);  XX_freq_samples(:,8);  XX_freq_samples(:,9); ...
    XX_freq_samples(:,10); XX_freq_samples(:,11); XX_freq_samples(:,12); XX_freq_samples(:,13); XX_freq_samples(:,14); ...
    XX_freq_samples(:,15); XX_freq_samples(:,16); XX_freq_samples(:,17); XX_freq_samples(:,18); XX_freq_samples(:,19); ... 
    XX_freq_samples(:,20); XX_freq_samples(:,21); XX_freq_samples(:,22); XX_freq_samples(:,23); XX_freq_samples(:,24); ...
    XX_freq_samples(:,25); XX_freq_samples(:,26); XX_freq_samples(:,27); XX_freq_samples(:,28); XX_freq_samples(:,29); ...
    XX_freq_samples(:,30); XX_freq_samples(:,31); XX_freq_samples(:,32)]);

plot(20*log10([flip(XX_sti(1:size(XX_sti,1)/2,1).') XX_sti((size(XX_sti,1)/2)+1:end,1).']));
grid on;
title('Measured Filter Response');
xlabel('STI windows over 32 FFT points');
ylabel('Power (dB)');

figure();
% plot(abs([XX_freq_samples(:,1); XX_freq_samples(:,2); XX_freq_samples(:,3); XX_freq_samples(:,4); ...
%     XX_freq_samples(:,5); XX_freq_samples(:,6); XX_freq_samples(:,7); XX_freq_samples(:,8); XX_freq_samples(:,9); ...
%     XX_freq_samples(:,10); XX_freq_samples(:,11); XX_freq_samples(:,12); XX_freq_samples(:,13); XX_freq_samples(:,14); ...
%     XX_freq_samples(:,15); XX_freq_samples(:,16); XX_freq_samples(:,17); XX_freq_samples(:,18); XX_freq_samples(:,19); ... 
%     XX_freq_samples(:,20); XX_freq_samples(:,21); XX_freq_samples(:,22); XX_freq_samples(:,23); XX_freq_samples(:,24); ...
%     XX_freq_samples(:,25); XX_freq_samples(:,26); XX_freq_samples(:,27); XX_freq_samples(:,28); XX_freq_samples(:,29); ...
%     XX_freq_samples(:,30); XX_freq_samples(:,31); XX_freq_samples(:,32)]));

plot([flip(XX_sti(1:size(XX_sti,1)/2,1).') XX_sti((size(XX_sti,1)/2)+1:end,1).']);
grid on;
title('Measured Filter Response');
xlabel('STI windows over 32 FFT points');
ylabel('Power (arb. units)');

% Figuring out what the output looks like based on the following input
% channelBandgap = 8;
% for i = 1:20
%     freq(i) = channelBandgap*i + 5;
% end
% for n = 1:4000
%     for ff = 1:20
%         for ee = 1:40
%             dataRe = 127*cos(0.1*2*pi*freq(ff)*n/fs);
%             dataIm = 127*sin(0.1*2*pi*freq(ff)*n/fs);
%             dataComp(ee,ff,n) = dataRe + 1j*dataIm;
%         end
%     end
% end
% % Plot magnitude and and phase of the input
% figure(); plot(abs(squeeze(dataComp(1,1,:)))); % Magnitude
% figure(); plot(angle(squeeze(dataComp(1,1,:)))); % Phase 
% % Plot output of the PFB by carrying out an FFT on this data (dataComp)
% figure(); plot(abs(squeeze(fftshift(fft(dataComp(1,:,1))))));
% figure(); plot(abs(squeeze(fftshift(fft(dataComp(1,1,:))))));



% XX_freq_samples = XX(1:frequency_sample_decimation:end,:);
% totalresponse = zeros(length(bin_axis),1);
%% Plot the filter response.
% SINGLE_RESPONSE = 1;
% figure(99); clf; hold on; grid on;
% ax = gca;
% if SINGLE_RESPONSE == 1
%     bin_response_idx = 1;
% %     bin_axis = -nfft/2+nfft/size(XX_freq_samples,1):nfft/size(XX_freq_samples,1):nfft/2;
% %     h1 = plot(bin_axis, 20*log10(abs(XX_freq_samples(:,bin_response_idx)) + 0.1));
%     h1 = plot(20*log10(abs(XX(:,1))));
%     legendHandle = h1;
%     legendText = {'Single Bin Response'};
%     axisTick = -nfft/2:2:nfft/2;
% else
%     numberOfResponses = nfft;
%     for i = 1:nfft/numberOfResponses:nfft
%         ax.ColorOrderIndex = 1;
% %         totalresponse = totalresponse + XX_freq_samples(:,i); 
%         totalresponse = totalresponse + XX(:,i); 
% %         h1 = plot(bin_axis, 20*log10(abs(XX_freq_samples(:,i)) + 0.1));
%         h1 = plot(20*log10(abs(XX(:,i))));
%     end
%     %totalresponse = sum(XX_freq_samples,2);
% %     h2 = plot(bin_axis, 20*log10(abs(totalresponse) + 0.1));
%     h2 = plot(20*log10(abs(totalresponse) + 0.1));
%     legendHandle = [h1, h2];
%     legendText = {'Single Bin Response', 'Full Filter Bank Response'};
%     axisTick = 0:2:nfft;
% end
% 
% % configure plot
% % xlim([min(bin_axis) max(bin_axis)]);
% ylim([-20, 50]);
% title('Measured Filter Response');
% xlabel('Frequency Bin Index');
% ylabel('Power (dB)');
% legend(legendHandle, legendText);
% set(ax, 'XTick', axisTick);
% set(ax, 'FontSize', 14);
% box on;
 