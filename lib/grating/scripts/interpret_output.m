% Interpret grating output
close all;
clearvars;

Nant = 200;
Nfloats = 4;
Nfft = 32;
spacing = Nant*Nfloats;

FILE = fopen('spec.dat');
Xfft = fread(FILE, Inf, 'float');
fclose(FILE);

FILE = fopen('spec2.dat');
Xpfb = fread(FILE, Inf, 'float');
fclose(FILE);

for na = 1:Nant
%     figure(1);
%     subplot(4,1,1);
%     plot(1:Nfft, 20*log10(abs(Xfft(4*(na-1)+1:spacing:spacing*Nfft))), '-b', ...
%          1:Nfft, 20*log10(abs(Xpfb(4*(na-1)+1:spacing:spacing*Nfft))), '--r');
%     title('Grating Accumulated Spectra - X');
%     xlabel('Frequency Bin Index');
%     ylabel('Power (dB, arb. units)');
%     legend('FFT', 'PFB');
% 
%     subplot(4,1,2);
%     plot(1:Nfft, 20*log10(abs(Xfft(4*(na-1)+2:spacing:spacing*Nfft))), '-b', ...
%          1:Nfft, 20*log10(abs(Xpfb(4*(na-1)+2:spacing:spacing*Nfft))), '--r');
%     title('Grating Accumulated Spectra - Y');
%     xlabel('Frequency Bin Index');
%     ylabel('Power (dB, arb. units)');
%     legend('FFT', 'PFB');
% 
%     subplot(4,1,3);
%     plot(1:Nfft, 20*log10(abs(Xfft(4*(na-1)+3:spacing:spacing*Nfft))), '-b', ...
%          1:Nfft, 20*log10(abs(Xpfb(4*(na-1)+3:spacing:spacing*Nfft))), '--r');
%     title('Grating Accumulated Spectra - Real(XY)');
%     xlabel('Frequency Bin Index');
%     ylabel('Power (dB, arb. units)');
%     legend('FFT', 'PFB');
% 
%     subplot(4,1,4);
%     plot(1:Nfft, 20*log10(abs(Xfft(4*(na-1)+4:spacing:spacing*Nfft))), '-b', ...
%          1:Nfft, 20*log10(abs(Xpfb(4*(na-1)+4:spacing:spacing*Nfft))), '--r');
%     title('Grating Accumulated Spectra - Imag(XY)');
%     xlabel('Frequency Bin Index');
%     ylabel('Power (dB, arb. units)');
%     legend('FFT', 'PFB');
    
    figure(na);
    % figure(ceil(na/20));
    % subplot(Nant/20,1,mod(na-1, Nant/20)+1);
    plot(1:Nfft, 20*log10(abs(Xfft(4*(na-1)+1:spacing:spacing*Nfft))), '-b',...
         1:Nfft, 20*log10(abs(Xpfb(4*(na-1)+1:spacing:spacing*Nfft))), '--r');
    title(['Subband ', num2str(na)]);
    xlabel('Frequency Bin Index');
    ylabel('Power (dB, arb. units)');
    legend('FFT', 'PFB');
end

