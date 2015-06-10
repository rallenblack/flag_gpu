close all;
clearvars;

Nant = 4;
Nfft = 1024;
NSTRIDE = 4*Nant; % 2 floats per complex input, 2 pols per antenna

FILE = fopen('data_4X');
X = fread(FILE, Inf, 'int8');
% FILE = fopen('fft_in.dat');
% X = fread(FILE, Inf, 'float');
fclose(FILE);

for i = 1:Nant
    figure(1);
    subplot(Nant, 1, i);
    Xr = X(4*(i-1)+1:NSTRIDE:end);
    Xi = X(4*(i-1)+2:NSTRIDE:end);
    Yr = X(4*(i-1)+3:NSTRIDE:end);
    Yi = X(4*(i-1)+4:NSTRIDE:end);
    Xr = Xr(1:4000); Xi = Xi(1:4000);
    Yr = Yr(1:4000); Yi = Yi(1:4000);
    plot(1:length(Xr), Xr, '-b',...
         1:length(Xi), Xi, '--b',...
         1:length(Yr), Yr, '-.r',...
         1:length(Yi), Yi, ':r');
    title(['Antenna ', num2str(i)]);
    legend('Re(X)', 'Im(X)', 'Re(Y)', 'Im(Y)');
    
    xc = Xr(1:Nfft) + 1j*Xi(1:Nfft);
    yc = Yr(1:Nfft) + 1j*Yi(1:Nfft);
    Xc = fft(xc, Nfft);
    Yc = fft(yc, Nfft);
    
    Xc(abs(Xc).^2 < 1e-10) = 0;
    Yc(abs(Yc).^2 < 1e-10) = 0;
    figure(2);
    subplot(Nant, 1, i);
    plot(1:Nfft, 20*log10(abs(Xc)), '-b',...
         1:Nfft, 20*log10(abs(Yc)), '--r');
    title(['FFT of Antenna ', num2str(i)]);
    legend('X(f)', 'Y(f)');
end