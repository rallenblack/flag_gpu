% Interpret FFT output
clearvars;
close all;

Nant = 4;
Nfft = 128;

FILE = fopen('myfile_nfft128.dat');
Xfft = fread(FILE, Inf, 'float');
fclose(FILE);

Xfft_real = Xfft(1:4:end);
Xfft_imag = Xfft(2:4:end);
Yfft_real = Xfft(3:4:end);
Yfft_imag = Xfft(4:4:end);
Xfft_pow  = Xfft_real.^2 + Xfft_imag.^2;
Yfft_pow  = Yfft_real.^2 + Yfft_imag.^2;

Xfft_pow(Xfft_pow < 1e-10) = 0;
Yfft_pow(Yfft_pow < 1e-10) = 0;

figure();
for i = 1:Nant
    subplot(Nant, 1, i);
    plot(1:Nfft, 10*log10(Xfft_pow(i:Nant:i+Nant*Nfft-1)), '-b',...
         1:Nfft, 10*log10(Yfft_pow(i:Nant:i+Nant*Nfft-1)), '--r');
    title(['Antenna ', num2str(i)]);
end
