% System Constants
fs        = 155e6; % Sampling frequency - used for noise level
Ninputs   = 40;    % Number of inputs/antennas
Nbins     = 500;   % Total number of frequency bins
Nfft      = 512;   % F-engine FFT size
Nfengines = 5;     % Number of F-engines
Nxengines = 20;    % Number of X-engines (i.e. Number of GPUs)

Nin_per_f        = Ninputs/Nfengines; % Number of inputs per F-engine
Nbin_per_x       = Nbins/Nxengines; % Number of bins per X-engine
Ntime_per_packet = 20; % Number of time samples (spectra snapshots) per packet

quant_res = 0.5e-8; % Arbitrary quantization resolution for noise generation

% Correlated data parameters (only used if data_flag = 5)
% kw = Karl Warnick
kw_bin = 1;

kw_xid = floor((kw_bin - 1)/Nbin_per_x) + 1;
kw_bin_r = mod(kw_bin - 1, Nbin_per_x) + 1;

% Generate model correlation matrix
Nel = Ninputs;
kd = 0.1*pi*(1:Nel);
kwR = zeros(Nel, Nel);
for m = 1:Nel
    for n = 1:Nel
        kwR(m,n) = sinc((kd(m) - kd(n))/pi);
    end
end

% create time samples
rng(1);
kwNs = 8; % 4000
kw_z = (randn(Nel, kwNs) + 1j*randn(Nel,kwNs))/sqrt(2);
kwM = sqrtm(kwR);
kw_x = kwM*kw_z;
% kw_x = repmat(kw_x(:,1), 1, kwNs);
% kw_x(:,1:end-1) = 0;
d_max = 4;
d_min = -4;
kw_x_real = int8(((real(kw_x) - d_min)/(d_max - d_min) - 0.5) * 256);
kw_x_imag = int8(((imag(kw_x) - d_min)/(d_max - d_min) - 0.5) * 256);

% Estimated correlation matrix
kw_x_quant = single(kw_x_real) + 1j*single(kw_x_imag);
kw_Rhat = (kw_x_quant*kw_x_quant')/kwNs;

data = zeros(Nfft,Ninputs,Ninputs);
data(1,:,:) = kw_Rhat;

real_data = real(data);
imag_data = imag(data);

fidr = fopen('Real_bram_data.bin','w');
fwrite(fidr, real_data, 'int8');

fid_i = fopen('Imag_bram_data.bin','w');
fwrite(fid_i, imag_data, 'int8');

