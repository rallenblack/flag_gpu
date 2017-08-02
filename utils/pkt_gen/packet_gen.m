% FLAG packet generator
clearvars;
close all;

% Clear any sockets opened by MATLAB
u = instrfindall;
delete(u);

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

% Noise parameters - only used if data_flag = 1
kb     = 1.3806488e-23; % Boltzmann's constant
Tsys   = 45;            % System temperature
BW     = fs/Nfft;       % Channel bandwidth
sigma2 = kb*Tsys*BW;    % Noise power per channel


% Flag to control what is sent
% data_flag
% 1 -> White noise
% 2 -> Unit-Energy Sinusoid in Single Element
% 3 -> Send all ones
% 4 -> Send chirp
% 5 -> Send spatially correlated data in a single bin
% 6 -> Send complex sinusodial data
% 7 -> Send ULA data
% 8 -> Send exponentially correlated noise.
% 9 -> Send pulsar data
% else -> Send all zeros
data_flag = 9;

% Sinusoid parameters (only used if data_flag = 2)
% It should be noted that the phase of the sinusoid will not change between
% time samples-- this is just for convenience. A more sophisticated packet
% generator would incorporate the phase shifts across time.
s_bin   = 401; % Sinusoid's absolute bin number (1-500)
s_ele   = 1; % Sinusoid's absolute element number (1-40)
s_phi   = 0; % Sinusoid's phase (magnitude is set to 1)
s_xid   = floor((s_bin - 1)/Nbin_per_x) + 1; % X-engine ID for desired bin
s_fid   = floor((s_ele - 1)/Nin_per_f) + 1;   % F-engine ID for desired input
s_bin_r = mod(s_bin - 1, Nbin_per_x) + 1; % Relative bin number (internal fengine index)
s_ele_r = mod(s_ele - 1, Nin_per_f) + 1; % Relative element number

% Chirp parameters (only used if data_flag = 4)
c_bin_start = 300;  % Absolute bin index in which the chirp will begin (1-500)
c_bin_end   = 350;  % Absolute bin index in which the chirp will end (1-500)
c_ele       = 1;    % Absolute element index (1-40)
c_phi       = 0;    % Phase of chirp (could be more sophisticated...)
c_ntime     = 500;  % Number of coarse channel time samples in one chirp
c_fid       = floor((c_ele - 1)/Nin_per_f) + 1; % F-engine ID for desired input
c_ele_r     = mod(c_ele - 1, Nin_per_f) + 1; % Relative element number
c_num_bins  = c_bin_end - c_bin_start + 1;
c_time_per_bin = c_ntime/c_num_bins;

% Correlated data parameters (only used if data_flag = 5)
% kw = Karl Warnick
kw_bin = 6;
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
data_windows = 5;
N = 4000*data_windows;
kwNs = 4000;
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

save('matlab_corr.mat', 'kw_Rhat');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Case 6 - Complex Sinusoid
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cs_Ns = N;
cs_freq = 0.5;
cs_n = 0:cs_Ns-1;
sigma = 2;
cs_re = 127 * (0.1 * cos(2*pi*cs_freq*cs_n)) + sigma^2*randn();
cs_im = 127 * (0.1 * sin(2*pi*cs_freq*cs_n)) + sigma^2*randn();


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Case 7 - ULA
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ULA_sigma2 = 8;
ULA_theta = 90; % Degrees
ULA_freqs = (0:499)*(303e3) + 1300e6; % All frequencies
ULA_c = 3e8; % speed of propagation (m/s)
ULA_d = ULA_c/ULA_freqs(end); % Element spacing
ULA_phi = ULA_d*cos(ULA_theta*pi/180)/ULA_c*2*pi*ULA_freqs; % Phase shift
ULA_N = 4000; % Number of samples
ULA_complex = zeros(ULA_N, 40, length(ULA_freqs)); % time x freq x elements
for ULA_b = 1:length(ULA_freqs) % Iterate over frequency
        ULA_a = exp(1j*(1:40)*ULA_phi(ULA_b)); % Steering vector
        %ULA_complex(:, :, ULA_b) = ...
        %    kron(ULA_a, sqrt(ULA_sigma2/2)*(randn(ULA_N, 1) + 1j*randn(ULA_N, 1)));
        ULA_complex(:,:,ULA_b) = kron(ULA_a, sqrt(ULA_sigma2)*ones(ULA_N, 1));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Case 8 - Correlated Exponential Noise (CEN)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
CEN_rho = 0.9;
CEN_stdG = 0.05;

CEN_A = toeplitz(CEN_rho.^(0:Ninputs-1));
CEN_G = diag(max(.2, 1+CEN_stdG*randn(Ninputs,1)));
CEN_Asqr = sqrtm(CEN_G*CEN_A*CEN_G');

CEN_N = 4000;
CEN = CEN_Asqr/sqrt(2)*(randn(Ninputs, CEN_N) + 1j*randn(Ninputs, CEN_N));

CEN_R = 1/CEN_N*(CEN*CEN');
figure(99);
imagesc(abs(CEN_R));

c_max = 4;
c_min = -4;
CEN_real = int8(((real(CEN) - c_min)/(c_max - c_min) - 0.5) * 256);
CEN_imag = int8(((imag(CEN) - c_min)/(c_max - c_min) - 0.5) * 256);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Case 9 - Simulated pulsar data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Increase the range of tau when dispersion measure causes m_D to exceed
% time samples.
D = 10; % Dispersion measure 
freq = (0:499)*(303e3) + 1300e6; % All frequencies
fo = freq(floor(length(freq)/2)); % Center frequency
m_D = 4.1488e3*((fo^-2)-(freq.^-2))*D; % Frequency dependent timing offset
Ntime = 4000;
tau = -2.8e-14:((2.5e-14)+(2.8e-14))/(Ntime-1):2.5e-14; % Range of timing offsets
pulseData = zeros(Ninputs, Nbins, Ntime);
% Noisy environment
for ii = 1:size(pulseData,3)
    for jj = 1:size(pulseData,2)
        pulseData(:,jj,ii) = 0.1*(randn(1) + 1j*rand(1));
    end
end

% Pulsar
pulse = 1;
for cyc = [0] % [0, -1000, 1000]
    for m = 1:Ninputs
        for k = 1:Nbins
            tmp = abs(m_D(k)-tau);
            [~,idx] = min(tmp);
            phi = m*2*pi*freq(k)*tau(idx+cyc);
            pulseData(m,k,idx+cyc) = pulse*exp(1j*phi) + 0.1*(randn(1) + 1j*rand(1));
        end
    end
end

keyboard;

% Create UDP sockets - 1 IP address per Xengine (xid)
for xid = 1:Nxengines

    remoteHost = ['10.10.1.', num2str(xid)];

    if xid == 1
        remoteHost = '10.17.16.200'; % It was 208 before
    end
    if xid == 2
        remoteHost = '10.17.16.201'; % It was 208 before
    end
    if xid == 3
        remoteHost = '10.17.16.202'; % It was 208 before
    end
    if xid == 4
        remoteHost = '10.17.16.203'; % It was 208 before
    end
    if xid == 12
        remoteHost = '10.17.16.211'; % It was 208 before
    end
    %if xid == 14
    %    remoteHost = '10.10.1.1';
    %end
    sock(xid) = udp(remoteHost, 'RemotePort', 60000, 'LocalPort', 60001);
    set(sock(xid), 'OutputBufferSize', 9000);
    set(sock(xid), 'OutputDatagramPacketSize', 9000);
end


% Generate packet payloads
mcnt = 0; % Each mcnt represents 20 packets across all F-engines in the
          % same time frame
  
for mcnt = [0:401] %while mcnt <= 10000
    disp(['Sending mcnt = ', num2str(mcnt)]);
    for xid = [1:4] % Set to a single X-engine for single HPC testing (Richard B.)
        for fid = 1:Nfengines
            w_idx = 1;
            
            % Create packet header
            % header contains the following
            % MSB = 1
            % bits  1-44: mcnt
            % bits 45-49: cal
            % bits 49-56: F-engine ID (fid)
            % bits 57-64: X-engine ID
            % LSB = 64
            header = uint64(mcnt)*2^20 + uint64(15)*2^16 + uint64(fid-1)*2^8 + uint64(xid-1);
            
            % Allocate memory for packet payload
            payload = zeros(16*Ntime_per_packet*Nbin_per_x+8, 1, 'uint8');
            
            % Shift header information into packet
            % (The socket wants bytes, so we need to mask and shift to get
            % uint8)
            payload(w_idx) = uint8(bitand(bitshift(header, -56), (2^8 - 1)));
            w_idx = w_idx + 1;
            payload(w_idx) = uint8(bitand(bitshift(header, -48), (2^8 - 1)));
            w_idx = w_idx + 1;
            payload(w_idx) = uint8(bitand(bitshift(header, -40), (2^8 - 1)));
            w_idx = w_idx + 1;
            payload(w_idx) = uint8(bitand(bitshift(header, -32), (2^8 - 1)));
            w_idx = w_idx + 1;
            payload(w_idx) = uint8(bitand(bitshift(header, -24), (2^8 - 1)));
            w_idx = w_idx + 1;
            payload(w_idx) = uint8(bitand(bitshift(header, -16), (2^8 - 1)));
            w_idx = w_idx + 1;
            payload(w_idx) = uint8(bitand(bitshift(header , -8), (2^8 - 1)));
            w_idx = w_idx + 1;
            payload(w_idx) = uint8(bitand(header,                (2^8 - 1)));
            w_idx = w_idx + 1;
            
            % Generate signal to send
            data = zeros(Nin_per_f, 2, Nbin_per_x, Ntime_per_packet); % 8x2x25x20
            switch data_flag
                case 1 % Send white noise
                    sig = sqrt(sigma2/2);
                    data = sig*randn(Nin_per_f, 2, Nbin_per_x, Ntime_per_packet);
                    data = round(data./quant_res);
                    data(data < 0) = 2^8 + data(data < 0);
                case 2 % Send single unit-energy sinusoid in single element
                    if fid == s_fid && xid == s_xid
                        s_real = real(exp(1j*s_phi));
                        s_imag = imag(exp(1j*s_phi));
                        data(s_ele_r, 1, s_bin_r, :) = s_real;
                        data(s_ele_r, 2, s_bin_r, :) = s_imag;
                    end
                case 3 % Send all ones
                    data = ones(Nin_per_f, 2, Nbin_per_x, Ntime_per_packet);
                case 4 % Send chirp
                    time_start = mcnt*Ntime_per_packet;
                    if fid == c_fid
                        for t = 0:Ntime_per_packet-1
                            abs_time = time_start + t;
                            c_bin_abs = round(abs_time/c_time_per_bin) + c_bin_start - 1;
                            c_bin = mod(c_bin_abs, c_num_bins) + c_bin_start - 1;
                            
                            c_xid = floor((c_bin - 1)/100) + 1;
                            if xid == c_xid
                                c_bin_r = mod(c_bin - 1, 100) + 1;
                                c_real = real(exp(1j*c_phi));
                                c_imag = imag(exp(1j*c_phi));
                                data(c_ele_r, 1, c_bin_r, t+1) = c_real;
                                data(c_ele_r, 2, c_bin_r, t+1) = c_imag;
                            end
                        end
                    end
                case 5 % Send correlated data in single bin
                    if xid == kw_xid
                        f_idxs = (fid - 1)*8+1:fid*8;
                        %f_idxs = f_idxs(end:-1:1);
                        t_idxs = mod(mcnt*20 + 1:(mcnt+1)*20, kwNs);
                        % disp(t_idxs);
                        t_idxs(t_idxs == 0) = kwNs;
                        data(:, 1, kw_bin_r, :) = kw_x_real(f_idxs,t_idxs);
                        data(:, 2, kw_bin_r, :) = kw_x_imag(f_idxs,t_idxs);
                        data(data < 0) = 2^8 + data(data < 0);
                    end
                case 6 % Send complex sinusoidal data
                    t_idxs = mod(mcnt*20 + 1:(mcnt+1)*20, cs_Ns);
                    t_idxs(t_idxs == 0) = kwNs;
                    for cs_bin = 1:Nbin_per_x
                        for cs_ele = 1:Nin_per_f
                            data(cs_ele,1,cs_bin,:) = cs_re(t_idxs);
                            data(cs_ele,2,cs_bin,:) = cs_im(t_idxs);
                        end
                    end
                    data(data < 0) = 2^8 + data(data < 0);
                case 7 % Send ULA complex data
                    t_idxs = mod(mcnt*20 + 1:(mcnt+1)*20, ULA_N);
                    t_idxs(t_idxs == 0) = ULA_N;
                    f_idxs = (fid - 1)*8+1:fid*8;
                    freq_idxs = 5*(xid-1) + [1:5, 101:105, 201:205, 301:305, 401:405];
                    tmp = ULA_complex(t_idxs, f_idxs, freq_idxs);
                    tmp2 = permute(tmp, [2, 3, 1]);
                    data(:,1,:,:) = real(tmp2);
                    data(:,2,:,:) = imag(tmp2);
                    data(data < 0) = 2^8 + data(data < 0);
                case 8
                    f_idxs = (fid - 1)*8+1:fid*8;
                    t_idxs = mod(mcnt*20 + 1:(mcnt+1)*20, CEN_N);
                    t_idxs(t_idxs == 0) = CEN_N;
                    for bin_idx = 1:Nbin_per_x
                        data(:, 1, bin_idx, :) = CEN_real(f_idxs,t_idxs);
                        data(:, 2, bin_idx, :) = CEN_imag(f_idxs,t_idxs);
                        data(data < 0) = 2^8 + data(data < 0);
                    end
                case 9 % Send pulsar data
                    t_idxs = mod(mcnt*20 + 1:(mcnt+1)*20, Ntime);
                    t_idxs(t_idxs == 0) = Ntime;
                    f_idxs = (fid - 1)*8+1:fid*8;
                    freq_idxs = 5*(xid-1) + [1:5, 101:105, 201:205, 301:305, 401:405];
                    tmp = pulseData(f_idxs, freq_idxs, t_idxs);
                    data(:,1,:,:) = real(tmp);
                    data(:,2,:,:) = imag(tmp);
                    data(data < 0) = 2^8 + data(data < 0);
                otherwise % Send all zeros
            end
            data = uint8(data);
            
            for nt = 1:Ntime_per_packet
                for nb = 1:Nbin_per_x
                    chan = squeeze(data(:,:,nb,nt));
                    % chan = reshape(chan(:), 2, 8).';
                    
                    payload(w_idx) = chan(1,1); w_idx = w_idx + 1;
                    payload(w_idx) = chan(1,2); w_idx = w_idx + 1;
                    payload(w_idx) = chan(2,1); w_idx = w_idx + 1;
                    payload(w_idx) = chan(2,2); w_idx = w_idx + 1;
                    payload(w_idx) = chan(3,1); w_idx = w_idx + 1;
                    payload(w_idx) = chan(3,2); w_idx = w_idx + 1;
                    payload(w_idx) = chan(4,1); w_idx = w_idx + 1;
                    payload(w_idx) = chan(4,2); w_idx = w_idx + 1;
                    payload(w_idx) = chan(5,1); w_idx = w_idx + 1;
                    payload(w_idx) = chan(5,2); w_idx = w_idx + 1;
                    payload(w_idx) = chan(6,1); w_idx = w_idx + 1;
                    payload(w_idx) = chan(6,2); w_idx = w_idx + 1;
                    payload(w_idx) = chan(7,1); w_idx = w_idx + 1;
                    payload(w_idx) = chan(7,2); w_idx = w_idx + 1;
                    payload(w_idx) = chan(8,1); w_idx = w_idx + 1;
                    payload(w_idx) = chan(8,2); w_idx = w_idx + 1;
                end
            end
            
            % Send data over socket
            fopen(sock(xid));
            fwrite(sock(xid), payload);
            fclose(sock(xid));
        end
    end
end
