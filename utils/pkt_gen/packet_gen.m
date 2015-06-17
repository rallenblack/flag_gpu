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
Nxengines = 10;    % Number of X-engines (i.e. Number of GPUs)

Nin_per_f        = Ninputs/Nfengines; % Number of inputs per F-engine
Nbin_per_x       = Nbins/Nxengines; % Number of bins per X-engine
Ntime_per_packet = 10; % Number of time samples (spectra snapshots) per packet

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
% else -> Send all zeros
data_flag = 3;

% Sinusoid parameters (only used if data_flag = 2)
% It should be noted that the phase of the sinusoid will not change between
% time samples-- this is just for convenience. A more sophisticated packet
% generator would incorporate the phase shifts across time.
s_bin = 1; % Sinusoid's absolute bin number (1-500)
s_ele = 1; % Sinusoid's absolute element number (1-40)
s_phi = pi/2; % Sinusoid's phase (magnitude is set to 1)
s_xid = floor((s_bin - 1)/100) + 1; % X-engine ID for desired bin
s_fid = floor((s_ele - 1)/8) + 1;   % F-engine ID for desired input
s_bin_r = mod(s_bin - 1, 100) + 1; % Relative bin number (internal fengine index)
s_ele_r = mod(s_bin - 1, 8) + 1; % Relative element number


% Create UDP sockets - 1 IP address per Xengine (xid)
for xid = 1:Nxengines
    remoteHost = ['10.10.', num2str(xid), '.233'];
    sock(xid) = udp(remoteHost, 'RemotePort', 8511, 'LocalPort', 8511);
    set(sock(xid), 'OutputBufferSize', 9000);
    set(sock(xid), 'OutputDatagramPacketSize', 9000);
end


% Generate packet payloads
mcnt = 0; % Each mcnt represents 10 packets across all F-engines in the
          % same time frame
while mcnt <= 1000
    disp(['Sending mcnt = ', num2str(mcnt)]);
    for xid = 1:1 % Set to a single X-engine for single HPC testing (Richard B.)
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
            data = zeros(Nin_per_f, 2, Nbin_per_x, Ntime_per_packet);
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
                        data(s_ele_r, 2, s_bin_r, :) = s_real;
                    end
                case 3 % Send all ones
                    data = ones(Nin_per_f, 2, Nbin_per_x, Ntime_per_packet);
                otherwise % Send all zeros
            end
            data = uint8(data);
            
            for nt = 1:Ntime_per_packet
                for nb = 1:Nbin_per_x
                    chan = squeeze(data(:,:,nb,nt));
                    chan = reshape(chan(:), 2, 8).';
                    
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
    mcnt = mcnt + 1;
end