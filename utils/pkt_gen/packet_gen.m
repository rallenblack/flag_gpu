% FLAG packet generator
u = instrfindall;
delete(u);

% System Constants
fs        = 155e6;
Ninputs   = 40;
Nbins     = 500;
Nfft      = 512;
Nfengines = 5;
Nxengines = 10;

Nin_per_f        = Ninputs/Nfengines;
Nbin_per_x       = Nbins/Nxengines;
Ntime_per_packet = 10;

quant_res = 0.5e-8;

% Noise parameters
kb     = 1.3806488e-23;
Tsys   = 45;
BW     = fs/Nfft;
sigma2 = kb*Tsys*BW;


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
s_bin = 1; % Sinusoid's bin number (1-500)
s_ele = 1; % Sinusoid's element number (1-40)
s_phi = pi/2; % Sinusoid's phase (magnitude is set to 1)
s_xid = floor((s_bin - 1)/100) + 1;
s_fid = floor((s_ele - 1)/8) + 1;
s_bin_r = mod(s_bin - 1, 100) + 1; % Relative bin number (internal fengine index)
s_ele_r = mod(s_bin - 1, 8) + 1; % Relative element number



% Create UDP sockets
for xid = 1:Nxengines
    remoteHost = ['10.10.', num2str(xid+3), '.233'];
    sock(xid) = udp(remoteHost, 'RemotePort', 8511, 'LocalPort', 8511);
    set(sock(xid), 'OutputBufferSize', 9000);
    set(sock(xid), 'OutputDatagramPacketSize', 9000);
end


% Generate packet payloads
mcnt = 0;
while mcnt <= 1000
    disp(['Sending mcnt = ', num2str(mcnt)]);
    for xid = 1:1
        for fid = 1:Nfengines
            w_idx = 1;
            
            % Create packet header
            header = uint64(mcnt)*2^20 + uint64(15)*2^16 + uint64(fid-1)*2^8 + uint64(xid-1);
            
            % Allocate memory for packet payload
            payload = zeros(16*Ntime_per_packet*Nbin_per_x+8, 1, 'uint8');
            
            % Shift header information into packet
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