fs        = 155e6; % Sampling frequency - used for noise level
Ninputs   = 40;    % Number of inputs/antennas
Nbins     = 500;   % Total number of frequency bins
Nfft      = 512;   % F-engine FFT size
Nfengines = 5;     % Number of F-engines
Nxengines = 20;    % Number of X-engines (i.e. Number of GPUs)

Nin_per_f        = Ninputs/Nfengines; % Number of inputs per F-engine
Nbin_per_x       = Nbins/Nxengines; % Number of bins per X-engine
Ntime_per_packet = 20; % Number of time samples (spectra snapshots) per packet
Ntime = 4000;

% Increase the range of tau when dispersion measure causes m_D to exceed
% time samples.
D = 10; % DM; 10 with these parameters gives a fairly fast pulsar
freq = (0:499)*(303e3) + 1300e6; % All frequencies
fo = freq(floor(length(freq)/2)); % Center frequency
m_D = 4.1488e3*((fo^-2)-(freq.^-2))*D; % Frequency dependent timing offset
tau = -2.8e-14:((2.5e-14)+(2.8e-14))/(Ntime-1):2.5e-14; % Range of timing offsets
% tau = -3e-14:(max(m_D)+3e-14)/(Nbins-1):max(m_D); % Range of timing offsets
pulseData = zeros(Ninputs, Nbins, Ntime);
% Noisy environment
for ii = 1:size(pulseData,3)
    for jj = 1:size(pulseData,2)
        pulseData(:,jj,ii) = 0.1*(randn(1) + 1j*randn(1));
    end
end

% for cyc = [0,1000,2000,3000] % Cycles
%     for k = 1:Nbins
%         tmp_tau = abs(m_D(k)-tau);
%         [~, t_idx(k)] = min(tmp_tau);
%         pulseData(:,k,t_idx(k)+cyc) = exp(1j*pi/4)*2/sqrt(2); % 1+1j
%         pulseData(:,k,(t_idx(k)+cyc)+4) = exp(1j*pi/4); % 0.707+0.707j
%         if ((t_idx(k)+cyc)-4) > 0
%             pulseData(:,k,(t_idx(k)+cyc)-4) = exp(1j*pi/4); % 0.707+0.707j
%         end
%     end
% end

% Pulsar
pulse = 1;
for cyc = [0, -1000, 1000]
    for m = 1:Ninputs
        for k = 1:Nbins
            tmp = abs(m_D(k)-tau);
            [~,idx] = min(tmp);
            phi = m*2*pi*freq(k)*tau(idx+cyc);
            pulseData(m,k,idx+cyc) = pulse*exp(1j*phi) + 0.1*(randn(1) + 1j*randn(1));
        end
    end
end

%%%%% In switch case statement
% t_idxs = mod(mcnt*20 + 1:(mcnt+1)*20, PNtime);
% t_idxs(t_idxs == 0) = PNtime;
% f_idxs = (fid - 1)*8+1:fid*8;
% freq_idxs = 5*(xid-1) + [1:5, 101:105, 201:205, 301:305, 401:405];
% tmp = pulseData(f_idxs, freq_idxs, t_idxs);
% data(:,1,:,:) = real(tmp);
% data(:,2,:,:) = imag(tmp);
% data(data < 0) = 2^8 + data(data < 0);

%%
figure(1); 
imagesc(squeeze(abs(pulseData(20,:,:))));
% imagesc(abs(exp(1j*phi)));