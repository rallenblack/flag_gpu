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
% fo = freq(floor(length(freq)/2)); % Center frequency
% tau = 4.1488e-3*((fo^-2)-(freq.^-2))*D; % Frequency dependent timing offset
% t = -2.8e-20:((2.5e-20)+(2.8e-20))/(Ntime-1):2.5e-20; % Range of timing offsets
tau = zeros(size(freq));
for k = 1:length(freq)
    if (k-1) ~= 0
        tau(k-1) = 4.1488e-3*((freq(k-1)^-2)-(freq(k)^-2))*D;
    end
end
tau(end) = 4.1488e-3*((freq(end-1)^-2)-(freq(end)^-2))*D;
t = 5e-24:((10e-23)-(5e-24))/(Ntime-1):10e-23; % Range of timing offsets
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
pulse = 100;
for cyc = [0]
    for m = 1:Ninputs
        for k = 1:Nbins
%             tmp = abs(t - tau(k));
%             [~,idx] = min(tmp);
%             phi = m*2*pi*freq(k)*t(idx+cyc);
%             pulseData(m,k,idx+cyc) = pulse*exp(1j*phi) + pulseData(m,k,idx+cyc); % 0.1*(randn(1) + 1j*randn(1));
            [tmp,idx] = min(abs(t - tau(k)));
            phi = m*2*pi*freq(k)*t(idx+cyc);
            noise = pulseData(m,k,idx+cyc);
            pulseData(m,k,idx+cyc) = pulse*exp(1j*phi) + noise; % 0.1*(randn(1) + 1j*randn(1));
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

wei = ones(40,500,14);
bf_data = zeros(4000,500,14);
for b = 1:14
    for f = 1:500
        w = wei(:,f,b);
        xn = squeeze(pulseData(:,f,:));
        bf_data(:,f,b) = w'*xn;
    end
end

bf_sti = zeros(100,25,14);
for k = 1:100
    bf_sti(k,:,:) = mean(bf_data(1+(k-1)*40:k*40,1:25,:),1);
end

figure(2);
imagesc(10*log10(abs(bf_data(:,:,6))).');
title('Simulated pulsar output (500 bins)');
ylabel('Frequency bin index');
xlabel('Time samples')

figure(3);
imagesc(10*log10(abs(bf_sti(:,:,6))).');
title('Simulated pulsar STI output (25 bins)');
ylabel('Frequency bin index');
xlabel('Time samples')