% Plots minimum Tsys over frequency

clearvars;
close all;

global_params;

sensA = sprintf('%s/%s%s_sens_map.mat', dir, stamp, 'A');
sensB = sprintf('%s/%s%s_sens_map.mat', dir, stamp, 'B');
sensC = sprintf('%s/%s%s_sens_map.mat', dir, stamp, 'C');
sensD = sprintf('%s/%s%s_sens_map.mat', dir, stamp, 'D');
sensE = sprintf('%s/%s%s_sens_map.mat', dir, stamp, 'E');
sensF = sprintf('%s/%s%s_sens_map.mat', dir, stamp, 'F');

sA = load(sensA);
sB = load(sensB);
sC = load(sensC);
sD = load(sensD);
sE = load(sensE);
sF = load(sensF);

D = 100;
Ap = pi*(D/2)^2;

maxS = max(sA.S,[],1);
TsysA_eta = Ap./maxS;

maxS = max(sB.S,[],1);
TsysB_eta = Ap./maxS;

maxS = max(sC.S,[],1);
TsysC_eta = Ap./maxS;

maxS = max(sD.S,[],1);
TsysD_eta = Ap./maxS;

maxS = max(sE.S,[],1);
TsysE_eta = Ap./maxS;

maxS = max(sF.S,[],1);
TsysF_eta = Ap./maxS;

Tsys_eta = zeros(500, 1);
f_idx = [1:5, 101:105, 201:205, 301:305, 401:405];
Tsys_eta(f_idx + sA.xid*5) = TsysA_eta;
Tsys_eta(f_idx + sB.xid*5) = TsysB_eta;
Tsys_eta(f_idx + sC.xid*5) = TsysC_eta;
Tsys_eta(f_idx + sD.xid*5) = TsysD_eta;
Tsys_eta(f_idx + sE.xid*5) = TsysE_eta;
Tsys_eta(f_idx + sF.xid*5) = TsysF_eta;

% Convert frequency bin indices to sky frequencies
bin_width = 303.75e3;
bin_idx = -249:250;
freqs = bin_idx*bin_width + center_freq;

figure();
plot(freqs/1e6, Tsys_eta);
xlabel('Frequency (MHz)');
ylabel('Tsys/\eta_{ap}');
title('Minimum Normalized Tsys in FoV');
grid on;

figure();
plot(freqs/1e6, Tsys_eta*0.65);
xlabel('Frequency (MHz)');
ylabel('Tsys (K)');
title('Minimum Tsys in FoV');
grid on;
