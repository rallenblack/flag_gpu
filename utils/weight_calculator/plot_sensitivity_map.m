% Plots sensitivity map
clearvars;
close all;

global_params;

% Specify output MAT files
mat = sprintf('%s/%s%s.mat', dir, stamp, bank);
load(mat);

% Calculate off-pointing covariance matrix
Roff = mean(R(:,:,:,end-10:end), 4);

Ntime = size(R,4);
Nbin = size(R,3);

[v,w] = get_grid_weights(R, Roff, dir, stamp, bank, idx, overwrite);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute Sensitivity map
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Computing sensitivity map');
S = zeros(Ntime, Nbin);
Pon = zeros(Ntime, Nbin);
Poff = zeros(Ntime, Nbin);
for b = 1:Nbin
    for t = 1:Ntime
        Pon(t,b) = real(w(idx,b,t)'*R(idx,idx,b,t)*w(idx,b,t));
        Poff(t,b) = real(w(idx,b,t)'*Roff(idx,idx,b)*w(idx,b,t));
        SNR = (Pon(t,b) - Poff(t,b))/Poff(t,b);
        S(t,b) = 2*kb*SNR/(10^-26*flux_density);
    end
end

sens_filename = sprintf('%s/%s%s_sens_map.mat', dir, stamp, bank);
save(sens_filename, 'S', 'xid');

% Time samples of interest
idxs = 9:length(ANT.az_off)-22;

% Convert frequency bin indices to sky frequencies
xid_bins = [1:5, 101:105, 201:205, 301:305, 401:405] + 5*xid;
bin_width = 303.75e3;
bin_idx = -249:250;
freqs = bin_idx*bin_width + center_freq;
xid_freqs = freqs(xid_bins);

% Interpolated map
Npoints = 80;
minX = min(ANT.az_off(idxs));
maxX = max(ANT.az_off(idxs));
minY = min(ANT.el_off(idxs));
maxY = max(ANT.el_off(idxs));
xval = linspace(minX, maxX, Npoints);
yval = linspace(minY, maxY, Npoints);
[X,Y] = meshgrid(linspace(minX,maxX,Npoints), linspace(minY,maxY,Npoints));
for b = 1:Nbin
    fprintf('Bin %d/%d\n', b, Nbin);
    map_fig = figure();
    Sq = griddata(ANT.az_off(idxs), ANT.el_off(idxs), real(S(idxs,b)), X, Y);
    imagesc(xval, yval, Sq); colorbar;
    set(gca, 'ydir', 'normal');
    colormap('jet');
    xlabel('Azimuth Offset (degrees)');
    ylabel('Elevation Offset (degrees)');
    title('Formed Beam Sensitivity Map');
    
    filename = sprintf('figures/sens_map_3_11_missing_%.4fMHz.fig', xid_freqs(b)/1e6);
    savefig(map_fig, filename);
end

% Convert frequency bin indices to sky frequencies
bin_width = 303.75e3;
bin_idx = -249:250;
freqs = bin_idx*bin_width + center_freq;

% Calculate minimum beamformed Tsys
D = 100;
Ap = pi*(D/2)^2;
maxS = max(max(S));
Tsys_nap = Ap/maxS;
disp(Tsys_nap);
