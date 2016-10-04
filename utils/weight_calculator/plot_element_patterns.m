% Plots element beam patterns

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

% Look at all Y-Pol elements
idx = y_idx;
[v,~] = get_grid_weights(R, Roff, dir, stamp, bank, idx, overwrite);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute Element Beampatterns
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Computing element beampattern maps');
pattern = zeros(length(idx), Ntime, Nbin);
for b = 1:Nbin
    for t = 1:Ntime
        for e = 1:length(idx)
            w = zeros(length(idx), 1);
            w(e) = 1;
            pattern(e,t,b) = real(abs(w'*v(idx,b,t)).^2);
        end
    end
end


% Convert frequency bin indices to sky frequencies
xid_bins = [1:5, 101:105, 201:205, 301:305, 401:405] + 5*xid;
bin_width = 303.75e3;
bin_idx = -249:250;
freqs = bin_idx*bin_width + center_freq;
xid_freqs = freqs(xid_bins);

% Interpolated map
Npoints = 80;
if use_radec
    minX = max(min(ANT.az_off(idxs)), -0.45);
    maxX = min(max(ANT.az_off(idxs)), 0.45);
    minY = max(min(ANT.el_off(idxs)), -0.35);
    maxY = min(max(ANT.el_off(idxs)), 0.35);
else
    minX = min(ANT.az_off(idxs));
    maxX = max(ANT.az_off(idxs));
    minY = min(ANT.el_off(idxs));
    maxY = max(ANT.el_off(idxs));
end
xval = linspace(minX, maxX, Npoints);
yval = linspace(minY, maxY, Npoints);
[X,Y] = meshgrid(linspace(minX,maxX,Npoints), linspace(minY,maxY,Npoints));

for b = 1:Nbin
    fprintf('Bin %d/%d\n', b, Nbin);
    beam_fig = figure();
    for e = 1:length(idx)
        subplot(4,5,e);
        Pq = griddata(ANT.az_off(idxs), ANT.el_off(idxs), pattern(e,idxs,b).', X, Y);
        imagesc(xval, yval, 10*log10(abs(Pq))); colorbar; hold on;
        [c,~] = contour(gca, xval, yval, 10*log10(abs(Pq)), [-3 -3], '-k'); hold off;
        set(gca, 'ydir', 'normal');
        title(sprintf('%dY', e));
        colormap('jet');
        if (~isempty(c))
            ellipse_t(b,e) = fit_ellipse(c(1,2:end), c(2,2:end));
        end
    end
    % Create wiki-compatible table of weights
    
    
    filename = sprintf('figures/elem_patterns_July30_daisy_%.4fMHz.fig', xid_freqs(b)/1e6);
    savefig(beam_fig, filename);
end


