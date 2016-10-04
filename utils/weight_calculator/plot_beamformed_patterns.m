% Plots beamformed patterns

clearvars;
close all;

global_params;

% Specify output MAT files
mat = sprintf('%s/%s%s.mat', dir, stamp, bank);
load(mat);

% Calculate off-pointing covariance matrix
Roff = mean(R(:,:,:,end-10:end), 4);
Roff = R_off;

Ntime = size(R,4);
Nbin = size(R,3);

[v,w] = get_grid_weights(R, Roff, dir, stamp, bank, idx, overwrite);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute Beamformed Beampatterns
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Computing beamformed beampattern maps');

Nbeam = length(az_vals);
pattern = zeros(Nbeam, Ntime, Nbin);
weights = zeros(64, size(v,2), Nbeam);
for beam = 1:Nbeam
    [~,beam_idx] = min(abs(ANT.az_off - az_vals(beam)).^2 + abs(ANT.el_off - el_vals(beam)).^2);
    
    for b = 1:Nbin
        w_beam = w(idx, b, beam_idx);
        weights(idx, b, beam) = w_beam;
        for t = 1:Ntime
            pattern(beam,t,b) = abs(w_beam'*v(idx,b,t)).^2;
        end
    end
end


Npoints = 240;
minX = min(ANT.az_off(idxs));
maxX = max(ANT.az_off(idxs));
minY = min(ANT.el_off(idxs));
maxY = max(ANT.el_off(idxs));
xval = linspace(minX, maxX, Npoints);
yval = linspace(minY, maxY, Npoints);
[X,Y] = meshgrid(linspace(minX,maxX,Npoints), linspace(minY,maxY,Npoints));

% Convert frequency bin indices to sky frequencies
xid_bins = [1:5, 101:105, 201:205, 301:305, 401:405] + 5*xid;
bin_width = 303.75e3;
bin_idx = -249:250;
freqs = bin_idx*bin_width + center_freq;
xid_freqs = freqs(xid_bins);

%plot_idx = [2 4 6 8 10 12 14];
plot_idx = [1 3 4 5 6 7 9];
myfig = figure();
for b = 1:Nbin
    fprintf('Bin %d/%d\n', b, Nbin);
    for beam = 1:Nbeam
        beam_fig = figure();
        Pq = griddata(ANT.az_off(idxs), ANT.el_off(idxs), pattern(beam,idxs,b), X, Y);
        imagesc(xval, yval, 10*log10(abs(Pq))); colorbar; hold on;
        caxis([-30, 0]);
        contour(gca, xval, yval, 10*log10(abs(Pq)), [-3 -3], '-k'); hold off;
        set(gca, 'ydir', 'normal');
        title(sprintf('Beam %d', beam));
        axis square;
        xlabel('Azimuth Offset (degrees)');
        ylabel('Elevation Offset (degrees)');
        colormap('jet');
        filename = sprintf('figures/beamformed%d_pattern_July30_daisy_%fMHz.fig', beam, xid_freqs(b)/1e6);
        savefig(beam_fig, filename);
        close(beam_fig);

        figure(myfig);
        subplot(3,3,plot_idx(beam));
        imagesc(xval, yval, 10*log10(abs(Pq))); colorbar; hold on;
        caxis([-30, 0]);
        contour(gca, xval, yval, 10*log10(abs(Pq)), [-3 -3], '-k'); hold off;
        set(gca, 'ydir', 'normal');
        colormap('jet');
        title(sprintf('Beam %d', beam));
        axis square;
    end
    
    filename = sprintf('figures/beamformed_patterns_July_30_daisy_%.4fMHz.fig', xid_freqs(b)/1e6);
    savefig(myfig, filename);
    close(myfig);
    myfig = figure();
end
close(myfig);

% Save weights into 7-beam binary file
weightsX_vectorized = single(zeros(64*Nbin*Nbeam*2,1));
weightsY_vectorized = single(zeros(64*Nbin*Nbeam*2,1));
weightsY_vectorized(1:2:end) = single(real(weights(:)));
weightsY_vectorized(2:2:end) = single(imag(weights(:)));

payload = [weightsX_vectorized; weightsY_vectorized];
offsets = single([az_vals; el_vals]*60);
cal_filename = sprintf('%s%s', stamp, bank);
algorithm = 'Max-SNR';

for i = (length(cal_filename)+1):64
    cal_filename = [cal_filename, ' '];
end
for i = (length(algorithm)+1):64
    algorithm = [algorithm, ' '];
end

weight_filename = sprintf('%s/%s_xid%d_weights.bin', dir, stamp, xid);
fprintf('Saving %s...\n', weight_filename);
FID = fopen(weight_filename, 'wb');
fwrite(FID, payload, 'float');
fwrite(FID, offsets(:), 'float');
fwrite(FID, cal_filename, 'char*1');
fwrite(FID, algorithm, 'char*1');
fwrite(FID, uint64(xid), 'uint64');
fclose(FID);

