% Plots map of eigenvalues

close all;
clearvars;

global_params;

% Specify output MAT files
mat = sprintf('%s/%s%s.mat', dir, stamp, bank);
load(mat);

% Calculate off-pointing covariance matrix
Roff = mean(R(:,:,:,end-10:end), 4);

Ntime = size(R,4);
Nbin = size(R,3);

% Select bin
b = 6;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot map of dominant eigenvalue
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
lambda = zeros(Ntime, 1);
for t = 1:Ntime
    lambda(t) = real(eigs(R(idx,idx,b,t), Roff(idx,idx,b), 1));
end

% Time samples of interest
idxs = 9:length(ANT.az_off)-22;

% Interpolated map
Npoints = 80;
figure();
minX = min(ANT.az_off(idxs));
maxX = max(ANT.az_off(idxs));
minY = min(ANT.el_off(idxs));
maxY = max(ANT.el_off(idxs));
xval = linspace(minX, maxX, Npoints);
yval = linspace(minY, maxY, Npoints);
[X,Y] = meshgrid(linspace(minX,maxX,Npoints), linspace(minY,maxY,Npoints));
Sq = griddata(ANT.az_off(idxs), ANT.el_off(idxs), real(lambda(idxs)), X, Y);
imagesc(xval, yval,Sq); colorbar;
set(gca, 'ydir', 'normal');
colormap('jet');
