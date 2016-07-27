close all;
clearvars;

% Create FITS file names
projID = 'TGBT16A_508_01';
lustre_dir = '/lustre/gbtdata';
sub_dir = 'BF';
dir = sprintf('%s/%s/%s', lustre_dir, projID, sub_dir);
stamp = '2016_07_25_04:32:35';

% Specify FITS files
fitsA = sprintf('%s/%sA.fits', dir, stamp);
fitsB = sprintf('%s/%sB.fits', dir, stamp);
fitsC = sprintf('%s/%sC.fits', dir, stamp);
fitsD = sprintf('%s/%sD.fits', dir, stamp);
fits = {fitsA, fitsB, fitsC, fitsD};

% Specify output MAT files
matA = sprintf('%s/%sA.mat', dir, stamp);
matB = sprintf('%s/%sB.mat', dir, stamp);
matC = sprintf('%s/%sC.mat', dir, stamp);
matD = sprintf('%s/%sD.mat', dir, stamp);
mat = {matA, matB, matC, matD};

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reconstruct covariance matrices from FITS files
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1:4
    if ~exist(mat{i}, 'file')
        [R, data_dmjd] = extract_covariances(fits{i});
        save(mat{i}, 'R', 'data_dmjd', '-v7.3');
    end
end

load(mat{1});

% Calculate off-pointing covariance matrix
Roff = mean(R(:,:,:,end-40:end), 4);

% Remove bad channels
idx = 1:40;
rm_idx = [13, 22];
idx(rm_idx) = [];

if 0
    Ntime = size(R,4);
    figure(1);
    for t = 1:5:Ntime
        imagesc(10*log10(abs(Roff(idx,idx,1)\R(idx,idx,1,t))));
        title(['Time = ', num2str(t)]);
        drawnow;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Extract antenna positions for scan
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Change projID because folder in lustre wasn't created properly
if 1
    projID = 'TGBT16A_508_02';
end
% Create fits file name
meta_dir = '/home/gbtdata';
sub_dir = 'Antenna';
dir = sprintf('%s/%s/%s', meta_dir, projID, sub_dir);
stamp = '2016_07_25_04:32:33';
ant_fits_file = sprintf('%s/%s.fits', dir, stamp);

% Extract offsets
[ant_dmjd, az_off, el_off] = get_antenna_positions(ant_fits_file);

% Plot the trajectory
figure();
plot(az_off, el_off, '-b', az_off(1), el_off(1), 'rx');
title('Grid Trajectory - Encoder Data');
xlabel('Azimuth Angle (deg)');
ylabel('Elevation Angle (deg)');
axis equal;

