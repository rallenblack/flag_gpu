% Specifies common-to-all parameters

% data_set selects a data set to analyze
% 1 => July 25, 2016, Grid Scan
% 2 => July 25, 2016, Daisy Scan
% 3 => July 29, 2016, Grid Scan
% 4 => July 31, 2016, Grid Scan, 3C295
data_set = 4;

% Specify if output files should be overwritten
overwrite = 1;

% Specify bank ID (only single-bank scripts use this)
bank = 'D';

% Create FITS file names
switch data_set
    case 1
        projID = 'TGBT16A_508_01';
        lustre_dir = '/lustre/gbtdata';
        sub_dir = 'BF';
        stamp = '2016_07_25_04:32:35';
        meta_projID = 'TGBT16A_508_02';
        meta_stamp = '2016_07_25_04:32:33';
        flux_density = -1; % Need to determine
        x_idx = [1, 20, 3:12, 32, 14, 21, 25, 17:19]; % placeholder
        y_idx = [15, 40, 23:24, 16, 26:28, 30, 29, 31, 13, 33:39]; % placeholder
        badX = [1,3,4,5,6,9,10,11,12,14,17,18,16,13]; % placeholder
        badY = [3,10,11,12]; % placeholder
    case 2
        projID = 'TGBT16A_508_01';
        lustre_dir = '/lustre/gbtdata';
        sub_dir = 'BF';
        stamp = '2016_07_25_05:05:26';
        meta_projID = 'TGBT16A_508_02';
        meta_stamp = '2016_07_25_05:05:24';
        flux_density = -1; % Need to determine
        x_idx = [1, 20, 3:12, 32, 14, 21, 25, 17:19]; % placeholder
        y_idx = [15, 40, 23:24, 16, 26:28, 30, 29, 31, 13, 33:39]; % placeholder
        badX = [1,3,4,5,6,9,10,11,12,14,17,18,16,13]; % placeholder
        badY = [3,10,11,12]; % placeholder
    case 3
        projID = 'TGBT16A_508_01';
        lustre_dir = '/lustre/gbtdata';
        sub_dir = 'TMP/BF';
        stamp = '2016_07_29_10:07:31';
        meta_projID = 'TGBT16A_508_03';
        meta_stamp = '2016_07_29_10:07:30';
        flux_density = 48.7; % Need to verify
        x_idx = [1, 20, 3:12, 32, 14, 15, 16, 17:19]; % placeholder
        y_idx = [21, 40, 23:24, 25, 26:28, 30, 29, 31, 13, 33:39]; % placeholder
        badX = []; % placeholder
        badY = [3, 11, 10,12]; % placeholder
    case 4
        projID = 'TGBT16A_508_01';
        lustre_dir = '/lustre/gbtdata';
        sub_dir = 'TMP/BF';
        stamp = '2016_07_31_23:33:51';
        meta_projID = 'TGBT16A_508_05';
        meta_stamp = '2016_07_31_23:33:51';
        flux_density = 22.15; % @ 1465 MHz
        x_idx = [1, 20, 3:12, 32, 14, 21, 25, 17:19];
        y_idx = [15, 40, 23:24, 16, 26:28, 30, 34, 31, 13, 33, 29, 35:39];
        badX = [1,3,4,5,6,9,10,11,12,14,17,18,16,13];
        badY = [12,14];
        center_freq = 1432.729e6;
end

meta_dir = '/home/gbtdata';
ant_sub_dir = 'Antenna';

dir = sprintf('%s/%s/%s', lustre_dir, projID, sub_dir);
ant_dir = sprintf('%s/%s/%s', meta_dir, meta_projID, ant_sub_dir);
ant_fits_file = sprintf('%s/%s.fits', ant_dir, meta_stamp);

% Specify which antenna elements will be used in beamformer and sensitivity
% measurements (data_set dependent)
all_idx = 1:40;
goodX = all_idx(x_idx);
goodX(badX) = [];
goodY = all_idx(y_idx);
goodY(badY) = [];

% Select y-pol for now
idx = goodY;

% Boltzmann's Constant
kb = 1.38e-23;
