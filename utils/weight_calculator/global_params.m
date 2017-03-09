% Specifies common-to-all parameters

% data_set selects a data set to analyze
% 1 => July 25, 2016, Grid Scan, 3C286
% 2 => July 25, 2016, Daisy Scan, 3C295
% 3 => July 29, 2016, Grid Scan, 3C123
% 4 => July 29, 2016, Daisy Scan, 3C123
% 5 => July 31, 2016, Grid Scan, 3C295
% 6 => July 30, 2016, Daisy Scan, 3C295
data_set = 1;

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
        off_stamp = '2016_07_25_04:32:35';
        meta_projID = 'TGBT16A_508_02';
        meta_stamp = '2016_07_25_04:32:33';
        meta_off_projID = 'TGBT16A_508_02';
        meta_off_stamp = '2016_07_25_04:32:33';
        flux_density = 14.90; % 3C286 @ 1465 MHz
        x_idx = [1:19];
        y_idx = [21:39];
        badX = [];
        badY = [2,9,12,17];
        idxs = [13:105, 209:1263];
        off_idxs = [5130:5140];
        center_freq = 1445.565e6;
        dmjd_delay = 10.8/(24*3600);
        dmjd_delay_off = dmjd_delay;
        az_vals = [-0.08496, 0.09283, -0.1496,   0.003936,  0.1494,  -0.09304, 0.09283];
        el_vals = [ 0.1639,  0.1639, 0.002164,  0.002164, 0.002164,  -0.1014, -0.1014];
        use_radec = 0;
    case 2
        projID = 'TGBT16A_508_01';
        lustre_dir = '/lustre/gbtdata';
        sub_dir = 'BF';
        stamp = '2016_07_25_05:05:26';
        meta_projID = 'TGBT16A_508_02';
        meta_stamp = '2016_07_25_05:05:24';
        off_stamp = '2016_07_25_04:32:35';
        meta_off_projID = 'TGBT16A_508_02';
        meta_off_stamp = '2016_07_25_04:32:33';
        flux_density = 22.15; % @ 1465 MHz
        x_idx = [1:19];
        y_idx = [21:39];
        badX = [];
        badY = [];
        idxs = [1:299];
        off_idxs = [5130:5140];
        center_freq = 1445.565e6;
        dmjd_delay = 10.8/(24*3600);
        dmjd_delay_off = dmjd_delay;
        use_radec = 0;
    case 3
        projID = 'TGBT16A_508_01';
        lustre_dir = '/lustre/gbtdata';
        sub_dir = 'TMP/BF';
        stamp = '2016_07_29_10:07:31';
        meta_projID = 'TGBT16A_508_03';
        meta_stamp = '2016_07_29_10:07:30';
        off_stamp = '2016_07_29_10:07:31';
        meta_off_projID = 'TGBT16A_508_03';
        meta_off_stamp = '2016_07_29_10:07:30';
        dmjd_delay = 2.75/(24*3600);
        dmjd_delay_off = dmjd_delay;
        flux_density = 48.7; % @ ? MHz
        idxs = [5:1372];
        off_idxs = [5533:5543];
        x_idx = [1, 20, 3:12, 32, 14:19];
        y_idx = [21, 40, 23:28, 30, 29, 31, 13, 33:39];
        badX = [];
        badY = [10, 12];
        center_freq = 1432.729e6;
        az_vals = [-0.1436, 0.1428, -0.2806, 0.005841, 0.2798, -0.1436,   0.1428];
        el_vals = [ 0.2088, 0.2088,  0.1122, 0.1122,   0.1122,  0.009618, 0.009618];
        use_radec = 0;
    case 4
        projID = 'TGBT16A_508_01';
        lustre_dir = '/lustre/gbtdata';
        sub_dir = 'TMP/BF';
        stamp = '2016_07_29_11:59:12';
        meta_projID = 'TGBT16A_508_03';
        meta_stamp = '2016_07_29_11:59:10';
        off_stamp = '2016_07_29_10:07:31';
        meta_off_projID = 'TGBT16A_508_03';
        meta_off_stamp = '2016_07_29_10:07:30';
        flux_density = 22.15; % @ 1465 MHz
        idxs = [5:1372];
        off_idxs = [5533:5543];
        x_idx = [1, 20, 3:12, 32, 14:19];
        y_idx = [21, 40, 23:28, 30, 29, 31, 13, 33:39];
        badX = [];
        badY = [10, 12];
        center_freq = 1432.729e6;
        dmjd_delay = 2.75/(24*3600);
        dmjd_delay_off = dmjd_delay;
        use_radec = 0;
    case 5
        projID = 'TGBT16A_508_01';
        lustre_dir = '/lustre/gbtdata';
        sub_dir = 'TMP/BF';
        stamp = '2016_07_31_23:33:51';
        meta_projID = 'TGBT16A_508_05';
        meta_stamp = '2016_07_31_23:33:51';
        off_stamp = '2016_07_31_23:33:51';
        meta_off_projID = 'TGBT16A_508_05';
        meta_off_stamp = '2016_07_31_23:33:51';
        flux_density = 22.15; % @ 1465 MHz
        idxs = [9:1373];
        off_idxs = [5573:5583];
        x_idx = [1, 20, 3:12, 32, 14, 21, 25, 17:19];
        y_idx = [15, 40, 23:24, 16, 26:28, 30, 34, 31, 13, 33, 29, 35:39];
        badX = [1,3,4,5,6,9,10,11,12,14,17,18,16,13];
        badY = [12,14];
        center_freq = 1432.729e6;
        dmjd_delay = 2.75/(24*3600);
        dmjd_delay_off = dmjd_delay;
        use_radec = 0;
    case 6
        projID = 'TGBT16A_508_01';
        lustre_dir = '/lustre/gbtdata';
        sub_dir = 'TMP/BF';
        stamp = '2016_07_30_23:23:45';
        meta_projID = 'TGBT16A_508_04';
        meta_stamp = '2016_07_30_23:23:43';
        off_stamp = '2016_07_30_23:23:45';
        meta_off_projID = 'TGBT16A_508_04';
        meta_off_stamp = '2016_07_30_23:42:44';
        flux_density = 22.15; % @ 1465 MHz
        idxs = [1:1080];
        off_idxs = [4816:4826];
        x_idx = [1, 20, 3:12, 32, 14, 21, 25, 17:19];
        y_idx = [15, 40, 23:24, 16, 26:28, 30, 34, 31, 13, 33, 29, 35:39];
        badX = [];
        badY = [12, 14];
        az_vals = [-0.04, 0.155, -0.09, 0.11, 0.3, -0.04, 0.155];
        el_vals = [ 0.12, 0.12, 0.01, 0.01, 0.01, -0.11, -0.11];
        center_freq = 1432.729e6;
        dmjd_delay = 2.9/(24*3600);
        dmjd_delay_off = dmjd_delay;
        use_radec = 1;
end

meta_dir = '/home/gbtdata';
ant_sub_dir = 'Antenna';

dir = sprintf('%s/%s/%s', lustre_dir, projID, sub_dir);
ant_dir = sprintf('%s/%s/%s', meta_dir, meta_projID, ant_sub_dir);
ant_off_dir = sprintf('%s/%s/%s', meta_dir, meta_off_projID, ant_sub_dir);
ant_fits_file = sprintf('%s/%s.fits', ant_dir, meta_stamp);
ant_off_fits_file = sprintf('%s/%s.fits', ant_dir, meta_off_stamp);

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
