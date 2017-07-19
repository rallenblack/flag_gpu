close all;

% Tone at 1367.3MHz in Bank C & D (Scan 1)
R_on_C = extract_covariances('/lustre/gbtdata/TGBT16A_508_01/TMP/BF/2017_05_23_14:02:35J.fits');
R_on_D = extract_covariances('/lustre/gbtdata/TGBT16A_508_01/TMP/BF/2017_05_23_14:02:35J.fits');

% No tone (Scan 2)
R_off_C = extract_covariances('/lustre/gbtdata/TGBT16A_508_01/TMP/BF/2017_05_22_01:36:57C.fits');
R_off_D = extract_covariances('/lustre/gbtdata/TGBT16A_508_01/TMP/BF/2017_05_22_01:36:57D.fits');

% Tone at 1367.3MHz in Bank C & D (Scan 3)
% [R_off_C,dmjd2,xid_c] = extract_covariances('/lustre/gbtdata/TGBT16A_508_01/TMP/BF/2016_10_07_15:38:36C.fits');
% [R_off_D,dmjd4,xid_d] = extract_covariances('/lustre/gbtdata/TGBT16A_508_01/TMP/BF/2016_10_07_15:38:36D.fits');

% Tone at 1367.3MHz in Bank C & D (Scan 4)
% [R_on_C,dmjd1,xid_c] = extract_covariances('/lustre/gbtdata/TGBT16A_508_01/TMP/BF/2016_10_07_15:49:39C.fits');
% [R_on_D,dmjd3,xid_d] = extract_covariances('/lustre/gbtdata/TGBT16A_508_01/TMP/BF/2016_10_07_15:49:39D.fits');


hot_c = mean(R_on_C,4);
hot_d = mean(R_on_D,4);
cold_c = mean(R_off_C,4);
cold_d = mean(R_off_D,4);

filename1 = 'Scan1_Otf_Hot_C.mat';
filename2 = 'Scan1_Otf_Hot_D.mat';
filename3 = 'Scan2_Otf_Cold_C.mat';
filename4 = 'Scan2_Otf_Cold_D.mat';
save(filename1, 'hot_c');
save(filename2, 'hot_d');
save(filename3, 'cold_c');
save(filename4, 'cold_d');

% fitsfilename = '/lustre/gbtdata/TGBT16A_508_01/TMP/BF/2016_10_07_15:38:36D.fits';
% info   = fitsinfo(fitsfilename);
%     bintbl = fitsread(fitsfilename, 'binarytable', 1);
%    
%     data = bintbl{3};
% filename1 = 'Cold_1367MHz_D.mat';
% save(filename1, 'data');
