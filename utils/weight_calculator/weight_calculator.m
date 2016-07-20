% Script to parse Antenna FITS files

% Antenna FITS filename
fits_file = '2016_01_15_23:36:23.fits';

% Extract the FITS file information
info = fitsinfo(fits_file);

% Extract the second binary table entry of the FITS file
data = fitsread(fits_file, 'binarytable', 2);

dmjd_idx = 1;
mnt_az_idx = 4;
mnt_el_idx = 5;
obsc_az_idx = 9;
obsc_el_idx = 10;

dmjd = data{dmjd_idx};

% mnt entries correspond to encoder values
mnt_az = data{mnt_az_idx};
mnt_el = data{mnt_el_idx};

% obsc entries correspond to commanded position of on-sky beam
obsc_az = data{obsc_az_idx};
obsc_el = data{obsc_el_idx};



