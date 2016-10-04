function [ dmjd, az_off, el_off, ra, dec ] = get_antenna_positions( fits_file, use_radec )
%GET_ANTENNA_POSITIONS Summary of this function goes here
%   Detailed explanation goes here

    % Extract the second binary table entry of the FITS file
    data = fitsread(fits_file, 'binarytable', 2);

    dmjd_idx = 1;
    ra_idx = 2;
    dec_idx = 3;
    mnt_az_idx = 4;
    mnt_el_idx = 5;
    obsc_az_idx = 9;
    obsc_el_idx = 10;

    dmjd = data{dmjd_idx};
    
    % ra and dec
    ra = data{ra_idx};
    dec = data{dec_idx};
    
    if use_radec
        az_off = ra - ra(1);
        el_off = dec - dec(1);
    else
        % mnt entries correspond to encoder values
        mnt_az = data{mnt_az_idx};
        mnt_el = data{mnt_el_idx};

        % obsc entries correspond to commanded position of on-sky beam
        obsc_az = data{obsc_az_idx};
        obsc_el = data{obsc_el_idx};

        az_off = mnt_az - obsc_az;
        el_off = mnt_el - obsc_el;
    end
end

