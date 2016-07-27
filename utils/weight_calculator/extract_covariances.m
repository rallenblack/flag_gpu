function [ R, dmjd ] = extract_covariances( fits_filename )
%EXTRACT_COVARIANCES Function that extracts the covariance matrices and
%reconstructs them from the FITS format into a 2D matrix format

    info   = fitsinfo(fits_filename);
    bintbl = fitsread(fits_filename, 'binarytable', 1);
    
    dmjd = bintbl{1};
    mcnt = bintbl{2};
    data = bintbl{3};
    
    Nel = 64;
    Nbin = 25;
    Nsamp = 4000;

    R = reconstruct_covariances(data, Nel, Nbin, Nsamp);
    R = R(1:40, 1:40, :, :);
end

