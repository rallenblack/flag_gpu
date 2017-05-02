function [ B, dmjd, xid ] = extract_b_output( fits_filename )
%EXTRACT_COVARIANCES Function that extracts the covariance matrices and
%reconstructs them from the FITS format into a 2D matrix format

    info   = fitsinfo(fits_filename);
    bintbl = fitsread(fits_filename, 'binarytable', 1);
    
    dmjd = bintbl{1};
    mcnt = bintbl{2};
    data = bintbl{3};
    keywords = info.PrimaryData.Keywords;
    xid = -1;
    for i = 1:size(keywords, 1)
        if strcmp(keywords{i,1}, 'XID')
            xid = str2double(keywords{i,2});
            break;
        end
    end
    
    Npol = 4; % X self-polarized, Y self-polarized, XY polarized (real), XY polarized (imaginary)
    Nbeam = 7;
    Nbin = 25;
    Nsti = 100;

    B = reshape(data, Nbeam, Npol, Nbin, Nsti);
    
end


