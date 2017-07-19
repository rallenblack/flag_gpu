function [ R, dmjd, xid ] = extract_covariances( fits_filename )
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
    
    
    Nel = 64;
    Nbin = 25;
    Nsamp = 4000;

    R = reconstruct_covariances_bdj(data, Nel, Nbin, Nsamp);
    R = R(1:40, 1:40, :, :);
    
    % Bias subtraction
    mu = 0.5*ones(40, 1);
    for i = 1:size(R,3)
        for j = 1:size(R,4)
            R(:,:,i,j) = R(:,:,i,j) - mu*mu';
        end
    end
end

function [ R ] = reconstruct_covariances_bdj( data, Nele, Nbin, Nsamp )
%RECONSTRUCT_COVARIANCES Converts vectorized lower-triangular block
%covariance matrices from FITS files into full 2D covariance matrices
%   Detailed explanation goes here
    Ncov = size(data, 1);
    R = zeros(Nele, Nele, Nbin, Ncov);
    
    Nbaselines_tot = (Nele/2 + 1)*Nele;

%   Create index map for correlator output
%
    RIdx = (Nbaselines_tot+1)*ones(Nele,Nele);
    cnt = 1;
    for i=0:2:Nele - 2
        for j = 0:2:i
            for ii=1:2
                for jj = 1:2
                    if ii == 1 & jj == 2 & i == j
                        RIdx(i+ii,j+jj) = Nbaselines_tot+1;
                    else
                        RIdx(i+ii,j+jj) = cnt;
                    end
                    cnt = cnt+1;
                end
            end
        end
    end
    RIdx = RIdx(:);
 
%   Remap each correlator output into a full covariance matrix
%   Do this for each STI time and each freq. channel
%
    for t = 1:Ncov
        if mod(t,100) == 0
            fprintf('Reconstructing Row %d/%d\n', t, Ncov);
        end
        for b = 1:Nbin
            b_off = Nbaselines_tot*(b-1)+1;
            rb = [data(t, b_off:b_off + Nbaselines_tot-1),0];
            Rb = reshape(rb(RIdx),Nele,Nele);             
            Rb = Rb + (Rb' - diag(diag(Rb'))); % Exploit symmetry
            Rb = Rb./(Nsamp - 1);
            R(:,:,b,t) = Rb;
        end
    end
end
