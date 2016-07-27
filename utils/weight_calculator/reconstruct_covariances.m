function [ R ] = reconstruct_covariances( data, Nele, Nbin, Nsamp )
%RECONSTRUCT_COVARIANCES Converts vectorized lower-triangular block
%covariance matrices from FITS files into full 2D covariance matrices
%   Detailed explanation goes here

    Ncov = size(data, 1);
    R = zeros(Nele, Nele, Nbin, Ncov);
    
    Nbaselines_tot = (Nele/2 + 1)*Nele;
    Nblocks        = (Nele/2 + 1)*Nele/4;

    blk_rows = zeros(Nele/2, Nele/2);
    for i = 1:Nele/2
        blk_rows(i,1:i) = (i-1)*i/2+1:(i-1)*i/2+i;
    end

    for t = 1:Ncov
        fprintf('Reconstructing Row %d/%d\n', t, Ncov);
        for b = 1:Nbin
            b_off = Nbaselines_tot*(b-1)+1;
            rb = data(t, b_off:b_off + Nbaselines_tot-1);

            
            Rb = zeros(Nele, Nele);
            
            % Speed this up!!!!
            for Nblk = 1:Nblocks
                block_r = rb(4*(Nblk-1)+1:4*Nblk);
                [row, col] = find(blk_rows == Nblk);
                Rb(2*row - 1, 2*col - 1) = block_r(1);
                if sum(diag(blk_rows) == Nblk) == 0
                    Rb(2*row - 1, 2*col) = block_r(2);
                end
                Rb(2*row    , 2*col - 1) = block_r(3);
                Rb(2*row    , 2*col    ) = block_r(4);
            end

            Rb = Rb + (Rb' - diag(diag(Rb'))); % Exploit symmetry
            Rb = Rb./Nsamp;
            R(:,:,b,t) = Rb;
        end
    end
end

