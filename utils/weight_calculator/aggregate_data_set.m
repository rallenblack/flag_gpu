% Aggregates antenna positions with covariance matrices

close all;
clearvars;

global_params;

% Specify FITS files
fitsA = sprintf('%s/%sA.fits', dir, stamp);
fitsB = sprintf('%s/%sB.fits', dir, stamp);
fitsC = sprintf('%s/%sC.fits', dir, stamp);
fitsD = sprintf('%s/%sD.fits', dir, stamp);
fitsE = sprintf('%s/%sE.fits', dir, stamp);
fitsF = sprintf('%s/%sF.fits', dir, stamp);
fits = {fitsA, fitsB, fitsC, fitsD, fitsE, fitsF};

% Specify off FITS files
fits_offA = sprintf('%s/%sA.fits', dir, off_stamp);
fits_offB = sprintf('%s/%sB.fits', dir, off_stamp);
fits_offC = sprintf('%s/%sC.fits', dir, off_stamp);
fits_offD = sprintf('%s/%sD.fits', dir, off_stamp);
fits_offE = sprintf('%s/%sE.fits', dir, off_stamp);
fits_offF = sprintf('%s/%sF.fits', dir, off_stamp);
fits_off = {fits_offA, fits_offB, fits_offC, fits_offD, fits_offE, fits_offF};

% Specify output MAT files
matA = sprintf('%s/%sA.mat', dir, stamp);
matB = sprintf('%s/%sB.mat', dir, stamp);
matC = sprintf('%s/%sC.mat', dir, stamp);
matD = sprintf('%s/%sD.mat', dir, stamp);
matE = sprintf('%s/%sE.mat', dir, stamp);
matF = sprintf('%s/%sF.mat', dir, stamp);
mat = {matA, matB, matC, matD, matE, matF};

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reconstruct covariance matrices from FITS files
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1:length(fits)
    if exist(fits{i}, 'file')
        if ~exist(mat{i}, 'file') || overwrite
            fprintf('Extracting covariances for %s\n', fits{i});
            [Rtmp, dmjd, xid] = extract_covariances(fits{i});
            if strcmp(fits{i}, fits_off{i}) ~= 1
                fprintf('Extracting off covariances for %s\n', fits{i});
                [Rtmp_off, dmjd_off, xid_off] = extract_covariances(fits_off{i});
            else
                fprintf('Off pointing file same as on\n');
                Rtmp_off = Rtmp;
                dmjd_off = dmjd;
                xid_off = xid;
            end
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % ON Pointing Aggregation
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Apply additional integration
            Ntime = size(Rtmp,4);
            R = zeros(size(Rtmp,1), size(Rtmp,2), size(Rtmp,3), floor(Ntime/4)-1);
            dmjd = dmjd + dmjd_delay;
            corrected_dmjd = (15150/15187.5)*(dmjd - dmjd(1)) + dmjd(1);

            data_dmjd = zeros(floor(Ntime/4)-1,1);
            for j = 1:floor(Ntime/4)-1
                R(:,:,:,j) = sum(Rtmp(:,:,:,4*(j-1)+1:4*j),4)/4;
                data_dmjd(j) = sum(corrected_dmjd(4*(j-1)+1:4*j))/4;
            end
    
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Extract antenna positions for scan
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            disp('Extracting antenna positions and computing offsets');

            % Extract offsets
            [ant_dmjd, az_off, el_off, ra, dec] = get_antenna_positions(ant_fits_file, use_radec);

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Associate offsets with correlation matrices
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            disp('Associating pointing offsets with correlation matrices');
            Ntime = size(R, 4);
            mymin  = zeros(Ntime, 1);
            my_el  = zeros(Ntime, 1);
            my_az  = zeros(Ntime, 1);
            my_ra  = zeros(Ntime, 1);
            my_dec = zeros(Ntime, 1);
            for t = 1:Ntime
                cur_dmjd = data_dmjd(t);
                tmp_dmjd = ant_dmjd;
                [~, idx1] = min(abs(tmp_dmjd - cur_dmjd));

                tmp_dmjd(idx1) = NaN;
                [~, idx2] = min(abs(tmp_dmjd - cur_dmjd));

                x1 = min(ant_dmjd(idx1), ant_dmjd(idx2));
                x2 = max(ant_dmjd(idx1), ant_dmjd(idx2));
                az1 = min(az_off([idx1,idx2]));
                az2 = max(az_off([idx1,idx2]));
                el1 = min(el_off([idx1,idx2]));
                el2 = max(el_off([idx1,idx2]));
                ra1 = min(ra([idx1,idx2]));
                ra2 = max(ra([idx1,idx2]));
                dec1 = min(dec([idx1,idx2]));
                dec2 = max(dec([idx1,idx2]));

                my_el(t) = (el2 - el1)/(x2 - x1)*(cur_dmjd - x1) + el2;
                my_az(t) = (az2 - az1)/(x2 - x1)*(cur_dmjd - x1) + az2;
                my_ra(t) = (el2 - el1)/(x2 - x1)*(cur_dmjd - x1) + el2;
                my_dec(t) = (az2 - az1)/(x2 - x1)*(cur_dmjd - x1) + az2;
            end

            ANT.az_off = my_az;
            ANT.el_off = my_el;
            ANT.ra = my_ra;
            ANT.dec = my_dec;
            ANT.dmjd = data_dmjd;
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % OFF Pointing Aggregation
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Apply additional integration
            Ntime = length(off_idxs);
            R_off = zeros(size(Rtmp_off,1), size(Rtmp_off,2), size(Rtmp_off,3), floor(Ntime/4)-1);
            dmjd_off = dmjd_off + dmjd_delay_off;
            corrected_dmjd = (15150/15187.5)*(dmjd_off - dmjd_off(1)) + dmjd_off(1);

            data_dmjd = zeros(floor(Ntime/4)-1, 1);
            Rtmp_off = Rtmp_off(:,:,:,off_idxs);
            for j = 1:floor(Ntime/4)-1
                R_off(:,:,:,j) = sum(Rtmp_off(:,:,:,4*(j-1)+1:4*j),4)/4;
                data_dmjd(j) = sum(corrected_dmjd(4*(j-1)+1:4*j))/4;
            end
    
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Extract antenna positions for scan
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            disp('Extracting antenna positions and computing offsets');

            % Extract offsets
            [ant_dmjd, az_off, el_off, ra, dec] = get_antenna_positions(ant_off_fits_file, use_radec);

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Associate offsets with correlation matrices
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            disp('Associating pointing offsets with correlation matrices');
            Ntime = size(R_off, 4);
            mymin  = zeros(Ntime, 1);
            my_el  = zeros(Ntime, 1);
            my_az  = zeros(Ntime, 1);
            my_ra  = zeros(Ntime, 1);
            my_dec = zeros(Ntime, 1);
            for t = 1:Ntime
                cur_dmjd = data_dmjd(t);
                tmp_dmjd = ant_dmjd;
                [~, idx1] = min(abs(tmp_dmjd - cur_dmjd));

                tmp_dmjd(idx1) = NaN;
                [~, idx2] = min(abs(tmp_dmjd - cur_dmjd));

                x1 = min(ant_dmjd(idx1), ant_dmjd(idx2));
                x2 = max(ant_dmjd(idx1), ant_dmjd(idx2));
                az1 = min(az_off([idx1,idx2]));
                az2 = max(az_off([idx1,idx2]));
                el1 = min(el_off([idx1,idx2]));
                el2 = max(el_off([idx1,idx2]));
                ra1 = min(ra([idx1,idx2]));
                ra2 = max(ra([idx1,idx2]));
                dec1 = min(dec([idx1,idx2]));
                dec2 = max(dec([idx1,idx2]));

                my_el(t) = (el2 - el1)/(x2 - x1)*(cur_dmjd - x1) + el2;
                my_az(t) = (az2 - az1)/(x2 - x1)*(cur_dmjd - x1) + az2;
                my_ra(t) = (el2 - el1)/(x2 - x1)*(cur_dmjd - x1) + el2;
                my_dec(t) = (az2 - az1)/(x2 - x1)*(cur_dmjd - x1) + az2;
            end

            OFF.az_off = my_az;
            OFF.el_off = my_el;
            OFF.ra = my_ra;
            OFF.dec = my_dec;
            OFF.dmjd = data_dmjd;

            save(mat{i}, 'R', 'ANT', 'R_off', 'OFF', 'xid');
        else
            fprintf('Loading %s\n', fits{i});
            load(mat{i});
        end
    end
end
