% DMJD delay analysis


% Aggregates antenna positions with covariance matrices

close all;
clearvars;

global_params;

% Specify FITS file
fits = sprintf('%s/%sA.fits', dir, stamp);
fits_off = sprintf('%s/%sA.fits', dir, off_stamp);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reconstruct covariance matrices from FITS file
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if exist(fits, 'file')
    fprintf('Extracting covariances for %s\n', fits);
    [Rtmp, act_dmjd, xid] = extract_covariances(fits);
    if strcmp(fits, fits_off) ~= 1
        fprintf('Extracting off covariances for %s\n', fits);
        [Rtmp_off, dmjd_off, xid_off] = extract_covariances(fits_off);
    else
        fprintf('Off pointing file same as on\n');
        Rtmp_off = Rtmp;
        dmjd_off = act_dmjd;
        xid_off = xid;
    end

    for delay = 2.8:0.005:2.9
        dmjd_delay = delay/(24*3600);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % ON Pointing Aggregation
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Apply additional integration
        Ntime = size(Rtmp,4);
        R = zeros(size(Rtmp,1), size(Rtmp,2), size(Rtmp,3), floor(Ntime/4)-1);
        dmjd = act_dmjd + dmjd_delay;
        corrected_dmjd = (15150/15187.5)*(dmjd - dmjd(1)) + dmjd(1);

        data_dmjd = zeros(floor(Ntime/4)-1,1);
        for j = 1:floor(Ntime/4)-1
            R(:,:,:,j) = sum(Rtmp(:,:,:,4*(j-1)+1:4*j),4)/4;
            data_dmjd(j) = sum(corrected_dmjd(4*(j-1)+1:4*j))/4;
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % OFF Pointing Aggregation
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Apply additional integration
        Ntime = length(off_idxs);
        R_off = zeros(size(Rtmp_off,1), size(Rtmp_off,2), size(Rtmp_off,3), floor(Ntime/4)-1);

        Rtmpoff = Rtmp_off(:,:,:,off_idxs);
        for j = 1:floor(Ntime/4)-1
            R_off(:,:,:,j) = sum(Rtmpoff(:,:,:,4*(j-1)+1:4*j),4)/4;
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
        
        Ntime = size(R,4);
        Nbin = size(R,3);
        
        % Look at all Y-Pol elements
        idx = y_idx;
        [v,~] = get_grid_weights(R, R_off, dir, stamp, bank, idx, overwrite);

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Compute Element Beampatterns
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        disp('Computing element beampattern maps');
        pattern = zeros(length(idx), Ntime, Nbin);
        for b = 1:Nbin
            for t = 1:Ntime
                for e = 1:length(idx)
                    w = zeros(length(idx), 1);
                    w(e) = 1;
                    pattern(e,t,b) = real(abs(w'*v(idx,b,t)).^2);
                end
            end
        end


        % Convert frequency bin indices to sky frequencies
        xid_bins = [1:5, 101:105, 201:205, 301:305, 401:405] + 5*xid;
        bin_width = 303.75e3;
        bin_idx = -249:250;
        freqs = bin_idx*bin_width + center_freq;
        xid_freqs = freqs(xid_bins);

        % Interpolated map
        Npoints = 80;
        %if use_radec
        %    minX = max(min(ANT.az_off(idxs)), -0.45);
        %    maxX = min(max(ANT.az_off(idxs)), 0.45);
        %    minY = max(min(ANT.el_off(idxs)), -0.35);
        %    maxY = min(max(ANT.el_off(idxs)), 0.35);
        %else
            minX = min(ANT.az_off(idxs));
            maxX = max(ANT.az_off(idxs));
            minY = min(ANT.el_off(idxs));
            maxY = max(ANT.el_off(idxs));
        %end
        xval = linspace(minX, maxX, Npoints);
        yval = linspace(minY, maxY, Npoints);
        [X,Y] = meshgrid(linspace(minX,maxX,Npoints), linspace(minY,maxY,Npoints));

        e = 1;
        b = 11;
        beam_fig = figure();
        Pq = griddata(ANT.az_off(idxs), ANT.el_off(idxs), pattern(e,idxs,b).', X, Y);
        imagesc(xval, yval, 10*log10(abs(Pq))); colorbar; hold on;
        contour(gca, xval, yval, 10*log10(abs(Pq)), [-3 -3], '-k'); hold off;
        set(gca, 'ydir', 'normal');
        title(sprintf('1Y - Delay %g', delay));
        colormap('jet');
        drawnow;
    end
end
