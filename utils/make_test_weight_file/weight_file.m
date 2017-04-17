function weight_file(weights,n_beams,F,M)
    for nb = 1:n_beams
        weights_vec_C = reshape(squeeze(weights(nb,:,:)).', F*M, 1); % Conjugate of the weights vector.

        % Interleave real and imaginary components (real then imaginary after)
        weights_C_real = real(weights_vec_C);
        weights_C_imag = imag(weights_vec_C);

        interleaved_w = zeros(2*F*M,1);
        interleaved_w(1:2:end) = weights_C_real;
        interleaved_w(2:2:end) = weights_C_imag;
        weights_H(:,nb) = interleaved_w;
    end

    % Create metadata for weight file
    offsets_el =  single([0,  0,    1,   1, 0,  -1,   -1]*2);
    offsets_xel = single([0, -1, -0.5, 0.5, 1, 0.5, -0.5]*2);
    offsets = [offsets_el; offsets_xel];
    offsets = offsets(:);
    cal_filename = '2016_06_13_16:58:04A.fits';
    to_skip1 = 64 - length(cal_filename);
    algorithm_name = 'Max Signal-to-Noise Ratio';
    to_skip2 = 64 - length(algorithm_name);
    xid = 3;

    % Write to binary file
    FID = fopen('weights_vec_C.bin','w');
    % Write payload
    fwrite(FID,single(weights_H(:)),'float');
    % Write metadata
    fwrite(FID,single(offsets),'float');
    fwrite(FID,cal_filename, 'char*1');
    if to_skip1 > 0
        fwrite(FID, char(zeros(1,to_skip1)));
    end
    fwrite(FID,algorithm_name, 'char*1');
    if to_skip2 > 0
        fwrite(FID, char(zeros(1,to_skip2)));
    end
    fwrite(FID, uint64(xid), 'uint64');
    fclose(FID);

end
