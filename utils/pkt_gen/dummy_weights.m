
% Simple script that saves the weights to a RTBF-compatible set of
% files

% Reshape weights
N_beam = 14; % Should be 7, but the number of offsets is wrong (line 56) so the weights read in the CUDA are off.
             % Leave this as it is because it is the simplest case of all
             % ones and the produces a weight vector of all ones.
N_ele = 64;
N_bin_total = 500;
N_bin = 25;
N_pol = 2;

%     w_padded = zeros(N_ele, N_beam, N_pol, N_bin_total);
%     w_padded(X_idx,:,1,:) = wX;
%     w_padded(Y_idx,:,2,:) = wY;

w_padded = ones(N_ele, N_bin_total, N_beam, N_pol);

% Save data into weight file formatted for RTBF code
banks = {'A', 'B', 'C', 'D',...
    'E', 'F', 'G', 'H',...
    'I', 'J', 'K', 'L',...
    'M', 'N', 'O', 'P',...
    'Q', 'R', 'S', 'T'};

interleaved_w = zeros(2*N_ele*N_bin*N_beam*N_pol,1);
chan_idx = [1:5, 101:105, 201:205, 301:305, 401:405];

for b = 1:length(banks)
    % Get bank name
    bank_name = banks{b};
    
    % Extract channels for bank
    % w1 = w_padded(:,:,:,chan_idx+5*(b-1));
    w1 = w_padded(:,chan_idx+5*(b-1),:,:);
    
    %%% Just for testing - Used .mat file to verify that the .bin file was
    %%% the same
    % wp_filename = strrep(filename, '.bin', sprintf('_%s.mat',bank_name));
    % save(wp_filename, 'w1');
    
    % Reshape for file format
    w2 = reshape(w1, N_ele*N_bin, N_beam*N_pol);
    w_real = real(w2(:));
    w_imag = imag(w2(:));
    interleaved_w(1:2:end) = w_real(:);
    interleaved_w(2:2:end) = w_imag(:);
    
    % Get filename
%     filename = '/lustre/projects/flag/weight_files/dummy_w.bin';
    filename = '/home/groups/flag/weight_files/dummy_w.bin';
    weight_file = strrep(filename, '.', sprintf('_%s.', bank_name));
    
    % Create metadata for weight file
    offsets_el = 0.7;
    offsets_az = 0.7;
    offsets =  0.7*ones(N_beam,1); %[offsets_el; offsets_az];
    offsets = offsets(:);
    cal_filename = 'test';
    to_skip1 = 64 - length(cal_filename);
    algorithm_name = 'Max Signal-to-Noise Ratio';
    to_skip2 = 64 - length(algorithm_name);
    xid = b-1;
    
    % Write to binary file
    WID = fopen(weight_file,'wb');
    if WID == -1
        error('Author:Function:OpenFile', 'Cannot open file: %s', weight_file);
    end
    
    % Write payload
    fwrite(WID, single(interleaved_w), 'single');
    
    % Write metadata
    fwrite(WID,single(offsets),'float');
    fwrite(WID,cal_filename, 'char*1');
    if to_skip1 > 0
        fwrite(WID, char(zeros(1,to_skip1)));
    end
    fwrite(WID,algorithm_name, 'char*1');
    if to_skip2 > 0
        fwrite(WID, char(zeros(1,to_skip2)));
    end
    fwrite(WID, uint64(xid), 'uint64');
    fclose(WID);
    
    % fprintf('Saved to %s\n', weight_file);
end
