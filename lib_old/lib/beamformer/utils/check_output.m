% Test beamformer outputs
N_el = 38;
N_bin = 50;
N_time = 40;
N_beam = 7;

N_weights = N_el*N_bin*N_beam;

% Read in beamformer weights
weight_file = fopen('weights.in', 'r');
[weights_f, count] = fread(weight_file, 2*N_weights, 'float');
fclose(weight_file);

weights = weights_f(1:2:end) + 1j*weights_f(2:2:end);
weights = reshape(weights, N_el, N_bin, N_beam);

% Assuming we had all ones from packet generator
data = ones(N_el, N_bin, N_time) + 1j*ones(N_el, N_bin, N_time);

% Assuming we had a single sinusoid in one element
% data = zeros(N_el, N_bin, N_time);
% data(1, 1, :) = 1;

% Output
output = zeros(N_bin, N_beam);
for beam = 1:N_beam
    for bin = 1:N_bin
        for time = 1:N_time
            output(bin, beam) = output(bin, beam) + abs(weights(:, bin, beam)'*data(:, bin, time)).^2;
        end
    end
end
output = output./N_time;

