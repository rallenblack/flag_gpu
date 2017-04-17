% Simple script to generate CFM beamformer weights for a ULA
% @author Richard Black
% @date 04/17/17

clearvars;
close all;

% Constants
c = 3e8; % speed of propagation (m/s)

% User parameters
theta = 90; % AOA in degrees
Nele = 64; % Number of elements
Nbeams = 200; % Number of beams
Nbins = 500; % Number of frequency bins

% Derived parameters
freqs = (0:Nbins-1)*(303e3) + 1300e6; % All frequencies
d = c/freqs(end); % Element spacing
thetas = linspace(theta-5, theta+5, Nbeams);

% Get steering vectors
a = zeros(Nele, length(freqs), Nbeams);
for b = 1:length(freqs) % Iterate over frequencies
    phi = d*cos(thetas*pi/180)/c*2*pi*freqs(b); % Phase shift
    exponent = kron((1:Nele).', phi);
    a(:,b,:) = exp(1j*exponent); % Steering vector
end

% Plot the beam pattern for one AOA
b_end = length(freqs);
my_a = squeeze(a(:,b_end,:));
gain = zeros(Nbeams, 1);
beam_idx = 90;
for nb = 1:Nbeams
    gain(nb) = abs(my_a(:,nb)'*(my_a(:,beam_idx)*my_a(:,beam_idx)')*my_a(:,nb));
end

figure();
plot(10*log10(gain));

% Get the responses in the AOAs of interest
desired_beam_thetas = [85.5, 87, 88.5, 90, 91.5, 93, 94.5];
d_idx = 1;
indices = zeros(length(desired_beam_thetas), 1);
for d_theta = desired_beam_thetas
    [~,indices(d_idx)] = min(d_theta - thetas);
    d_idx = d_idx + 1;
end

final_w = zeros(Nele, Nbins, 14);
final_w(:,:,1:7) = a(:,:,indices);
final_w(:,:,8:14) = a(:,:,indices);

% Get dimensions in correct order
weights = permute(final_w, [3, 2, 1]);

weight_file(weights, 14, Nbins, Nele);