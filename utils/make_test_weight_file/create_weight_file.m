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
Nele_real = 40; % Real number of elements
Nbeams = 400; % Number of beams
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
    gain(nb) = abs(my_a(1:Nele_real,nb)'*(my_a(1:Nele_real,beam_idx)*my_a(1:Nele_real,beam_idx)')*my_a(1:Nele_real,nb));
end

figure();
plot(10*log10(gain));

% Get the responses in the AOAs of interest
% desired_beam_thetas = [85.5, 87, 88.5, 90, 91.5, 93, 94.5];
desired_beam_thetas = [90.05, 87, 88.5, 90, 91.5, 93, 89.95];
d_idx = 1;
indices = zeros(length(desired_beam_thetas), 1);
for d_theta = desired_beam_thetas
    [~,indices(d_idx)] = min(abs(d_theta - thetas));
    d_idx = d_idx + 1;
end


% Plot the overlaid responses just for fun
my_window = chebwin(Nele_real, 20);
beams = zeros(Nbeams, length(indices));
new_w = zeros(Nele_real, Nbins, length(indices));
for t_idx = 1:length(indices)
    for nb = 1:Nbeams
        for nf = 1:Nbins
            new_w(:,nf, t_idx) = my_window.*a(1:Nele_real, nf, indices(t_idx));
        end
        beams(nb,t_idx) = abs(new_w(:,end,t_idx)'*my_a(1:Nele_real,nb))^2;
    end
end
figure();
plot(thetas, 10*log10(beams));

final_w = zeros(Nele, Nbins, 14);
final_w(1:Nele_real,:,1:7) = new_w;
final_w(1:Nele_real,:,8:14) = new_w;

% Get dimensions in correct order
weights = permute(final_w, [3, 2, 1]);

% Select out frequencies of interest
xid = 12; % zero-indexed
freq_idxs = xid*5 + [1:5, 101:105, 201:205, 301:305, 401:405];
final_weights = weights(:,freq_idxs,:);

weight_file(final_weights, 14, 25, Nele);

% Open weight file and confirm
FID = fopen('weights.in','rb');
% First 64x25x2 floats are beam X0
% Next are beam X1...X7
% Them beams Y0...Y7
% Internally, each beam has ordering (fast->slow), re/im, elements, frequency
X = zeros(64, 25, 7);
Y = zeros(64, 25, 7);
for nb = 1:7
    Xtmp = fread(FID, 25*64*2, 'single');
    Xtmp = Xtmp(1:2:end) + 1j*Xtmp(2:2:end);
    X(:,:,nb) = reshape(Xtmp, 64, 25);
end
for nb = 1:7
    Ytmp = fread(FID, 25*64*2, 'single');
    Ytmp = Ytmp(1:2:end) + 1j*Ytmp(2:2:end);
    Y(:,:,nb) = reshape(Ytmp, 64, 25);
end

% Plot beam patterns
beams = zeros(Nbeams, length(indices));
for t_idx = 1:7
    for nb = 1:Nbeams
        beams(nb,t_idx) = abs(X(:,end,t_idx)'*my_a(1:Nele,nb))^2;
    end
end
figure();
plot(thetas, 10*log10(beams));