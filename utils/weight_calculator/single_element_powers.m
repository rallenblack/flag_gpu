%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Extract single-element powers
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clearvars;
close all;

global_params;

mat = sprintf('%s/%s%s.mat', dir, stamp, bank);
load(mat);

Ntime = size(R,4);

b = 11;
for i = 1:19
    x_pow = real(squeeze(R(x_idx(i), x_idx(i), b, 2:end-1)));
    y_pow = real(squeeze(R(y_idx(i), y_idx(i), b, 2:end-1)));
    figure(1); subplot(4,5,i);
    plot(2:Ntime-1, x_pow);
    title(sprintf('%dX', i));
    figure(2); subplot(4,5,i);
    plot(2:Ntime-1, y_pow);
    title(sprintf('%dY', i));
    drawnow;
end
