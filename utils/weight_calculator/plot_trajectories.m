% Plots trajectory of telescope on-axis pointing
clearvars;
close all;

global_params;

% Specify output MAT files
mat = sprintf('%s/%s%s.mat', dir, stamp, bank);
load(mat);

myfig = figure();
plot(ANT.az_off(idxs), ANT.el_off(idxs), '-b',...
     ANT.az_off(1), ANT.el_off(1), 'rx');
xlabel('Azimuth Offset (degrees)');
ylabel('Elevation Offset (degrees)');
title('Scan Trajectory');
grid on;
filename = sprintf('figures/trajectory_%s.fig', stamp);
savefig(myfig, filename);

myfig = figure();
plot(ANT.ra, ANT.dec, '-b',...
     ANT.ra(1), ANT.dec(1), 'rx');
xlabel('Right Ascension (degrees)');
ylabel('Declination (degrees)');
title('Scan Trajectory');
grid on;
filename = sprintf('figures/trajectory_radec_%s.fig', stamp);
savefig(myfig, filename);
