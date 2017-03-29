% Read correlations
clearvars;
close all;

% White noise
% filename = '/lustre/pulsar/users/rprestag/FLAG/JUNK/JUNK/BF/2016_07_25_02:45:31A.fits';

% Correlated data
% filename = '/lustre/pulsar/users/rprestag/FLAG/JUNK/JUNK/BF/2016_07_25_02:37:42A.fits';

% Grid data
% filename = '/lustre/gbtdata/TGBT16A_508_01/BF/2016_07_26_23:45:30A.fits';

% info = fitsinfo(filename);
% data = fitsread(filename, 'binarytable', 1);
% tmp = data{3};
% R = zeros(2*length(tmp), 1);
% R(1:2:end) = real(tmp(1,:));
% R(2:2:end) = imag(tmp(1,:));

Nele = 40;
Nele_tot = 64;
Nbin = 160;
Nsamp = 125;
Nbaselines_tot = (Nele_tot/2 + 1)*Nele_tot;
Nbaselines     = (Nele + 1)*Nele/2;
Nblocks        = (Nele_tot/2 + 1)*Nele_tot/4;

%big = figure();
%big2 = figure();
%small = figure();
tmp_idx = 1;
tmp = [1 6 11 17 22];

blk_rows = zeros(Nele_tot/2, Nele_tot/2);
for i = 1:Nele_tot/2
    blk_rows(i,1:i) = (i-1)*i/2+1:(i-1)*i/2+i;
end

Rtot = zeros(Nele_tot, Nele_tot, Nbin);
PATH = '/home/mburnett/dibas/lib/python/';
mcnt = [0]%, 200, 400, 600];
%for mcnt = 0:2:198
for k = 1:length(mcnt)
    FILE = fopen([PATH, sprintf('cor_mcnt_%d.out', mcnt(k))], 'r');
    [R, count] = fscanf(FILE, '%g\n');
    fclose(FILE);

    for Nb = 1:Nbin

        rb_real = R(2*Nbaselines_tot*(Nb - 1)+1:2:2*Nbaselines_tot*Nb);
        rb_imag = R(2*Nbaselines_tot*(Nb - 1)+2:2:2*Nbaselines_tot*Nb);
        rb = rb_real + 1j*rb_imag;

        Rb = zeros(Nele_tot, Nele_tot);
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

        Rtot(:,:,Nb) = Rtot(:,:,Nb) + Rb.*Nsamp;

        fig_mod = ceil(Nb/40);
        fig_mod_plot = mod(Nb,40);
        figure(fig_mod);
        if fig_mod_plot == 0
            fig_mod_plot = 40;
        end
        subplot(8,5,fig_mod_plot);
        imagesc(abs(Rtot(1:Nele, 1:Nele, Nb)));
        title(['Bin ', num2str(Nb)]);
        drawnow;
    end
end


