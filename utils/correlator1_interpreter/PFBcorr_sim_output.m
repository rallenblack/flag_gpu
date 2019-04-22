% Read correlations
clearvars;
%close all;

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
Nbaselines_tot = (Nele_tot/2 + 1)*Nele_tot; % Upper or lower triangular elements plus the diagonals.
Nbaselines     = (Nele + 1)*Nele/2;
Nblocks        = (Nele_tot/2 + 1)*Nele_tot/4;
Nfft = 32;

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
PATH = '/lustre/flag/';
mcnt = [0]; %, 200, 400, 600];
%for mcnt = 0:2:198
% for k = 1:length(mcnt)
%     disp(['Processing mcnt=', num2str(mcnt(k))]);
%     FILE = fopen([PATH, sprintf('cor_mcnt_%d_B.out', mcnt(k))], 'r');
%     [R, count] = fscanf(FILE, '%g\n');
%     fclose(FILE);
% 
%     for Nb = 1:Nbin
% 
%         rb_real = R(2*Nbaselines_tot*(Nb - 1)+1:2:2*Nbaselines_tot*Nb);
%         rb_imag = R(2*Nbaselines_tot*(Nb - 1)+2:2:2*Nbaselines_tot*Nb);
%         rb = rb_real + 1j*rb_imag;
% 
%         Rb = zeros(Nele_tot, Nele_tot);
%         for Nblk = 1:Nblocks
%             block_r = rb(4*(Nblk-1)+1:4*Nblk);
%             [row, col] = find(blk_rows == Nblk);
%             Rb(2*row - 1, 2*col - 1) = block_r(1);
%             if sum(diag(blk_rows) == Nblk) == 0
%                 Rb(2*row - 1, 2*col) = block_r(2);
%             end
%             Rb(2*row    , 2*col - 1) = block_r(3);
%             Rb(2*row    , 2*col    ) = block_r(4);
%         end
% 
%         Rb = Rb + (Rb' - diag(diag(Rb'))); % Exploit symmetry
%         Rb = Rb./Nsamp;
% 
%         Rtot(:,:,Nb) = Rtot(:,:,Nb) + Rb.*Nsamp;
%         
%         % Commented out in recent file modified by mitch %%%%%%%%%%%%%
%         fig_mod = ceil(Nb/40);
%         fig_mod_plot = mod(Nb,40);
%         figure(fig_mod);
%         if fig_mod_plot == 0
%             fig_mod_plot = 40;
%         end
%         subplot(8,5,fig_mod_plot);
%         imagesc(abs(Rtot(1:Nele, 1:Nele, Nb)));
% %         imagesc(abs(Rtot(:, :, Nb)));
%         title(['Bin ', num2str(Nb)]);
%         drawnow;
%         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     end
% end

coarseBin = 5;
Rtot1 = zeros(Nele_tot, Nele_tot, coarseBin);
for k = 1:length(mcnt)
    disp(['Processing mcnt=', num2str(mcnt(k))]);
    FILE = fopen([PATH, sprintf('cor_mcnt_%d_B.out', mcnt(k))], 'r');
    [R, count] = fscanf(FILE, '%g\n');
    fclose(FILE);

    for bb = 1:coarseBin
        Rtmp = zeros(Nele_tot, Nele_tot);
        for Nb = (0:Nfft-1)*coarseBin+bb
            
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
            
            Rtmp = Rtmp + Rb.*Nsamp;
%             Rtot(:,:,Nb) = Rtot(:,:,Nb) + Rb.*Nsamp;
            
        end
        Rtot1(:,:,bb) = Rtmp;
        figure(1);
        subplot(5,5,bb);
        imagesc(abs(Rtot1(1:Nele, 1:Nele, bb)));
%         imagesc(abs(Rtot(:, :, Nb)));
        title(['Bin ', num2str(bb)]);
        drawnow;
    end
end

% % Added from more recent file to plot PFB output
% idx = 1:160;
% idx1 = reshape(idx, [5,32]);
% idx2 = idx1';
% stitch_idx = reshape(idx2, [160,1]);
% 
% % stitch_idx = [];
% % for kk = 0:31
% %     stitch_idx = [stitch_idx, kk:32:kk+128];
% % end
% % stitch_idx = stitch_idx + 1;
% % tmp2 = squeeze(Rtot(1,1,stitch_idx));
% % 
% % mat_idx = [];
% % for jj = 0:4
% %     mat_idx = [mat_idx, (jj:32:jj+128)'];
% % end
% % stitch_idx = [mat_idx; mat_idx+4; mat_idx+8; mat_idx+12; mat_idx+16; mat_idx+20; mat_idx+24] + 1;
% % keyboard;
% % tmp3 = squeeze(Rtot(1,1,stitch_idx));
% 
% tmp2 = squeeze(Rtot(18,18,stitch_idx));
% figure(11);
% plot(0:length(tmp2)-1, 10*log10(abs(tmp2))); grid on;
% tmp = squeeze(Rtot(18,18,:));
% figure(10);
% plot(0:length(tmp)-1, 10*log10(abs(tmp))); grid on;

