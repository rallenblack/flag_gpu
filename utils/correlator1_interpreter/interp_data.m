% Read correlations
clearvars;
close all;

FILE = fopen('cor_mcnt_0.out', 'r');
[R, count] = fscanf(FILE, '%g\n');
fclose(FILE);

Nele = 40;
Nele_tot = 64;
Nbin = 25;

Nbaselines_tot = (Nele_tot/2 + 1)*Nele_tot;
Nbaselines     = (Nele + 1)*Nele/2;
Nblocks        = (Nele_tot/2 + 1)*Nele_tot/4;

big = figure();
big2 = figure();
small = figure();
tmp_idx = 1;
tmp = [1 6 11 17 22];

blk_rows = zeros(Nele_tot/2, Nele_tot/2);
for i = 1:Nele_tot/2
    blk_rows(i,1:i) = (i-1)*i/2+1:(i-1)*i/2+i;
end

for Nb = 1:1
    
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
    figure(big);
    %subplot(5,5,Nb);
    imagesc(abs(Rb));
    title(['Bin ', num2str(Nb)]);
    
    figure(big2);
    %subplot(5,5,Nb);
    imagesc(abs(Rb(1:Nele, 1:Nele)));
    title(['Bin ', num2str(Nb)]);
    
    pow1(Nb) = Rb(1,1);
end

figure(small);
plot(pow1);


