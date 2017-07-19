data_root = '/lustre/projects/flag';
meta_root = '/home/gbtdata';
proj_id = 'AGBT16B_400_05';


b = extract_b_output('/lustre/gbtdata/TGBT16A_508_01/TMP/BF/2017_05_03_23:17:56A.fits');

N_bin = 25;
N_pol = 4; % Order: XX*, YY*, real(XY*), and imag(XY*).
idx = 0;
for i = 1:1 % N_bin
    for k = 1:1 % Order: XX*, YY*, real(XY*), and imag(XY*).
        idx = idx+1;
        figure(idx);
        imagesc(squeeze(10*log10(abs(b(:,k,i,:)))).'); % Order: beam, pol, bin, and sti.
        if k == 1
            title(['X Frequency bin ' num2str(i)]);
        elseif k == 2
            title(['Y Frequency bin ' num2str(i)]);
        elseif k == 3
            title(['Real XY Frequency bin ' num2str(i)]);
        elseif k == 4
            title(['Imaginary XY Frequency bin ' num2str(i)]);
        end
        xlabel('Beam Index');
        ylabel('STI Index');
    end
end