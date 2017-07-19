close all; clear all;
w = zeros(40,1);
y = zeros(25,40);

% Tone at 1367.3MHz in Bank C
% [R_on_C,dmjd1,xid_c] = extract_covariances('/lustre/gbtdata/TGBT16A_508_01/TMP/BF/2016_10_07_14:37:11C.fits');
% [R_on_D,dmjd2,xid_d] = extract_covariances('/lustre/gbtdata/TGBT16A_508_01/TMP/BF/2016_10_07_14:37:11D.fits');
% 
% % No tone
% [R_off_C,dmjd3,xid_c] = extract_covariances('/lustre/gbtdata/TGBT16A_508_01/TMP/BF/2016_10_07_15:01:14C.fits');
% [R_off_D,dmjd4,xid_d] = extract_covariances('/lustre/gbtdata/TGBT16A_508_01/TMP/BF/2016_10_07_15:01:14D.fits');

% % Tone at 1367.3MHz in Bank C
% % [R_on_C,dmjd1,xid_c] = extract_covariances('/lustre/gbtdata/TGBT16A_508_01/TMP/BF/2016_10_07_15:49:39C.fits');
% % [R_off_C,dmjd2,xid_c] = extract_covariances('/lustre/gbtdata/TGBT16A_508_01/TMP/BF/2016_10_07_15:38:36C.fits');
% % [R_on_D,dmjd3,xid_d] = extract_covariances('/lustre/gbtdata/TGBT16A_508_01/TMP/BF/2016_10_07_15:49:39D.fits');
% % [R_off_D,dmjd4,xid_d] = extract_covariances('/lustre/gbtdata/TGBT16A_508_01/TMP/BF/2016_10_07_15:38:36D.fits');
% 
load('Scan1_Otf_Hot_C.mat');
R_on_C = hot_c;
load('Scan1_Otf_Hot_D.mat');
R_on_D = hot_d;
load('Scan2_Otf_Cold_C.mat');
R_off_C = cold_c;
load('Scan2_Otf_Cold_D.mat');
R_off_D = cold_d;
xid_c = 1;
xid_d = 2;
% 
% % Bank C
% Rprime = zeros(40,40,25,1138);
% % Rprime = zeros(40,40,25);
% for i = 1:25
%     for j = 1:40
%         w = zeros(40,1);
%         w(j) = 1;
%         %         hot = w(1:40)'*mean(R_on(:,:,i,:),4)*w(1:40);
%         %         cold = w(1:40)'*mean(R_off(:,:,i,:),4)*w(1:40);
%         %         hot = R_on_C(j,j,i); %mean(R_on_C(j,j,i,:),4);
%         %         cold = R_off_C(j,j,i); %mean(R_off_C(j,j,i,:),4);
%         hot = mean(R_on_C(j,j,i,:),4);
%         cold = mean(R_off_C(j,j,i,:),4);
%         y(i,j) = hot/cold;
%     end
%     %     w = ones(40,1);
%     %     w([3,20,30,40]) = 0;
%     %     w =mean(R_off(:,:,i,:),4)\w;
%     %     hot = w(1:40)'*mean(R_on(:,:,i,:),4)*w(1:40);
%     %     cold = w(1:40)'*mean(R_off(:,:,i,:),4)*w(1:40);
%     %     y2(i) = hot/cold;
% end
% 
% for i = 1:25
%     for j = 1:40
%         R_on_C(3,3,:,:) = Rprime(3,3,:,:);
%         R_on_C(20,20,:,:) = Rprime(20,20,:,:);
%         R_on_C(30,30,:,:) = Rprime(30,30,:,:);
%         R_on_C(40,40,:,:) = Rprime(40,40,:,:);
%         hot_c = mean(R_on_C(j,j,i,:),4);
%         cold_c = mean(R_off_C(j,j,i,:),4);
%         y2(i) = hot_c/cold_c;
%         %         R_on_C(3,3,:) = Rprime(3,3,:);
%         %         R_on_C(20,20,:) = Rprime(20,20,:);
%         %         R_on_C(30,30,:) = Rprime(30,30,:);
%         %         R_on_C(40,40,:) = Rprime(40,40,:);
%         %         hot_c = R_on_C(j,j,i);
%         %         cold_c = R_off_C(j,j,i);
%         y2(i) = hot_c/cold_c;
%     end
% end
% 
% 
% 
% % Bank D
% Rprime = zeros(40,40,25,1138);
% % Rprime = zeros(40,40,25);
% for i = 1:25
%     for j = 1:40
%         w = zeros(40,1);
%         w(j) = 1;
%         hot = w(1:40)'*mean(R_on_D(:,:,i,:),4)*w(1:40);
%         cold = w(1:40)'*mean(R_off_D(:,:,i,:),4)*w(1:40);
%         %         hot_d = R_on_D(j,j,i); %mean(R_on_D(j,j,i,:),4);
%         %         cold_d = R_off_D(j,j,i); %mean(R_off_D(j,j,i,:),4);
%         y_d(i,j) = hot/cold;
%     end
% end
% 
% for i = 1:25
%     for j = 1:40
%         R_on_D(3,3,:,:) = Rprime(3,3,:,:);
%         R_on_D(20,20,:,:) = Rprime(20,20,:,:);
%         R_on_D(30,30,:,:) = Rprime(30,30,:,:);
%         R_on_D(40,40,:,:) = Rprime(40,40,:,:);
%         hot_d = mean(R_on_D(j,j,i,:),4);
%         cold_d = mean(R_off_D(j,j,i,:),4);
%         y2_d(i) = hot_d/cold_d;
%         %         R_on_D(3,3,:) = Rprime(3,3,:);
%         %         R_on_D(20,20,:) = Rprime(20,20,:);
%         %         R_on_D(30,30,:) = Rprime(30,30,:);
%         %         R_on_D(40,40,:) = Rprime(40,40,:);
%         %         hot_d = R_on_D(j,j,i);
%         %         cold_d = R_off_D(j,j,i);
%         %         y2_d(i) = hot_d/cold_d;
%     end
% end
% % % Tone at 1367.3MHz in Bank C
% % [R_on_C,dmjd1,xid_c] = extract_covariances('/lustre/gbtdata/TGBT16A_508_01/TMP/BF/2016_10_07_15:49:39C.fits');
% % [R_off_C,dmjd2,xid_c] = extract_covariances('/lustre/gbtdata/TGBT16A_508_01/TMP/BF/2016_10_07_15:38:36C.fits');
% % [R_on_D,dmjd3,xid_d] = extract_covariances('/lustre/gbtdata/TGBT16A_508_01/TMP/BF/2016_10_07_15:49:39D.fits');
% % [R_off_D,dmjd4,xid_d] = extract_covariances('/lustre/gbtdata/TGBT16A_508_01/TMP/BF/2016_10_07_15:38:36D.fits');
% 
% % load('Scan1_Hot_1367MHz_C.mat');
% % R_on_C = hot_c;
% % load('Scan1_Hot_1367MHz_D.mat');
% % R_on_D = hot_d;
% % load('Scan2_Cold_NoTone_C.mat');
% % R_off_C = cold_c;
% % load('Scan2_Cold_NoTone_D.mat');
% % R_off_D = cold_d;
% % xid_c = 1;
% % xid_d = 2;
% 
% % Bank C
% Rprime = zeros(40,40,25,1138);
% % Rprime = zeros(40,40,25);
% for i = 1:25
%     for j = 1:40
%         w = zeros(40,1);
%         w(j) = 1;
%         %         hot = w(1:40)'*mean(R_on(:,:,i,:),4)*w(1:40);
%         %         cold = w(1:40)'*mean(R_off(:,:,i,:),4)*w(1:40);
%         %         hot = R_on_C(j,j,i); %mean(R_on_C(j,j,i,:),4);
%         %         cold = R_off_C(j,j,i); %mean(R_off_C(j,j,i,:),4);
%         hot = mean(R_on_C(j,j,i,:),4);
%         cold = mean(R_off_C(j,j,i,:),4);
%         y(i,j) = hot/cold;
%     end
%     %     w = ones(40,1);
%     %     w([3,20,30,40]) = 0;
%     %     w =mean(R_off(:,:,i,:),4)\w;
%     %     hot = w(1:40)'*mean(R_on(:,:,i,:),4)*w(1:40);
%     %     cold = w(1:40)'*mean(R_off(:,:,i,:),4)*w(1:40);
%     %     y2(i) = hot/cold;
% end
% 
% for i = 1:25
%     for j = 1:40
%         R_on_C(3,3,:,:) = Rprime(3,3,:,:);
%         R_on_C(20,20,:,:) = Rprime(20,20,:,:);
%         R_on_C(30,30,:,:) = Rprime(30,30,:,:);
%         R_on_C(40,40,:,:) = Rprime(40,40,:,:);
%         hot_c = mean(R_on_C(j,j,i,:),4);
%         cold_c = mean(R_off_C(j,j,i,:),4);
%         y2(i) = hot_c/cold_c;
%         %         R_on_C(3,3,:) = Rprime(3,3,:);
%         %         R_on_C(20,20,:) = Rprime(20,20,:);
%         %         R_on_C(30,30,:) = Rprime(30,30,:);
%         %         R_on_C(40,40,:) = Rprime(40,40,:);
%         %         hot_c = R_on_C(j,j,i);
%         %         cold_c = R_off_C(j,j,i);
%         y2(i) = hot_c/cold_c;
%     end
% end
% 
% 
% 
% % Bank D
% Rprime = zeros(40,40,25,1138);
% % Rprime = zeros(40,40,25);
% for i = 1:25
%     for j = 1:40
%         w = zeros(40,1);
%         w(j) = 1;
%         hot = w(1:40)'*mean(R_on_D(:,:,i,:),4)*w(1:40);
%         cold = w(1:40)'*mean(R_off_D(:,:,i,:),4)*w(1:40);
%         %         hot_d = R_on_D(j,j,i); %mean(R_on_D(j,j,i,:),4);
%         %         cold_d = R_off_D(j,j,i); %mean(R_off_D(j,j,i,:),4);
%         y_d(i,j) = hot/cold;
%     end
% end
% 
% for i = 1:25
%     for j = 1:40
%         R_on_D(3,3,:,:) = Rprime(3,3,:,:);
%         R_on_D(20,20,:,:) = Rprime(20,20,:,:);
%         R_on_D(30,30,:,:) = Rprime(30,30,:,:);
%         R_on_D(40,40,:,:) = Rprime(40,40,:,:);
%         hot_d = mean(R_on_D(j,j,i,:),4);
%         cold_d = mean(R_off_D(j,j,i,:),4);
%         y2_d(i) = hot_d/cold_d;
%         %         R_on_D(3,3,:) = Rprime(3,3,:);
%         %         R_on_D(20,20,:) = Rprime(20,20,:);
%         %         R_on_D(30,30,:) = Rprime(30,30,:);
%         %         R_on_D(40,40,:) = Rprime(40,40,:);
%         %         hot_d = R_on_D(j,j,i);
%         %         cold_d = R_off_D(j,j,i);
%         %         y2_d(i) = hot_d/cold_d;
%     end
% end
% 
% % figure(1);
% % plot(10*log10(abs(y2)));
% % title('Y-factor');
% % ylabel('Power');
% % xlabel('Frequency bins');
% 
% figure(1);
% imagesc(10*log10(abs(y)));
% colorbar;
% title('Y-factor for XID 1');
% xlabel('Elements');
% ylabel('Frequency bins');
% 
% figure(2);
% imagesc(10*log10(abs(y_d)));
% colorbar;
% title('Y-factor for XID 2');
% xlabel('Elements');
% ylabel('Frequency bins');

% % figure(1);
% % plot(10*log10(abs(y2)));
% % title('Y-factor');
% % ylabel('Power');
% % xlabel('Frequency bins');
% 
% figure(1);
% imagesc(10*log10(abs(y)));
% colorbar;
% title('Y-factor for XID 1');
% xlabel('Elements');
% ylabel('Frequency bins');
% 
% figure(2);
% imagesc(10*log10(abs(y_d)));
% colorbar;
% title('Y-factor for XID 2');
% xlabel('Elements');
% ylabel('Frequency bins');

figure(3);
imagesc(10*log10(abs(squeeze(mean(R_on_C(:,:,6,1),4)))));
colorbar;
title('Single frequency bin Hot-covariance');
xlabel('Elements');
ylabel('Elements');

figure(4);
imagesc(10*log10(abs(squeeze(mean(R_off_C(:,:,6,1),4)))));
colorbar;
title('Single frequency bin Cold-covariance');
xlabel('Elements');
ylabel('Elements');


bank_on = zeros(500,1);
f_idx = [1:5, 101:105, 201:205, 301:305, 401:405];
bank_on(f_idx + xid_c*5,1) = 10*log10(abs(squeeze(mean(R_on_C(4,4,:,1),4))));
bank_on(f_idx + xid_d*5,1) = 10*log10(abs(squeeze(mean(R_on_D(4,4,:,1),4))));

bank_off = zeros(500,1);
bank_off(f_idx + xid_c*5,1) = 10*log10(abs(squeeze(mean(R_off_C(4,4,:,1),4))));
bank_off(f_idx + xid_d*5,1) = 10*log10(abs(squeeze(mean(R_off_D(4,4,:,1),4))));



figure(5);
plot(bank_on, 'r');
hold on;
plot(bank_off,'b');
title('Test tone PSD');
xlabel('Frequency bins');
ylabel('Power');
legend('Hot','Cold');
hold off;

