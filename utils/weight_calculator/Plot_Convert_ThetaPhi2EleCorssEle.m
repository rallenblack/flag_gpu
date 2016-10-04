% Used to convert from theta/phi 2D plot to elevation/cross elevation 2D plot

Ap_MaxG = Sq;
phi_polar = ANT.az_off(idxs)*pi/180; % theta/phi 2D plot range
theta_polar = ANT.el_off(idxs)*pi/180;
% 2D grid; 
% Initialize X/Y mesh

grid_spacing = (theta_polar(2) - theta_polar(1)); %0.1*pi/180
grid_size = ceil(theta_polar(end)/grid_spacing)+1; % X/Y grid number
theta_XY = [];
a1 = ((1:grid_size) - (grid_size+1)/2)*grid_spacing;
[X,Y] = meshgrid(a1,a1);
X = X.';  % fix meshgrid problem
Y = Y.';
rho1 = sqrt(X.^2 + Y.^2);
z1 = 1;
% find relationship between Find the relationship between theta/phi and X/Y
theta1 = atan(rho1/z1);                % theta angles for pattern cut
phi1 = atan2(Y,X);                  
theta_XY = reshape(theta1,grid_size*grid_size,1); % from 0 to theta_max (FOV)
phi_XY   = reshape(phi1,grid_size*grid_size,1);   % from -pi to pi
tic

% Convert theta/phi 2D matrix to X/Y 2D matrix
for p = 1:length(phi_XY)
    for t = 1:length(theta_XY)
        index_p = find(abs(phi_polar - phi_XY(t)) <= 2*pi/length(phi_polar));
        index_t = find(abs(theta_polar - theta_XY(t)) <= max(theta_polar)/length(theta_polar));
        for ip = 1:length(index_p)
            for it = 1:length(index_t)
                Ap_MaxG_average(ip,it) = Ap_MaxG(index_p(ip),index_t(it));
            end
        end
        Ap_MaxG_XY(p,t) = sum(sum(Ap_MaxG_average))/ip/it;
%         if Ap_MaxG_XY(p,t) >= 1;
%             keyboard
%         end
        Ap_MaxG_average = [];
     end
end
toc

linewidth = 2;
fontsize = 14;

Reflector_D = 1;
lambda = 3e8/80e9;
HPBW = 1.014/(Reflector_D/lambda)*180/pi;

figure()
theta_grid = theta_polar*180/pi-theta_polar(end)*180/pi/2;
contourf(theta_grid/HPBW,theta_grid/HPBW,reshape(Ap_MaxG_XY(1,:)*1e4,grid_size,grid_size),'LineWidth',linewidth);
colorbar;
xlabel('Cross Elevation (HPBW)','FontSize',fontsize)
ylabel('Elevation (HPBW)','FontSize',fontsize)
set(gca,'FontSize',fontsize)
caxis([0 70])
xlim([-4.5 4.5])
ylim([-4.5 4.5])
title('Maximum-Sensitivity Beamformer')

