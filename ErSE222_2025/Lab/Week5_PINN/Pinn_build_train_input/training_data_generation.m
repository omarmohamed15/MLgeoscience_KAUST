% This code generates training and testing parameter fields as input for training.
% For the training parameters, a set of data is randomly generated, with values confined within the entire model.

clear 
close all
clc

load('layer_velocity.mat');

v1 = v(1:41,1:41);
v = v1;

n = size(v); 
dx = 25; dz = dx; h  = [dz dx];
z  = [0:n(1)-1]'*h(1)/1000;
x  = [0:n(2)-1]*h(2)/1000;

figure;
pcolor(x,z,v/1000);
shading interp
axis ij
colormap(jet)
h=colorbar;
h.Label.String='Velocity (km/s)';
xlabel('Distance (km)','FontName','Times New Roman','FontSize',12)
ylabel('Depth (km)','FontName','Times New Roman','FontSize',12);
set(gca,'FontSize',18)



n = size(v);                % model size
dx = 0.025; dz = 0.025;     % The grid spacing is 25 m, which is converted to kilometers in practical applications.
nx = n(2); nz = n(1);       % grid size
h  = [dz dx];


%Start defining random points.

N_train = 10000; %% number of the random points

v0 = ones(N_train,1)*1500;             %  velocity of the random points used to calculate the background wavefield 

v = v/1000; v0 = v0/1000;              % The velocity values are uniformly set to km/s.

src_z = 2; 
sz = (src_z-1)*dz;                     %% depth of the source
src_x = fix(nx/2); 
sx = (src_x-1)*dx;                     %% horizontal location of the source



z  = [0:n(1)-1]'*h(1);                 % Define the spatial coordinates of z on a regular grid.
x  = [0:n(2)-1]*h(2);                  % Define the spatial coordinates of x on a regular grid.
[X,Y] = meshgrid(x,z);                 % The grid spacing is 0.025 km, with z arranged along the row direction and x along the column direction.

x1 = [0:1001-1]*0.001;                 % Define the spatial coordinates of x.
z1 = [0:1001-1]'*0.001;                % Define the spatial coordinates of z.
[Xq,Yq] = meshgrid(x1,z1);             % The grid spacing is 0.001 km

v_in = interp2(x,z,v,Xq,Yq);           %% velocity interpolation     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

xx = rand(N_train,1)*1.0 + 0.0;      %% random x coordinate values                       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
zz = rand(N_train,1)*1.0 + 0.0;      %% random z coordinate values                           %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
s_x= ones(N_train,1)*sx;

xx_in = round(xx/0.001)+1;             % calculate grid points                                             %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
zz_in = round(zz/0.001)+1;                                                                     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

v_train = zeros(N_train,1);            % Initialize the selected random training velocity values
U0_imag_train = zeros(N_train,1);      % Initialize the selected random training wavefield values
U0_real_train = zeros(N_train,1);

f = 5.0; %% frequency
%% ANALYTICAL Solution (Background wavefield)
% Distance from source to each point in the model
r = @(zz,xx)(zz.^2+xx.^2).^0.5;       %
% Angular frequency
omega = 1*2*pi*f;
% Wavenumber
vv = 1.5;                            % velocity value
K = (omega./vv);
% x = (0:2501-1)*0.001;
% z = (0:2501-1)'*0.001;
% [zz1,xx1] = ndgrid(z,x);

G_2D_analytic = @(zz,xx)0.25i * besselh(0,2,(K) .* r(zz,xx));

for i = 1:N_train
    
    G_2D = (G_2D_analytic(zz(i) - sz, xx(i) - sx))*7.7;    
    
    v_train(i,1) = v_in(zz_in(i),xx_in(i));      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    U0_real_train(i,1) = real(G_2D);
    U0_imag_train(i,1) = imag(G_2D);

end

m_train = 1./v_train.^2;
m0_train = ([(1./(v0).^2)]);

x_train = xx;                                      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
z_train = zz;                                      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sx_train=s_x;

save layer_5Hz_train_data.mat U0_real_train U0_imag_train x_train z_train sx_train m_train m0_train


%% Numerical results

z  = [0:n(1)-1]'*h(1);
x  = [0:n(2)-1]*h(2);

[zz,xx] = ndgrid(z,x);                      % same as meshgrid

x_star = xx(:);                             % Arrange the data in columns
z_star = zz(:);                             % Arrange the data in columns
sx_star=ones(n(1)*n(2),1)*sx;

npmlz = 60; npmlx = npmlz;
Nz = nz + 2*npmlz;
Nx = nx + 2*npmlx;

v0 = ones(n)*1.500;                         

v_e=extend2d(v,npmlz,npmlx,Nz,Nx);          % Expand the velocity model

Ps1 = getP_H(n,npmlz,npmlx,src_z,src_x);
Ps1 = Ps1'*12000;

[o,d,n] = grid2odn(z,x);
n=[n,1];

nb = [npmlz  npmlx 0];
n  = n + 2*nb;

f = 5; omega = 2*pi*f;
A = Helm2D((omega)./v_e(:),o,d,n,nb);
U  = A\Ps1;

U_2D = reshape(full(U),[nz+2*npmlz,nx+2*npmlx]);
U_2d = U_2D(npmlz+1:end-npmlz,npmlx+1:end-npmlx);      % total wavefield

G_2D = (G_2D_analytic(zz - sz, xx - sx))*7.7;          % background wavefield

G_2D(src_z,src_x) = (G_2D(src_z-1,src_x) + G_2D(src_z+1,src_x) + G_2D(src_z,src_x-1) + G_2D(src_z,src_x+1))/4;

dU_2d = U_2d-G_2D;                                    % Perturb wavefield                      

dU_real_star = real(dU_2d(:));                        
dU_imag_star = imag(dU_2d(:));

dU_real_star(abs(dU_real_star)>2)=0;                  %

save layer_5Hz_test_data.mat x_star z_star sx_star dU_real_star dU_imag_star


% imagesc(reshape(dU_imag_star,101,101))
nx = nz; n = [nz,nx]; 
dx = 25; dz = dx; h  = [dz dx];
z  = [0:n(1)-1]'*h(1)/1000;
x  = [0:n(2)-1]*h(2)/1000;

figure;
pcolor(x,z,(reshape(dU_real_star,nz,nx)));
shading interp
axis ij
colorbar; colormap(jet)
caxis([-0.5 0.5]);
h=colorbar('Ticks',[-0.4 -0.2 0 0.2 0.4]);
h.Label.String='Amplitude';
xlabel('Distance (km)','FontName','Times New Roman','FontSize',12)
ylabel('Depth (km)','FontName','Times New Roman','FontSize',12);
set(gca,'FontSize',18)



'finish!'