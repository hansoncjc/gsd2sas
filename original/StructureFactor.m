function [S_3,q_3_x,q_3_y,q_3_z,S_2,q_2_para_grid,q_2_perp_grid,S_1,q_1_bin] = StructureFactor(x, box, N_grid)

% Computes the static structure factor for a snapshot of particles
%
% INPUTS
% x: (N-by-3) particle positions
% box: (N-by-1) dimensions of the simulation box
% N_grid = (scalar) number of grid points in the smallest dimension
%
% OUTPUTS
% S_3: (Ng-by-Ng-by-Ng) static structure factor as a function of 3D wavevector
% q_3_x: (Ng-by-Ng-by-Ng) x component of the wavevector
% q_3_y: (Ng-by-Ng-by-Ng array) y component of the wavevector
% q_3_z: (Ng-by-Ng-by-Ng array) z component of the wavevector
% S_2: (Nperp-by-Npara) S as a function of 2D wavevector
% q_2_para_grid: (Npara-by-Nperp) component of q parallel to x-direction
% q_2_perp_grid: (Npara-by-Nperp) component of q perpendicular x-direction
% S_1: (Nmag-by-1) S as a function of 1D wavevector magnitude
% q_1_bin: (Nmag-by-1) wavevector magnitude

%% 3D S(q)

% Other quantities
N = size(x,1); % number of particles
L_grid = min(box)/N_grid; % grid spacing

N_grid = round(box./L_grid); % real number of grid points in each direction
L_grid = box./N_grid; % real grid spacing

% Assign particles to a grid cell.
[cells,~] = CellList(x,box,L_grid);

% Convert cell subscripts to linear indices.
lincell = sub2ind(N_grid, cells(:,1), cells(:,2), cells(:,3));

% Count the number of particles in each cell.  This places each particle on
% the (midpoints of the) grid.
[xgrid,~] = histcounts(lincell,1:(prod(N_grid)+1));

% Convert the grid vector to a 3 dimensional array.
xgrid = reshape(xgrid, N_grid);

% Take the Fourier transform of the gridded positions to calculate the
% structure factor S(q).  S(i,j,k) is the structure factor for the three
% dimensional wavevector q = [qx,qy,qz] in Fourier space. 
S_3 = abs(fftshift(fftn(xgrid))).^2/N;

% Associate a wavevector with each S value.
dq = 2*pi./box; % distance between grid points in reciprocal space
q_3_x = (-N_grid(1)/2:N_grid(1)/2-1)*dq(1); % coordinates of the reciprocal grid in one dimension
q_3_y = (-N_grid(2)/2:N_grid(2)/2-1)*dq(2); % coordinates of the reciprocal grid in one dimension
q_3_z = (-N_grid(3)/2:N_grid(3)/2-1)*dq(3); % coordinates of the reciprocal grid in one dimension
[q_3_x,q_3_y,q_3_z] = ndgrid(q_3_x,q_3_y,q_3_z); % q_3_x(i,j,k) contains the qx coordinate of the i,j,k reciprocal grid point. Similar for qy and qz.

%% 2D S(q)

% Get 2D q vector
q_2_para = abs(q_3_x); % parallel component of q (x direction)
q_2_perp = sqrt(q_3_y.^2+q_3_z.^2); % perpendicular component (in yz-plane)

% Bin the q values
q_2_para_bin = (0:dq(1):max(q_2_para(:))+dq(1))';  % bin centers
q_2_perp_bin = (0:dq(2):max(q_2_perp(:))+dq(2))';  % bin centers

q_2_para_binedge = [q_2_para_bin; q_2_para_bin(end)+dq(1)] - dq(1)/2;  % bin edges
q_2_perp_binedge = [q_2_perp_bin; q_2_perp_bin(end)+dq(2)] - dq(2)/2;  % bin edges

[~,~,ind_para] = histcounts(q_2_para(:), q_2_para_binedge);  % bin the q_para values
[~,~,ind_perp] = histcounts(q_2_perp(:), q_2_perp_binedge);  % bin the q_perp values

% Average the S values for each q_para within each q_perp bin.
S_2 = accumarray([ind_para,ind_perp], S_3(:), [length(q_2_para_bin),length(q_2_perp_bin)], @mean);

% Create grid for Sxy
[q_2_para_grid, q_2_perp_grid] = ndgrid(q_2_para_bin, q_2_perp_bin);

%% 1D S(q)

% 1D magnitude of the wavevector at each grid point
q_1 = sqrt(q_3_x.^2+q_3_y.^2+q_3_z.^2);

% Bin the q values
dq = mean(dq); % bin size
q_1_bin = (0:dq:max(q_1(:))+dq)';  % bin centers
q_1_binedge = [q_1_bin; q_1_bin(end)+dq] - dq/2;  % bin edges
[~,~,ind] = histcounts(q_1(:),q_1_binedge);  % bin the q values

% Average the S(q) values within a q bin.
S_1 = accumarray(ind, S_3(:), [length(q_1_bin),1], @mean);

end