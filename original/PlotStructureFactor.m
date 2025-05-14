function [] = PlotStructureFactor(x, box)

% INPUTS
% x = (N-by-3) particle positions
% box = (1-by-3) box dimensions

% Add the N_grid
N = size(x,1);
N_grid = 300;

close all

% Compute the stucture factor
[S_3,q_3_x,q_3_y,q_3_z,S_2,q_2_para,q_2_perp,S_1,q_1] = StructureFactor(x,box,N_grid);
h_S_2_perp = figure(); hold on
h_S_2_para_ave = figure(); hold on
h_S_2_perp_ave = figure(); hold on

% Plot spherically averaged S(q)
figure
plot(q_1(2:end), S_1(2:end), 'LineWidth', 3)
xlabel('q')
ylabel('S')
xlim([0,10])
set(gca,'FontUnits','normalized','FontSize',0.05,...
    'FontWeight','bold','LineWidth',1,'PlotBoxAspectRatio',[1,1,1])

% Plot S(q_para,q_perp=0)
figure
plot(q_2_para(2:end,1), S_2(2:end,1), 'LineWidth',2)
xlabel('q_{||}')
ylabel('S')
xlim([0,10])
set(gca,'FontUnits','normalized','FontSize',0.05,...
    'FontWeight','bold','LineWidth',1,'PlotBoxAspectRatio',[1,1,1])

% Plot S(q_para=0, q_perp)
figure(h_S_2_perp)
plot(q_2_perp(1,2:end),S_2(1,2:end), 'Color','k', 'LineWidth',2)
xlabel('q_\perp')
ylabel('S')
xlim([0,10])
set(gca,'FontUnits','normalized','FontSize',0.05,...
    'FontWeight','bold','LineWidth',1,'PlotBoxAspectRatio',[1,1,1])

% The previous 2 plots are noisy. Average the S values within a small
% wedge of angle theta_tol from the q_para = 0 or q_perp = 0 axes.
theta_tol = pi/12;

% Get cylindrical coordinates for each point
q_2_r = sqrt(q_2_para.^2+q_2_perp.^2);
q_2_theta = atan2(q_2_perp, q_2_para);

% Bin the q values
dq = [q_2_para(2,1) - q_2_para(1,1), q_2_perp(1,2) - q_2_perp(1,1)];
q_2_r_para_bin = (0:dq(1):max(q_2_r(:))+dq(1))';  % r bins for q_perp=0 (para) average
q_2_r_perp_bin = (0:dq(2):max(q_2_r(:))+dq(2))';  % r bins for q_para=0 (perp) average
q_2_r_para_binedge = (0:dq(1):max(q_2_r(:))+2*dq(1))' - dq(1)/2;  % r bin edges for q_perp=0 (para) average
q_2_r_perp_binedge = (0:dq(2):max(q_2_r(:))+2*dq(2))' - dq(2)/2;  % r bin edges for q_para=0 (perp) average
q_2_theta_binedge = [0; theta_tol; pi/2-theta_tol; pi/2]; % theta bins

[~,~,ind_r_para] = histcounts(q_2_r(:), q_2_r_para_binedge);  % bin the r values for q_perp=0 (para) average
[~,~,ind_r_perp] = histcounts(q_2_r(:), q_2_r_perp_binedge);  % bin the r values for q_para=0 (perp) average
[~,~,ind_theta] = histcounts(q_2_theta(:), q_2_theta_binedge);  % bin the q_perp values

% Average the S values for each r within each theta bin.
S_2_para = accumarray([ind_r_para,ind_theta], S_2(:), [length(q_2_r_para_bin),3], @mean);
S_2_perp = accumarray([ind_r_perp,ind_theta], S_2(:), [length(q_2_r_perp_bin),3], @mean);

% Keep only the correct theta bins
S_2_para = S_2_para(:,1);
S_2_perp = S_2_perp(:,3);

% Plot the 1D S(q)
figure(h_S_2_para_ave)
plot(q_2_r_para_bin(2:end), S_2_para(2:end), 'LineWidth', 3)
xlabel('q_{||}')
ylabel('S')
xlim([0,10])
set(gca,'FontUnits','normalized','FontSize',0.05,...
    'FontWeight','bold','LineWidth',1,'PlotBoxAspectRatio',[1,1,1])

figure(h_S_2_perp_ave)
plot(q_2_r_perp_bin(2:end), S_2_perp(2:end), 'Color', 'k', 'LineWidth', 3)
xlabel('q_\perp')
ylabel('S')
xlim([0,10])
set(gca,'FontUnits','normalized','FontSize',0.05,...
    'FontWeight','bold','LineWidth',1,'PlotBoxAspectRatio',[1,1,1])

% Filter out values larger than some tolerance
tol = 3;
S_2(S_2 >= tol) = nan;

% Plot the two-dimensional S(qxy,qz)
figure
colormap parula
h = pcolor(q_2_para, q_2_perp, S_2);
h.EdgeColor = 'none';
h.FaceColor = 'interp';
colorbar
xlabel('q_{||}')
ylabel('q_\perp')
xlim([0,5])
ylim([0,5])
set(gca,'FontUnits','normalized','FontSize',0.05,...
    'FontWeight','bold','LineWidth',1,'PlotBoxAspectRatio',[1,1,1])
    
% Isolate the qz = 0 plane
ind_0 = find(q_3_z == 0);
q_3_x_0 = q_3_x(ind_0);
q_3_y_0 = q_3_y(ind_0);
S_3_0 = S_3(ind_0);

% The previous calculations return vectors.  Reshape the vectors into
% grids.
Ngrid = size(q_3_z,1);
q_3_x_0 = reshape(q_3_x_0,Ngrid,Ngrid);
q_3_y_0 = reshape(q_3_y_0,Ngrid,Ngrid);
S_3_0 = reshape(S_3_0,Ngrid,Ngrid);

% Filter out values larger than some tolerance
tol = 3;
S_3_0(S_3_0 >= tol) = nan;

% Plot S(q) in the qz = 0 plane
figure
colormap hot
h = pcolor(q_3_x_0,q_3_y_0,S_3_0);
h.EdgeColor = 'none';
colorbar
xlabel('q_x')
ylabel('q_y')
xlim([-10,10])
ylim([-10,10])
title('q_z = 0')
set(gca,'FontUnits','normalized','FontSize',0.05,...
    'FontWeight','bold','LineWidth',1,'PlotBoxAspectRatio',[1,1,1])

S_3_ave = mean(S_3,3);
tol = 3;
S_3_ave(S_3_ave >= tol) = nan;

% Plot S(q) averaged over qz plane
figure
colormap hot
h = pcolor(q_3_x(:,:,1),q_3_y(:,:,1),S_3_ave);
h.EdgeColor = 'none';
colorbar
xlabel('q_x')
ylabel('q_y')
xlim([-10,10])
ylim([-10,10])
title('q_z average')
set(gca,'FontUnits','normalized','FontSize',0.05,...
    'FontWeight','bold','LineWidth',1,'PlotBoxAspectRatio',[1,1,1])

end 

