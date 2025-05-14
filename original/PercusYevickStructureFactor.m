function [S] = PercusYevickStructureFactor(q,phi)

% Volume fraction dependent parameters.
alpha = (1+2*phi).^2./(1-phi).^4;
beta = -6*phi.*(1+phi/2).^2./(1-phi).^4;
gamma = 1/2*phi.*(1+2*phi).^2./(1-phi).^4;

% Direct correlation function from the Ornstein-Zernike equation.
c = -3*(3*gamma-beta*q.^2 + ...
    (-3*gamma+(beta+6*gamma)*q.^2-2*(alpha+beta+gamma)*q.^4).*cos(2*q) + ...
    q.*(-6*gamma+(alpha+2*beta+4*gamma)*q.^2).*sin(2*q))./q.^6;

% Structure factor from the Percus-Yevick approximation.
S = 1./(1-phi.*c);

end