function out = angular_spectrum(field, wavelength_m, dx_m, dy_m, distance_m)
%ANGULAR_SPECTRUM Research reference ASM propagation in SI units.
% Axis order: rows=y, columns=x. Excludes evanescent components.
% Does not ensure alias-free sampling for arbitrary geometry.
validateattributes(field, {'numeric'}, {'2d','finite'});
validateattributes(wavelength_m, {'numeric'}, {'scalar','positive','finite'});
validateattributes(dx_m, {'numeric'}, {'scalar','positive','finite'});
validateattributes(dy_m, {'numeric'}, {'scalar','positive','finite'});
validateattributes(distance_m, {'numeric'}, {'scalar','real','finite'});
[ny,nx] = size(field);
fx = ifftshift((-floor(nx/2):ceil(nx/2)-1)/(nx*dx_m));
fy = ifftshift((-floor(ny/2):ceil(ny/2)-1)/(ny*dy_m));
[FX,FY] = meshgrid(fx,fy);
q = 1/(wavelength_m^2) - FX.^2 - FY.^2;
mask = q >= 0;
H = exp(1i*2*pi*distance_m*sqrt(max(q,0))) .* mask;
if distance_m ~= 0
    limx = 1/(wavelength_m*sqrt(1+(2*abs(distance_m)/(nx*dx_m))^2));
    limy = 1/(wavelength_m*sqrt(1+(2*abs(distance_m)/(ny*dy_m))^2));
    H = H .* (abs(FX)<=limx) .* (abs(FY)<=limy);
end
out = ifft2(fft2(double(field)).*H);
end
