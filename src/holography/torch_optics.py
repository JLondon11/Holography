"""Differentiable scalar angular-spectrum propagation. SI units, research only."""
import torch

def propagate(field: torch.Tensor, wavelength_m: float, pitch_m: float, z_m: float):
    if not field.is_complex() or field.ndim < 2:
        raise ValueError("Expected complex tensor [..., height, width]")
    if wavelength_m <= 0 or pitch_m <= 0:
        raise ValueError("Wavelength and pitch must be positive")
    h, w = field.shape[-2:]
    fy = torch.fft.fftfreq(h, d=pitch_m, device=field.device).reshape(h, 1)
    fx = torch.fft.fftfreq(w, d=pitch_m, device=field.device).reshape(1, w)
    q = wavelength_m**-2 - fx**2 - fy**2
    kz = 2 * torch.pi * torch.sqrt(torch.clamp(q, min=0))
    transfer = torch.exp(1j * z_m * kz) * (q >= 0)
    if z_m != 0:
        limx = 1 / (wavelength_m * (1 + (2*abs(z_m)/(w*pitch_m))**2)**0.5)
        limy = 1 / (wavelength_m * (1 + (2*abs(z_m)/(h*pitch_m))**2)**0.5)
        transfer = transfer * (fx.abs() <= limx) * (fy.abs() <= limy)
    return torch.fft.ifft2(torch.fft.fft2(field) * transfer)
