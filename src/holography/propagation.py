"""Fourier optics reference routines. Research-only numerical implementation.

Conventions: spatial coordinates in metres; wavelength in metres; z in metres.
Array axes are (y, x). FFT uses numpy convention. Propagating angular-spectrum
components use exp(+i*kz*z); evanescent components are excluded.
"""
from __future__ import annotations
import numpy as np

def angular_spectrum(field: np.ndarray, wavelength_m: float, dx_m: float,
                     dy_m: float, distance_m: float, bandlimit: bool = True) -> np.ndarray:
    """Propagate a sampled complex field using a shifted-free FFT angular spectrum.

    The optional cutoff removes evanescent waves and applies a conservative
    rectangular bandlimit to help avoid transfer-function aliasing. It is not
    an all-purpose guarantee of alias-free propagation; users must validate
    sampling and padding for their specific optical geometry.
    """
    u = np.asarray(field, dtype=np.complex128)
    if u.ndim != 2 or not np.all(np.isfinite(u)):
        raise ValueError("field must be finite and two-dimensional")
    if wavelength_m <= 0 or dx_m <= 0 or dy_m <= 0:
        raise ValueError("wavelength and pixel pitches must be positive")
    ny, nx = u.shape
    fx = np.fft.fftfreq(nx, dx_m)[None, :]
    fy = np.fft.fftfreq(ny, dy_m)[:, None]
    inv_lam_sq = 1.0 / wavelength_m**2
    q = inv_lam_sq - fx**2 - fy**2
    propagating = q >= 0
    kz = 2 * np.pi * np.sqrt(np.maximum(q, 0.0))
    transfer = np.exp(1j * distance_m * kz) * propagating
    if bandlimit and distance_m != 0:
        # Band-limited angular-spectrum support (rectangular conservative rule)
        fx_limit = 1.0 / (wavelength_m * np.sqrt(1 + (2 * abs(distance_m) / (nx * dx_m))**2))
        fy_limit = 1.0 / (wavelength_m * np.sqrt(1 + (2 * abs(distance_m) / (ny * dy_m))**2))
        transfer *= (np.abs(fx) <= fx_limit) & (np.abs(fy) <= fy_limit)
    return np.fft.ifft2(np.fft.fft2(u) * transfer)

def intensity(field: np.ndarray) -> np.ndarray:
    """Nonnegative intensity (arbitrary units) from a complex electric field."""
    return np.abs(np.asarray(field)) ** 2
