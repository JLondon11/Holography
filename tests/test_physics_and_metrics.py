"""Numerical smoke tests, not experimental validation."""
import numpy as np
from holography.propagation import angular_spectrum
from holography.metrics import reconstruction_metrics

def test_zero_distance_identity():
    rng = np.random.default_rng(7)
    field = rng.normal(size=(64, 64)) + 1j * rng.normal(size=(64, 64))
    got = angular_spectrum(field, 532e-9, 3.74e-6, 3.74e-6, 0)
    np.testing.assert_allclose(got, field, atol=1e-12, rtol=1e-12)

def test_constant_plane_wave_phase():
    field = np.ones((64, 64), dtype=complex)
    z, wavelength = 0.002, 532e-9
    got = angular_spectrum(field, wavelength, 3.74e-6, 3.74e-6, z)
    expected = np.exp(1j * 2 * np.pi * z / wavelength)
    np.testing.assert_allclose(got, expected, atol=1e-10, rtol=1e-10)

def test_metrics_consistent():
    got = reconstruction_metrics(np.zeros((2, 2)), np.ones((2, 2)) * 0.1)
    np.testing.assert_allclose(got["mse"], 0.01)
    np.testing.assert_allclose(got["rmse"], 0.1)
    np.testing.assert_allclose(got["psnr_db"], 20)

def test_invalid_arrays_rejected():
    import pytest
    with pytest.raises(ValueError):
        reconstruction_metrics(np.zeros((2,2)), np.zeros((3,3)))
