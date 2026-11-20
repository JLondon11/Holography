"""Generate an ORIGINAL qualitative numerical-propagation illustration.

This image is a simulated demonstration, not validation against measurement.
Run after: pip install -e . && pip install matplotlib
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from holography.propagation import angular_spectrum, intensity

n = 256
pitch = 3.74e-6
coord = (np.arange(n) - n // 2) * pitch
xx, yy = np.meshgrid(coord, coord)
waist = 60e-6
field = np.exp(-(xx**2 + yy**2) / waist**2).astype(complex)
distances_m = (0.0, 0.005, 0.015)
fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), constrained_layout=True)
for ax, z in zip(axes, distances_m):
    out = angular_spectrum(field, 532e-9, pitch, pitch, z)
    im = ax.imshow(intensity(out), cmap="viridis", origin="lower",
                   extent=[coord[0]*1e3, coord[-1]*1e3]*2,
                   vmin=0, vmax=1)
    ax.set_title(f"z = {z*1e3:.1f} mm")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
fig.colorbar(im, ax=axes, label="Intensity (a.u.)")
fig.savefig("figures/gaussian_propagation.png", dpi=300)
fig.savefig("figures/gaussian_propagation.pdf")
print("Original simulated figure generated; not physical experimental data")
