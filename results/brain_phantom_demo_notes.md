# Synthetic brain-shaped hologram training — first demonstration

Run completed locally on CPU (PyTorch 2.10.0+cpu, Python 3.13.5), 96x96 samples, phase-only CNN, 35 optimization steps, wavelength 532 nm, pixel pitch 3.74 um, depths 4/6/8 mm, train phantom seeds 19/23/29 and held-out seed 31.

Standalone demonstration:
- Initial train MSE: 0.71723878
- Final train MSE: 0.68622392
- Held-out three-plane intensity MSE: 0.68606031

**Interpretation:** outputs remain very blurry and do not meet anatomical reconstruction targets. The optical target power, support/background handling, loss weighting, depth crosstalk and architecture require improvement. This is NOT a real brain MRI experiment and does NOT validate publication claims.

Images and checkpoint from local execution are separate deliverables pending direct binary GitHub upload; scripts committed to this branch reproduce the experiment approach but should be rerun and verified before numerical equivalence is claimed.
