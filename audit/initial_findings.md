# Initial scientific audit — 2026-10-30

**Source:** Justin London, *Fourier Optics and Deep Learning Methods for Fast 3D Reconstruction in Digital Holography*, arXiv:2508.06703v1, August 2025.

## Critical reproducibility issue — Table I, page 14
The AP SP-3D-AS-CH filtered row reports MSE=0.009, RMSE=0.0304, PSNR=44.08 dB. On a common pixel domain, RMSE must equal sqrt(MSE), which would give approximately 0.09487. For normalized peak intensity 1, MSE=0.009 gives PSNR approximately 20.46 dB. The reported metrics therefore cannot all use the same sample set and normalization as currently stated. **Do not silently replace historical numbers**. Recalculate directly from preserved reconstructions, or mark unreproducible.

## Questions to resolve
- Optical wavelength on page 11 has contradictory unit notation. Reconstruct the parameter table using SI dimensions.
- Separate CGH *synthesis* from digital holography *measurement and inversion*.
- Clarify how MRI intensities and segmented points become object-field amplitude and phase.
- State whether reported filtering acts on SLM phase, object intensity, predicted intensity, or error feedback.
- Quantify depth accuracy, axial point-spread response, cross-plane leakage, runtime, and constraints on the SLM.
- HoloNet training/test splits, code revision, hardware, seed, optimizer and numerical normalization require full provenance.
- No physical SLM measurement is established by Pepper's ghost projection.

## Policy
The historical preprint is **source material**, not a reproducible validation benchmark. All new result tables and figures must be programmatically derived from preserved run manifests, raw output arrays and scripts.
