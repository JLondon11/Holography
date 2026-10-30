# Holography — IEEE Journal Research

Reproducible research workspace for **Fourier optics, digital holography, and physics-constrained three-dimensional computer-generated holography**.

## Scientific status
This repository starts as a **research implementation**, not a validated optical instrument or a publication-ready result. Historical numerical results in the August 2025 preprint must be audited independently before reuse. No figures or tables may be represented as measured or experimentally validated without appropriate evidence.

## Project layout
- `manuscript/` — IEEE LaTeX manuscript, references, and revisions.
- `audit/` — equation, numerical-result, figure, and provenance audits.
- `src/holography/` — Python Fourier-optics and reconstruction algorithms.
- `matlab/` — MATLAB implementations and validation scripts.
- `tests/` — physics and numerical regression tests.
- `experiments/` — tracked configurations and evaluation protocols.
- `figures/` — original figures and reproducible generation scripts.
- `data/` — dataset manifests only; avoid committing patient data or large scans.
- `results/` — lightweight verified result summaries with provenance.
- `.github/workflows/` — continuous integration.

## Research integrity
1. Preserve the original 2025 arXiv version (arXiv:2508.06703v1) as a *historical baseline*, not a verified ground truth.
2. Cite sources and verify permission before redistributing third-party images or datasets.
3. Produce each quantitative figure from executable code and attributable measurements.
4. Verify units, propagation sampling, energy conservation, PSNR/MSE/RMSE consistency, seeds, and runtime methodology.
5. Label simulated data, numerical benchmarks, and optical measurements separately.
6. Do not merge unreviewed changes into the primary branch.

## Getting started
Python code and a small numerical test suite will be developed on the journal-revision branch. MATLAB and deep-learning components are separate research deliverables and should not be claimed complete until implemented and validated.

**Repository:** https://github.com/JLondon11/Holography
