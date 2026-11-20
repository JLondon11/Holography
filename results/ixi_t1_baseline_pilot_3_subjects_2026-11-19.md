# Three-subject baseline pilot — real IXI MRI validation (user-reported, 2026-11-19)

**Evidence source:** User-provided terminal output of `experiments/compare_mri_baselines.py`. The local raw CSV, summary JSON, and checkpoint were not independently retrieved or rerun by the assistant. The 93-subject test partition was not accessed in this benchmark.

## Configuration
- Real acquired IXI validation subjects: IXI-002, IXI-020, IXI-041.
- Full validation cohort: 88; evaluated pilot: 3.
- Optical propagation: numerically simulated ASM with wavelength 532 nm, pixel pitch 3.74 µm, planes 4/6/8 mm, spatial resolution 128.
- Iterative baselines: 40 steps, direct Adam lr=0.05; CPU, PyTorch 2.2.2, Python 3.11.13.
- Frozen checkpoint SHA-256: `cfb80dfbb701b8126657163045755455cf5d64bef676a3d8f22a45b2be601c8c`
- Manifest SHA-256: `dc68afaf4d40183aff16a50bfe7a41cb06405861a5d5efaa95abf0fcc8037983`
- Seed: 20261119.

## Subject MSE

| Subject | PhaseCNN | Direct Adam | MP-GS |
|---|---:|---:|---:|
| IXI-002 | 0.82630 | 0.70401 | 0.81683 |
| IXI-020 | 0.66092 | 0.63125 | 0.71604 |
| IXI-041 | 0.57882 | 0.57174 | 0.63309 |

## Pilot aggregate

| Method | Mean MSE | Median MSE | Mean runtime (s) | Median runtime (s) |
|---|---:|---:|---:|---:|
| PhaseCNN | 0.6886794434653387 | 0.6609183549880981 | 0.1129414983248959 | 0.051296217017807066 |
| Direct Adam | 0.6356656120883093 | 0.6312466859817505 | 1.8442100379616022 | 0.8310452789301053 |
| MP-GS | 0.7219883335961236 | 0.7160404523213705 | 0.816654181612345 | 0.816137864952907 |

## Interpretation / caveats
- Direct Adam beats both other approaches on all 3 subjects by MSE.
- PhaseCNN beats MP-GS on 2 of 3 subjects, but MP-GS beats PhaseCNN on IXI-002.
- Based on **mean** runtimes, CNN is about 16.33x faster than Adam and 7.23x faster than MP-GS. The first-run overhead / small sample may bias this comparison; use warmed-up repeated timing for the full benchmark.
- Adam is a per-target numerical optimizer rather than amortized neural inference. CNN training costs are excluded from these timings.
- MRI intensities are prescribed target patterns, **not** directly measured optical tissue fields. All propagated fields are simulated, with no physical SLM measurements.
- These data do not establish the anatomical fidelity or scientific novelty required for IEEE publication.
- Do not use the untouched 93-subject test cohort for hyperparameter selection.
- Required next step: execute pre-specified full 88-validation-subject run, preserve full per-subject output and manifest, then inspect failure cases and depth fidelity.
