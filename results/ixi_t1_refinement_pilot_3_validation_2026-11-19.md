# Real IXI MRI CNN-initialized phase refinement: 3-subject pilot

**Evidence status:** User-reported terminal output of `experiments/refine_mri_validation.py` on 2026-11-19. The underlying CSV/checkpoint/scan data were not directly inspected or independently executed by the assistant. Do not treat this document as a raw-data archive.

## Scope
- Three acquired IXI T1 validation MRI subjects: IXI-002, IXI-020, IXI-041 (pilot subset of the 88 validation subjects).
- 93 held-out test MRI subjects **not accessed** by this experiment.
- Input images are acquired MRIs; target intensity is a derived prescribed optical intensity pattern; all propagation is numerical simulation, **not** optical-bench measurement or a measurement of MRI tissue's coherent optical field.
- 128×128 pixels, 532 nm wavelength, 3.74 μm pitch, 4/6/8 mm propagation depths.
- PhaseCNN checkpoint SHA256: `cfb80dfbb701b8126657163045755455cf5d64bef676a3d8f22a45b2be601c8c`.
- Manifest SHA256: `dc68afaf4d40183aff16a50bfe7a41cb06405861a5d5efaa95abf0fcc8037983`.
- Adam lr=0.05, 5/10/20 steps, initialization seed 20261119; frozen CNN; no additional network training.

## Reported measurements

| Method | Steps | Mean MSE | Median MSE | Mean CPU seconds |
|---|---:|---:|---:|---:|
| PhaseCNN | 0 | 0.6886794368426005 | 0.6609184145927429 | 0.06244739730997632 |
| CNN-initialized Adam | 5 | 0.5613936848110622 | 0.5415124694506327 | 0.148075147299096 |
| CNN-initialized Adam | 10 | 0.5024630957179599 | 0.488053431113561 | 0.20406815332050124 |
| CNN-initialized Adam | 20 | **0.4273065361711714** | 0.41564470529556274 | 0.3050768896549319 |
| Random-initialized Adam | 5 | 1.3621439006593494 | 1.327316681543986 | 0.051439768013854824 |
| Random-initialized Adam | 10 | 1.1672627925872803 | 1.1358621716499329 | 0.10615149427515765 |
| Random-initialized Adam | 20 | 0.901786572403378 | 0.8829874595006307 | 0.21117228432558477 |

Reported per-subject CNN vs CNN+20 MSE:
- IXI-002: 0.82630 -> 0.51214
- IXI-020: 0.66092 -> 0.41564
- IXI-041: 0.57882 -> 0.35414

## Calculated interpretation
- CNN+20 versus CNN alone: approx 37.96% lower cohort mean MSE.
- CNN+20 versus random Adam at 20 steps: approx 52.62% lower cohort mean MSE.
- CNN+20 mean runtime ~0.305 s vs CNN-only ~0.062 s. Comparisons here include the CNN inference cost for warm-start refinement; timings are not a repeated/warmed-up hardware performance study.
- Small pilot (n=3), with only one initialization seed and a single optimization schedule. Do not infer statistical significance.
- In this setting the frozen PhaseCNN provides a numerically advantageous initialization. This is not evidence of superiority against strong published neural holography baselines or physical systems.

## Next protocol
Run the same preregistered settings on all 88 validation subjects. Report paired per-subject MSE, how often warm-start beats random-start, distribution of paired differences, per-plane errors where available, and runtime. Preserve the held-out test split until all design decisions are frozen.
