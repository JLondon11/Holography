# Hybrid PhaseCNN + Physics Refinement — 88 Real IXI T1 Validation Subjects

**Data provenance:** User-reported terminal output on 2026-11-19. This is an aggregate-results record, **not** independent access to the underlying local checkpoint, MRI images, or full CSV. The acquired MRI data remain on the local workstation.

## Experiment
- Cohort: full **88 IXI T1 validation subjects**; no held-out test set used in this run.
- 581 total acquired MRI scans were split previously into 400 training, 88 validation, and 93 held-out test subjects.
- Original PhaseCNN checkpoint SHA-256: `cfb80dfbb701b8126657163045755455cf5d64bef676a3d8f22a45b2be601c8c`.
- Manifest SHA-256: `dc68afaf4d40183aff16a50bfe7a41cb06405861a5d5efaa95abf0fcc8037983`.
- Optical simulation parameters: wavelength 532 nm, pixel pitch 3.74 µm, 128x128 pixels, depth planes 4/6/8 mm.
- Adam learning rate 0.05, iteration budgets [5,10,20], seed 20261119, CPU inference.
- Input: MRI-derived prescribed target intensity. Propagation: scalar angular spectrum numerical simulation only; **not** optical bench measurements or reconstruction of true tissue phase.

## Reported full-cohort results

| Method | Steps | Mean MSE | Median MSE | Mean CPU seconds |
|---|---:|---:|---:|---:|
| PhaseCNN | 0 | 0.7039870782331987 | 0.6845525205135345 | 0.031751116873982195 |
| CNN-initialized Adam | 5 | 0.5698296902983477 | 0.5580212771892548 | 0.11227873898513446 |
| CNN-initialized Adam | 10 | 0.5063419948247346 | 0.49736495316028595 | 0.19133151504502166 |
| CNN-initialized Adam | 20 | **0.4290823080774509** | 0.42360086739063263 | 0.34407673755380197 |
| Random-initialized Adam | 5 | 1.3989201050364612 | 1.3814699252446492 | 0.08288137806150851 |
| Random-initialized Adam | 10 | 1.1970865410385712 | 1.188560883204142 | 0.1633764154023745 |
| Random-initialized Adam | 20 | 0.9193444906762153 | 0.9171130458513896 | 0.3193468612298602 |

Calculated from cohort means: 20-step warm-start refinement has ~39.05% lower MSE than original CNN and ~53.33% lower MSE than random-initialized Adam at 20 steps. The 20-step hybrid is ~33.25% below a separately run 40-step direct-Adam benchmark (MSE 0.642779), but step budgets and runs differ.

## Preliminary subject-specific observations
From 88 console entries, CNN MSE exceeds 20-step hybrid MSE for every subject. Examples with high residual hybrid MSE:
- IXI-373: CNN 1.40258 -> hybrid 0.76634.
- IXI-337: CNN 1.27821 -> hybrid 0.71128.
- IXI-571: CNN 0.99039 -> hybrid 0.57944.
- IXI-255: CNN 0.88342 -> hybrid 0.58119.
- IXI-541: CNN 1.01963 -> hybrid 0.57324.
This console log does not include per-subject random Adam values, so do **not** infer an 88/88 hybrid-vs-random win rate. Run `experiments/analyze_refinement.py` on the local `per_subject_refinement.csv` to determine it.

## Scientific constraints and next decisions
1. Check the paired CSV for complete subject counts, matched initialization/iteration budgets, win/tie/loss and outlier performance.
2. Inspect actual MRI target vs reconstructed intensity planes for failures and success cases. Calculate axial depth crosstalk and spatial fidelity.
3. Study convergence behavior and assess multi-start optimizer variance and alternative optimization budgets without test leakage.
4. These validation data have already been used to choose the CNN checkpoint; any further design adjustment remains validation tuning. Freeze complete protocol before the 93 held-out test scans are evaluated.
5. The learned initializer + gradient-based phase optimization is a candidate hybrid workflow, **not established novelty** or demonstrated clinical/optical validation. Check prior literature before novelty claims.

