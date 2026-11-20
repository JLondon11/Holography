# Real IXI T1 depth-selectivity proxy — 88 validation subjects

**Evidence provenance:** This is user-reported terminal output from `experiments/evaluate_depth_cohort.py`. Local MRI volumes, checkpoint, and per-plane CSV files were not independently retrieved or audited here.

## Data and optical settings
- 88 real acquired IXI T1 validation MRI volumes, three MRI-derived target planes per subject.
- No held-out test-set access in this evaluation.
- PhaseCNN followed by 20 iterations of target-aware Adam refinement, learning rate 0.05.
- 128x128 resolution, 532 nm wavelength, 3.74 µm pixel pitch, 4/6/8 mm simulated propagation distances.
- Checkpoint SHA-256: `cfb80dfbb701b8126657163045755455cf5d64bef676a3d8f22a45b2be601c8c`
- Manifest SHA-256: `dc68afaf4d40183aff16a50bfe7a41cb06405861a5d5efaa95abf0fcc8037983`

## Full validation findings

| Metric | Reported value |
|---|---:|
| Validation subjects | 88 |
| Plane comparisons with valid match assessment | 264 |
| Correct within-subject plane matches | 264 |
| Fraction correct | 1.0 |
| Median diagonal correlation minus best off-diagonal | 0.1140865825118837 |
| Planes flagged for high native target similarity (>0.9) | 14 |
| PhaseCNN cohort mean MSE | 0.7039870714599435 |
| 20-step hybrid cohort mean MSE | 0.4290823089805516 |

## Interpretation
The numerical reconstructions at 4, 6, and 8 mm correlate most strongly with their assigned MRI-derived target plane rather than the two other slices in all 264 evaluated plane comparisons. The 39.05% lower MSE than the original PhaseCNN is reproduced by this diagnostic.

**This is a within-subject, target-aware spatial-correlation ranking, not physical optical crosstalk, demonstrated axial resolution, volumetric optical measurement, or clinically faithful 3D brain reconstruction.** Native MRI slices are correlated; 14 planes have another target with correlation over 0.9. A correct ranking with a modest median margin (0.1141) does not quantify physical separability. The model is optimized against those same target images during refinement; this is not independent ground truth or evidence of a learned 3D tissue-optical field.

## Next research requirements
1. Inspect full `depth_matrix_long.csv` for margin distributions, near-ties and MRI slice similarity.
2. Measure on-target vs off-target cross-depth errors and address cases where source MRI target images are intrinsically similar. Treat any derived selectivity metric as a proxy, not calibrated optical crosstalk.
3. Investigate off-axis energy, optical bandlimit/FFT boundary effects, speckle, and reconstructed anatomical edges using acquired MRI targets only.
4. Predefine the full final protocol and only then use the 93 held-out test subjects.
