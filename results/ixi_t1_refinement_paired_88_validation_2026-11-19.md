# Paired hybrid refinement: full 88-subject real IXI MRI validation (2026-11-19)

**Evidence source:** Numerical results transcribed from user-provided `experiments/analyze_refinement.py` output. Underlying CSV and actual MRI volumes are held locally; this report is not an independent rerun.

- Status: validation-only IXI T1 acquired MRI-derived display intensity targets; **not measured optical reconstruction**.
- Subjects: 88/88 validation subjects. Pilot subset: false. Held-out test data accessed in this benchmark: false.
- Model SHA-256: `cfb80dfbb701b8126657163045755455cf5d64bef676a3d8f22a45b2be601c8c`
- Manifest SHA-256: `dc68afaf4d40183aff16a50bfe7a41cb06405861a5d5efaa95abf0fcc8037983`

## Paired comparisons (positive difference = lower hybrid MSE)

| Steps | Comparator | Hybrid mean MSE | Comparator mean MSE | Relative mean MSE reduction | Mean paired advantage | Median paired advantage | Win/Tie/Loss |
|---|---|---:|---:|---:|---:|---:|---|
| 5 | random Adam | 0.5698296903 | 1.3989201050 | 59.27% | 0.8290904147 | 0.8288177550 | 88/0/0 |
| 5 | PhaseCNN | 0.5698296903 | 0.7039870782 | 19.06% | 0.1341573879 | 0.1272895088 | 88/0/0 |
| 10 | random Adam | 0.5063419948 | 1.1970865410 | 57.70% | 0.6907445462 | 0.6887303293 | 88/0/0 |
| 10 | PhaseCNN | 0.5063419948 | 0.7039870782 | 28.08% | 0.1976450834 | 0.1857150892 | 88/0/0 |
| 20 | random Adam | **0.4290823081** | 0.9193444907 | **53.33%** | **0.4902621826** | **0.4873029590** | **88/0/0** |
| 20 | PhaseCNN | **0.4290823081** | 0.7039870782 | **39.05%** | **0.2749047702** | **0.2609882553** | **88/0/0** |

20-step hybrid relative to random Adam paired advantage IQR: 0.4621479561 to 0.5072053348.
20-step hybrid relative to PhaseCNN paired advantage IQR: 0.2155426343 to 0.3051065380.

## Time comparison (mean CPU seconds per subject)

| Method | Steps | Seconds |
|---|---:|---:|
| PhaseCNN alone | 0 | 0.0317511169 |
| CNN + Adam | 5 | 0.1122787390 |
| CNN + Adam | 10 | 0.1913315150 |
| CNN + Adam | 20 | 0.3440767376 |
| Random Adam | 20 | 0.3193468612 |

**Caveats:** Warm-start includes CNN inference cost; no repeat-timing study or training-time amortization. Random Adam uses one initialization seed and only the specified 5/10/20-step budgets; results do not prove superiority to tuned/multi-start direct optimization or peer-reviewed benchmarks. These validation images selected the best checkpoint, so observed validation results do not represent an untouched test set.

## Highest 20-step hybrid MSE

| Subject | PhaseCNN MSE | Hybrid MSE | Random Adam MSE |
|---|---:|---:|---:|
| IXI-373 | 1.4025750160 | 0.7663435737 | 1.3864668210 |
| IXI-337 | 1.2782088518 | 0.7112787366 | 1.2768917878 |
| IXI-255 | 0.8834157586 | 0.5811892351 | 1.0737606684 |
| IXI-571 | 0.9903858304 | 0.5794375340 | 1.1099739869 |
| IXI-541 | 1.0196291208 | 0.5732352734 | 1.1059605877 |

## Next research gates
1. Visualize the acquired MRI slices and all three numerically propagated target/reconstruction planes (especially IXI-373 and IXI-337).
2. Quantify amplitude/intensity structure fidelity and axial crosstalk; explicitly distinguish prescribed intensity planes from any measured optical tissue fields.
3. Introduce phase quantization and realistic SLM limitations only if justified by actual optical hardware specifications; avoid fabricated experimental settings.
4. Predefine final models, step budgets, metrics, and comparison methods before testing the 93 held-out subjects.
5. Investigate novelty with direct prior-art comparisons and acknowledge existing neural-init + phase-refinement approaches.
