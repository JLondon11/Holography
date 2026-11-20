# Real IXI MRI holography — anatomical edge and native-depth diagnostic, 88 validation subjects

**Provenance:** User-reported terminal outputs on 2026-11-19 from
`experiments/analyze_native_adjusted_depth.py` and
`experiments/evaluate_edge_fidelity.py`. Raw acquired MRI data, saved local CSVs
and checkpoint were not independently accessed in this commit.

## Dataset and scope

- Real acquired IXI T1 MRI: 88 validation subjects; three MRI-derived optical display target intensity planes each (264 plane reconstructions).
- Frozen PhaseCNN vs PhaseCNN-initialized Adam phase-only refinement, 20 steps at learning rate 0.05.
- Wavelength 532 nm, SLM pixel pitch 3.74 μm, propagation distances 4/6/8 mm, 128×128 sampling.
- Held-out 93-subject test split remains unused **by these validation scripts**.
- Checkpoint SHA-256: `cfb80dfbb701b8126657163045755455cf5d64bef676a3d8f22a45b2be601c8c`.
- Manifest SHA-256: `dc68afaf4d40183aff16a50bfe7a41cb06405861a5d5efaa95abf0fcc8037983`.
- Simulated scalar coherent-wave propagation from acquired MRI **intensity targets**, not MRI tissue optics or measured laboratory holography.

## Edge-fidelity measurements (user-reported)

| Depth | CNN mean MSE | Hybrid mean MSE | CNN mean gradient-magnitude Pearson r | Hybrid mean gradient-magnitude Pearson r | CNN mean relative gradient MAE | Hybrid mean relative gradient MAE |
|---|---:|---:|---:|---:|---:|---:|
| 4 mm | 0.7352617295 | 0.4387106972 | 0.2112830086 | **0.5968618009** | 1.0663051948 | 0.9751757380 |
| 6 mm | 0.7032947590 | 0.4120312975 | 0.2068110009 | **0.5584507034** | 0.8311911048 | 0.7254649751 |
| 8 mm | 0.6734047501 | 0.4365049242 | 0.2181476445 | **0.5181049669** | 0.8860803367 | 0.7885440300 |

Hybrid exceeds CNN in gradient-magnitude correlation on **264/264 evaluable validation planes**. **This metric does not establish correct medical boundaries or clinical image fidelity.** Relative gradient errors remain large, consistent with visible speckle and detail loss.

## Native-similarity-adjusted depth margins (user-reported)

Exploratory index = [r(reconstructed_i, target_i) – max_j!=i r(reconstructed_i,target_j)] / [1 – max_j!=i r(target_i,target_j)].

Index is descriptive; it is not a probability or calibrated physical crosstalk.

| Plane | Median raw margin | Median native-target contrast | Median adjusted margin | Eligible | Correct target matches |
|---|---:|---:|---:|---:|---:|
| 4 mm | 0.1192404463 | 0.1454252641 | 0.8292355042 | 88/88 | 88/88 |
| 6 mm | 0.1406405637 | 0.3957162188 | 0.3544485769 | 88/88 | 88/88 |
| 8 mm | 0.0802497303 | 0.1454252641 | 0.5490240004 | 88/88 | 88/88 |

All denominators meet the implemented minimum native contrast of 0.05. The ranking changes under normalization, so claims of 8 mm physical optical failure would be premature.

## Research decisions

1. Freeze and preserve the PhaseCNN checkpoint and initial 20-step hybrid as **reference benchmarks**; keep no-regression plots and metrics.
2. A candidate edge-aware loss should first be studied on validation data with a fixed compute budget, honest comparison with original MSE and gradient metrics, and no claimed improvement until results exist.
3. Visually inspect true MRI-derived reconstructed intensities and gradients (including the difficult IXI-337/373) to detect speckle amplification and misleading edge-correlation gains.
4. Model and quantify physical limitations (SLM sampling, zero-order leakage, phase quantization, diffraction efficiency, axial cross-talk) only with justified optical assumptions; don't relabel simulated fields as laboratory measurements.
5. Final manuscript should describe anatomical MRIs as *image sources for prescribed holographic display targets*, not optical measurements of brain tissue. Reserve untouched 93-subject evaluation until protocol and final model choice are frozen.
