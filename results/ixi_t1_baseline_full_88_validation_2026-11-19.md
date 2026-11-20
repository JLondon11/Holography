# Full 88-subject real-IXI validation benchmark (user-reported 2026-11-19)

**Evidence status:** Transcribed from user-provided `experiments/compare_mri_baselines.py` terminal output. Raw `per_subject_methods.csv`, `summary.json`, MRI volumes, and model checkpoint remain on the user's Mac; this commit does not independently verify them.

## Protocol and provenance

- 88 of 88 acquired IXI T1 validation subjects; no held-out test data accessed by this benchmark.
- PhaseCNN frozen checkpoint SHA256: `cfb80dfbb701b8126657163045755455cf5d64bef676a3d8f22a45b2be601c8c`.
- Manifest SHA256: `dc68afaf4d40183aff16a50bfe7a41cb06405861a5d5efaa95abf0fcc8037983`.
- 128×128 target images; mean-normalized MRI-derived intensity targets; propagated numerically at 532 nm, 3.74 μm pixel pitch, 4/6/8 mm depths.
- Adam: 40 iterations, lr=0.05; MP-GS: 40 iterations. Seed=20261119; CPU, PyTorch 2.2.2, Python 3.11.13.
- Methods: frozen PhaseCNN, per-subject direct Adam phase optimization, multi-plane GS variant.
- No measurements of optical propagation in hardware, MRI tissue optical phase, or clinically meaningful anatomical reconstruction.

## User-reported results

| Method | Mean MSE | Median MSE | Mean CPU runtime (s) | Median CPU runtime (s) |
|---|---:|---:|---:|---:|
| PhaseCNN | 0.7039870797007373 | 0.6845525105794271 | 0.042854627332417294 | 0.032758656481746584 |
| Direct Adam | **0.6427790360017256** | **0.6397028466065724** | 0.7898629747333789 | 0.6127026150352322 |
| Multi-plane GS | 0.7330149169446843 | 0.7323375940322876 | 0.7566171401677738 | 0.6521345059736632 |

- CNN MSE relative to direct Adam: +9.52% (worse).
- CNN MSE relative to MP-GS: −3.96% (better).
- Direct Adam / PhaseCNN mean runtime ratio: 18.43×.
- MP-GS / PhaseCNN mean runtime ratio: 17.65×.
- These are arithmetic ratios of mean runtimes, not controlled warmed-up speedup measurements; CNN training cost excluded.
- User-reported large CNN errors include IXI-373 (CNN 1.40257 vs Adam 0.93072), IXI-337 (CNN 1.27821 vs Adam 0.84756), IXI-548 (CNN 1.04113 vs Adam 0.77349), IXI-541 (CNN 1.01963 vs Adam 0.75143).

## Valid interpretation and limitations

The CNN achieved faster **amortized inference** and lower mean MSE than the implemented MP-GS variant under these simulation settings, but did not match per-case direct Adam optimization. The GS implementation is not a reproduction of a named paper or necessarily an optimal implementation. Direct Adam optimizes using the target image available at evaluation time, while CNN weights are frozen. Across-subject variance, anatomical fidelity, optical energy efficiency, depth cross-talk, target rescaling, sensitivity to optical sampling, phase quantization, SLM fill factor and off-axis effects have not yet been characterized.

**Scientific priorities before publication:**
1. Run a paired per-subject statistical comparison using the actual local CSV; do not infer p-values from aggregate metrics.
2. Investigate numerical and anatomical error on IXI-373, IXI-337, IXI-548, IXI-541 plus good examples; use validation data only.
3. Implement a CNN-initialized fixed-iteration physics refinement and compare against matched iteration/time budgets, ensuring no gradient leakage to trained weights or held-out test.
4. Compare 40/80/160 iterations and multiple random starts for optimization methods; provide warm-up-controlled runtimes and paired uncertainty estimates.
5. Preserve 93 held-out test subjects for final protocol-frozen evaluation.
