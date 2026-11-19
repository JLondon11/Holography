# IXI T1 PhaseCNN — 88-subject validation cohort

**Status:** User-reported output of `experiments/evaluate_mri_validation.py`. Aggregate values transcribed from the terminal report supplied 2026-11-19. The raw `per_subject.csv`, full `summary.json`, trained checkpoint, and MRI scan files are on the user's Mac and were **not uploaded or independently verified** in this GitHub commit.

## Provenance
- Dataset: acquired IXI T1 MRI, 581 subjects; training 400 / validation 88 / held-out test 93.
- Evaluated cohort: **validation only**, 88 subjects.
- Model checkpoint path: `results/ixi_t1_phasecnn/best_checkpoint.pt` (local workstation).
- Model: `phase_cnn`; image size 128; optical plane distances 0.004, 0.006, 0.008 m.
- Target mapping: MRI slice intensity -> normalized desired holographic intensity; numerical angular-spectrum propagation. **No measured optical holography or direct tissue transmittance measurement.**

## Reported validation metrics

| Metric | Value |
|---|---:|
| CNN MSE mean | 0.7039870679370435 |
| CNN MSE median | 0.6845525187541985 |
| CNN MSE standard deviation | 0.16612066546945745 |
| Uniform-intensity reference MSE mean | 1.0295596009710217 |
| Uniform reference MSE median | 1.0152542729398828 |
| Relative reduction from cohort means | 0.31622504683256497 (31.62%) |
| Median subject-relative reduction | 0.3116926030748591 (31.17%) |
| Fraction of subjects better than uniform reference | 1.0 (88/88) |

## Interpretation and gates
1. The CNN outperformed the **trivial uniform-intensity reference** on all validation subjects in the reported run, using the same mean-normalized output-intensity target.
2. This is **not** a comparison with established CGH methods, and these high absolute MSEs do not demonstrate brain anatomical preservation or optical reconstruction quality.
3. The held-out test set must not be repeatedly accessed to tune architecture, preprocessing, or losses.
4. Next: archive full local model/manifest provenance, plot genuine validation reconstructions, implement matched alternating-projection / Gerchberg–Saxton and direct phase-optimization baselines, and quantify interplane crosstalk with identically defined optical sampling.
5. Report all benchmarks as numerical propagation from acquired MRI targets until validated with physical optical hardware.
