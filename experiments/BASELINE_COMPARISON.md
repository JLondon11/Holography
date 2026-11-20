# Real IXI MRI — matched optical baseline benchmark

## Scope and limitations

The benchmark runs **only** on real acquired IXI T1 MRI validation subjects from the checksum-verified manifest. It compares a frozen pretrained PhaseCNN to (a) direct per-subject Adam phase optimization and (b) a simultaneous multi-plane Gerchberg-Saxton-style phase projection implementation. All three use the same wavelength, SLM pitch, depth planes, resolution, and mean-normalized target-intensity MSE.

No synthetic brain images, phantom MRI, training on test subjects, measured optical system, or generated clinical content are involved. Numerical propagation is simulation of a display wavefront, not a measurement of actual brain optical phase.

## Run from the project root on Intel macOS (CPU)

```bash
cd ~/Desktop/Holography
conda activate biomedai-medmnist
git pull --ff-only origin research/ieee-holography-v1
python -m pip install -e '.[mri]'

# Initial pilot on 3 acquired validation subjects, to check runtime and code paths:
python experiments/compare_mri_baselines.py \
  --manifest '/Volumes/BioMedAI Data Lake/IXI-T1/ixi_t1_manifest.json' \
  --checkpoint results/ixi_t1_phasecnn/best_checkpoint.pt \
  --out results/ixi_t1_phasecnn/baselines_pilot \
  --iterations 40 --max-subjects 3

# Complete prespecified 88-subject validation benchmark:
python experiments/compare_mri_baselines.py \
  --manifest '/Volumes/BioMedAI Data Lake/IXI-T1/ixi_t1_manifest.json' \
  --checkpoint results/ixi_t1_phasecnn/best_checkpoint.pt \
  --out results/ixi_t1_phasecnn/baselines_full \
  --iterations 40
```

Outputs:
- `per_subject_methods.csv`: subject, method, scan checksum, three per-plane errors, average MSE, CPU elapsed time.
- `summary.json`: complete-cohort summary, model and manifest SHA-256, algorithm parameters, runtime and methodological limitations.
- Console: a progress line after each verified MRI volume.

**Do not report metrics before running the experiment.** The model is already trained; Adam and GS optimize each target independently at evaluation time. Their per-subject optimization cost is included in wall-clock timing, whereas PhaseCNN training cost is not. This is an inference-latency comparison, not an end-to-end compute-cost comparison. No hyperparameters should be tuned on the held-out test cohort.

The GS implementation is an identifiable multi-plane GS variant, **not** a claimed reproduction of any specific paper's method. Validation on mathematical reference propagation, harmonized initialization, alternative budgets, crosstalk and physically accurate SLM constraints is needed before submitting to IEEE.
