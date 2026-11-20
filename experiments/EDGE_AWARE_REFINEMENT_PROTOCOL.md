# Edge-aware hologram refinement — reproducible IXI MRI validation experiment

The **primary control** is the existing PhaseCNN-initialized 20-step Adam
intensity-only optimization (frozen checkpoint). The **experimental candidate**
starts from exactly the same PhaseCNN phase but optimizes
\`MSE + alpha × normalized_gradient_magnitude_MAE\`, with \`alpha=0.1\`.
Both methods use a single phase-only SLM plane, 532 nm light, pitch 3.74 µm,
and three simulated propagation depths: 4, 6, and 8 mm.

**Scientific status:** These are computational display-target experiments
derived from 581 real IXI acquired T1 MRI volumes. The 88 validation subjects
are used here. The output optical fields are numerical simulations, not an
optical bench experiment or direct coherent measurements of brain tissue.

## Run from Intel Mac

\`\`\`bash
cd ~/Desktop/Holography
conda activate biomedai-medmnist
git pull --ff-only origin research/ieee-holography-v1

# Implementation smoke/pilot on 3 real validation MRI volumes:
python experiments/compare_edge_aware_refinement.py \
  --manifest '/Volumes/BioMedAI Data Lake/IXI-T1/ixi_t1_manifest.json' \
  --checkpoint results/ixi_t1_phasecnn/best_checkpoint.pt \
  --out results/ixi_t1_phasecnn/edge_aware_pilot \
  --alpha 0.1 --steps 20 --max-subjects 3

# Full 88 validation subjects if pilot is successful:
python experiments/compare_edge_aware_refinement.py \
  --manifest '/Volumes/BioMedAI Data Lake/IXI-T1/ixi_t1_manifest.json' \
  --checkpoint results/ixi_t1_phasecnn/best_checkpoint.pt \
  --out results/ixi_t1_phasecnn/edge_aware_full \
  --alpha 0.1 --steps 20
\`\`\`

Outputs are \`summary.json\` and \`per_subject_plane_methods.csv\`, both
saved locally, with paired per-plane comparisons.

## Diagnostics

- Whether mean MSE **increases** relative to intensity-only control, and by how much.
- Whether gradient-magnitude correlation improves, in how many of 264 plane
  comparisons, and whether gradient MAE improves.
- Worst per-plane proportional increase in intensity MSE.
- Comparison of per-subject CPU elapsed inference + refinement times.
- Results at 4, 6, and 8 mm separately, not merely averaged.

A higher edge correlation alone is **insufficient** to prefer the candidate.
Our validation-only decision criterion is to require a meaningful improvement
in structural fidelity *without an unacceptable intensity-MSE penalty*,
and to inspect actual acquired MRI-derived reconstruction images.

The edge-loss gradient calculation uses centered finite differences on the
interior pixels; the **reported** gradient diagnostics reuse the previously
implemented NumPy full-image gradient magnitude. Both variants evaluate
identical reported metrics.

The new candidate is not trained as another network. It is a separate
per-subject numerical optical optimizer, not faster amortized inference.

## Data integrity and limitations

The 93 test subjects are **not touched by this script**. However, an initial
PhaseCNN training run previously evaluated its test MSE once, so that set
cannot honestly be described as completely untouched. A fully independent
acquired-MRI external cohort would be preferable for unbiased final claims.

Never report simulated optical fields as physical hologram measurements, and
never infer clinical anatomical fidelity from a simple gradient statistic.
Prior art must be reviewed before claiming novelty for CNN warm-start
phase optimization or edge-aware losses.
