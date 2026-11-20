# IXI real MRI validation: native-similarity and edge-fidelity diagnostics

The **frozen** PhaseCNN plus 20-step physics refinement has achieved strong
validation-cohort MSE improvements. The present diagnostics investigate whether
the matched-depth correlation advantage is partly due to native anatomical
similarity and whether reconstructed intensity preserves anatomical edges.

All MRI input data are acquired real IXI T1 scans. Simulated coherent-wave
propagation is **not** an optical bench experiment, a model of MRI tissue's
optical scattering phase, or clinical fidelity validation.

## Intel Mac commands

Run from project root in conda environment \`biomedai-medmnist\`:

\`\`\`bash
cd ~/Desktop/Holography
conda activate biomedai-medmnist
git pull --ff-only origin research/ieee-holography-v1

# Pure postprocessing; no new MRI processing needed:
python experiments/analyze_native_adjusted_depth.py \
  --csv results/ixi_t1_phasecnn/depth_cohort/depth_matrix_long.csv \
  --out results/ixi_t1_phasecnn/native_adjusted_depth

# Real 3D MRI files read and phase optimization repeated on validation subjects:
python experiments/evaluate_edge_fidelity.py \
  --manifest '/Volumes/BioMedAI Data Lake/IXI-T1/ixi_t1_manifest.json' \
  --checkpoint results/ixi_t1_phasecnn/best_checkpoint.pt \
  --out results/ixi_t1_phasecnn/edge_fidelity \
  --steps 20
\`\`\`

Outputs:
- \`native_adjusted_depth/per_plane_native_adjusted.csv\`
- \`native_adjusted_depth/summary.json\`
- \`edge_fidelity/per_subject_plane_edges.csv\`
- \`edge_fidelity/summary.json\`

## Metric definitions and limits

For a reconstructed depth plane \`i\` and MRI-derived target intensity plane
\`j\`, let \`r_ij\` be the Pearson correlation, and let \`t_ij\` be
the native MRI target-to-target Pearson correlation.

- Raw matching margin: \`r_ii - max(r_ij for j!=i)\`.
- Native target contrast: \`1 - max(t_ij for j!=i)\`.
- Exploratory native-adjusted matching margin:
  \`raw_matching_margin / native_target_contrast\`, omitting instances where
  native target contrast is below 0.05. The ratio can exceed one and MUST NOT
  be interpreted as a probability or as calibrated optical crosstalk.
- Gradient magnitude is from 2-D finite differences of **intensity**.
  The evaluator reports correlation between reconstructed and target gradient
  magnitudes, alongside a relative gradient-magnitude mean absolute error.
  Both are descriptive image-structure metrics, not clinically relevant
  boundary/lesion localization scores.

Use these diagnostics to understand optical reconstruction limitations, not
to optimize to the validation cohort indefinitely. The held-out 93 subjects
are reserved for the final prespecified evaluation. Optical phase patterns
must be interpreted separately from reconstructed intensity.
