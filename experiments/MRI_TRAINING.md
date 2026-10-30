# Subject-level MRI-derived CGH model training

This project now includes a real-file training **implementation**, not a completed experiment on human MRI. There is no authorized MRI cohort in the repository. A manuscript claim requires actual licensed source files, subject identifiers, split provenance, architecture/baseline audit and repeated benchmark runs.

Install: `python -m pip install -e '.[mri]'`

Create a private manifest based on `data/manifest_example.json`, with train/validation/test subjects and independently computed SHA-256 values. Use `.nii/.nii.gz/.npy/.npz` arrays. Example:

```bash
python experiments/train_mri_subjects.py --manifest /private/mri_manifest.json \
  --out results/mri_subjects --epochs 20 --size 128 --model phase_cnn --device cpu
```

Optional unrolled model: `--model unrolled` (more expensive; may require substantial memory). The data contract rejects duplicate subject IDs and missing/mismatched file checksums. Test is scored after choosing a checkpoint from validation, and is not used for checkpoint selection.

### Critical scientific limitations

MRI anatomical image intensities are assigned to optically reconstructed intensity planes. **MRI voxel values are not complex optical transmission functions.** The model does not claim to infer tissue optical phase or measured 3D holographic brain anatomy. Slice selection and optical depth mapping are experimental choices. Energy and background normalization are approximations. PhaseCNN is not a faithful HoloNet implementation. The actual MRI dataset, pre-registration, clinical reuse permissions, optical calibration and independent reproducibility verification remain open.
