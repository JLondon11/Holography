# MRI input to simulated CGH

MRI data source is not yet available in this repository. Do not interpret brain-shaped synthetic inputs as actual MRI. This code can read `.npy`, `.npz` (key `volume`), and `.nii[.gz]` (requires optional nibabel). Use appropriately licensed, deidentified MRI and record voxel spacing and orientation before making anatomical claims.

## Run
```bash
python -m pip install -e '.[deep,test]' scipy matplotlib nibabel
python experiments/optimize_mri_volume.py --input /path/to/brain.nii.gz --out results/brain_mri --size 128 --steps 100
```

This solves **one** phase-only mask numerically. It is not deep-learning training. Output includes phase, target slices, reconstructed propagated intensities, loss trace, original and optical metadata and an input SHA-256. MRI intensities are used as illustrative *intensity targets*, not inferred physical transmittance/phase.

### Local verification
A synthetic 32x32x24 volume was used strictly as a software test, output size 32 and 8 Adam steps, CPU. Initial optimization loss 2.526048; final 1.462883. MRI loader unit tests passed 3/3. These values are not evidence of anatomical or optical fidelity.

### Next scientific gates
1. Obtain licensed, named real MRI scans; record subject-level splits and preprocessing.
2. Define tissue/SLM optical mapping and physical energy constraints.
3. Train CNN/unrolled network across independent subjects and compare against per-volume optimization on held-out subjects.
4. Calibrate or validate against optical bench measurements before claims of experimentally reconstructed holograms.
