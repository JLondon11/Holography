# IXI real-MRI acquisition and validation

**Source of record:** https://brain-development.org/ixi-dataset/

Official IXI data consist of approximately 600 acquired MR image sets from three London hospitals, licensed CC BY-SA 3.0. For initial holography experiments use the official **T1 images (all images)** NIfTI archive, not synthetic approximations or AI-generated MRI.

1. Download the T1 archive from the official IXI page onto authorized local research storage. Keep its original archive intact and record its checksum, retrieval date, license and source.
2. Extract only into a private research data directory; preserve filenames and do not commit scan binaries or identifiable metadata to public GitHub.
3. Install real MRI dependencies: `python -m pip install -e '.[mri]'`.
4. Build a verified subject-level manifest:
   `python data/build_ixi_manifest.py --input-dir '/private/IXI-T1' --output '/private/ixi_t1_manifest.json'`
5. Confirm the manifest includes all three non-overlapping splits and real acquisition-site records. Check subject counts and metadata against the archive.
6. Train the physics-based CNN only after source and usage authorization are reviewed:
   `python experiments/train_mri_subjects.py --manifest '/private/ixi_t1_manifest.json' --out results/ixi_t1_phasecnn --epochs 20 --size 128 --model phase_cnn`.
7. Repeat under independent seeds, collect pixel-level and depth-resolved metrics, assess optical model sensitivity, compare to established baselines, then generate journal figures.

**Important:** These commands do not assert the scan archive has been downloaded or models have been trained. The study's real-MRI claims begin only after the source data are acquired and checked.

**Caveat:** The acquired MRI intensities define desired optical output patterns. This is not direct measured optical transmittance of human brain tissue. Camera/SLM experiments must be independently labeled and verified.
