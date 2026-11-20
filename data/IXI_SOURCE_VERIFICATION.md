# Verified real-world IXI dataset source — 2026-10-31

Official upstream source: https://brain-development.org/ixi-dataset/ .

Verified against its published index:
- Approximately 600 MRI subjects were scanned at Guy's Hospital, Hammersmith Hospital, and the Institute of Psychiatry.
- NIfTI T1, T2, PD, MRA, and diffusion MRI are advertised.
- The upstream page states Creative Commons CC BY-SA 3.0 and requests acknowledgment.
- Original IXI T1 filenames follow `IXI002-Guys-0828-T1.nii.gz`, `IXI012-HH-1211-T1.nii.gz`, etc. The ingestion parser has been corrected to require the `IXI###` prefix **without a hyphen between IXI and the subject number**.
- Reader must inspect each real NIfTI affine, voxel spacing, voxel values, and SHA-256 hash.

**Critical status:** No upstream IXI archive was downloaded, no actual MRI file was validated, and no network-driven model training occurred as part of this change. Running tests of filename syntax is not an imaging experiment.

## Acquisition gate
Official dataset host links the archive from its T1 "all images" entry. Transfer the actual licensed source archive to governed storage; keep the original archive checksum and source URL. After extraction, run `data/build_ixi_manifest.py` and separately verify that the manifest reflects actual acquired files rather than user-supplied provenance claims. Train only after real scan verification and review of dataset redistribution restrictions. Never publish direct patient identifiers or identifiable raw volumes.

A declared `origin: real_acquisition` string alone does not prove a file is a real scan. Independent source-of-record verification is required before publication.
