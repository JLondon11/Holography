# Real IXI MRI acquisition on macOS

Source: https://brain-development.org/ixi-dataset/
Official T1 archive: https://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-T1.tar

Use a workstation with enough storage for both the archive and extracted volumes. The acquisition script intentionally makes **no synthetic data**. It retains the source tar, calculates its SHA-256 and rejects unsafe archive entries.

```bash
cd /path/to/Holography
python -m pip install -e '.[mri]'

python data/acquire_ixi_t1.py \
  --output-dir '/Volumes/BioMedAI Data Lake/IXI-T1' \
  --download --extract

python data/build_ixi_manifest.py \
  --input-dir '/Volumes/BioMedAI Data Lake/IXI-T1/nifti' \
  --output '/Volumes/BioMedAI Data Lake/IXI-T1/ixi_t1_manifest.json'

python experiments/train_mri_subjects.py \
  --manifest '/Volumes/BioMedAI Data Lake/IXI-T1/ixi_t1_manifest.json' \
  --out results/ixi_t1_phasecnn --epochs 20 --size 128 --model phase_cnn
```

The last command should only be run after reviewing source provenance, manifest contents, sufficient disk space, and study permissions. This training uses **real MRI-derived intensity targets** but a simulated optical propagator. It does **not** create an optically measured hologram.

The archive is external research data, not committed to public GitHub. Publication figures must be generated from verified experiment results, not from earlier phantom demonstrations. If the official host is down, stop and record the failure rather than switching silently to an unverified mirror.

Note: source archive checksums generated locally demonstrate file integrity after acquisition; without a published upstream digest, they do not establish cryptographic authenticity.
