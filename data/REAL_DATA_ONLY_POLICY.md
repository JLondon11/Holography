# Real-data-only policy — effective 2026-10-30

The active holography research branch must use **empirical, acquired data** for all model training, validation, testing and publication-quality anatomical figures.

- **Excluded:** synthetic brain phantoms, procedurally generated anatomy, random-image training datasets, generated MRI substitutes, and simulations described as empirical measurements.
- **Permitted:** numerically propagated optical reconstructions **computed from independently acquired, real MRI**. Numerical physics simulation is still needed to generate a computer-generated hologram, but it is not an optical-bench measurement.
- **Unit tests:** simple numerical constants or arrays may verify software/math identities; these are neither research datasets nor publishable experiments.
- **Provenance gate:** dataset manifests must state real acquisition, source URL, data license, citation, real subject IDs, disjoint train/validation/test partitions and input checksums. These are declarations; custodial review of the dataset must still verify their truth.
- **Privacy:** only publicly licensed, deidentified anatomical volumes may be distributed. Never commit raw private clinical scans or direct patient identifiers.
- **Reporting:** MRI image intensity is not a measured optical phase/scattering field; numerical SLM reconstructions must be labeled 'simulated optical propagation from acquired MRI target patterns'.
- **Historical preservation:** synthetic examples created in earlier project iterations remain only in Git history; they are excluded from current benchmarks, tables, figures and paper claims. No synthetic training/demo entrypoints remain in the active branch.
- **Training gate:** if there are no verified real MRI volumes locally accessible, do not initiate training or invent data/metrics.

## Candidate empirical dataset

IXI: https://brain-development.org/ixi-dataset/ — approximately 600 acquired MR scans from three London hospitals, NIfTI, CC BY-SA 3.0 as listed by the dataset host. Verify license and dataset-specific conditions before retrieval. Maintain subject-level separation and record the original source and hashes. Data is NOT presently imported or audited.

Alternative: OpenNeuro, https://openneuro.org/ — select a specific released dataset and confirm its license and citation before use.

## Next run gate

1. Acquire a real licensed T1-weighted MRI cohort.
2. Check dataset contents and permission; hash every file; publish nonidentifying manifest.
3. Run `experiments/train_mri_subjects.py` on actual acquired images.
4. Evaluate against empirical held-out MRI subjects; preserve original source commits and metadata.
5. Only then generate paper figures and tables from logged experiments.
