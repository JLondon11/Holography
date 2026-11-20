# Real brain MRI acquisition and research protocol

The original manuscript references brain MRI slices and point-cloud brain data, but the exact source dataset, licenses, identities, pre-processing transforms and split assignments are not currently provided. Do NOT claim that the synthetic brain phantom is the study's MRI.

Candidate open data: [open-brain](https://github.com/Losses/open-brain), an MRI-derived openly licensed research dataset, subject to reviewing each included file and usage conditions; and [Dryad ex-vivo 7T MRI](https://doi.org/10.5061/dryad.119f80q), CC0 as listed in the dataset's repository description.

For actual 3D hologram research: preserve voxel spacing and orientation; document per-volume intensity normalization; define a defensible real anatomy -> complex optical field mapping; keep subject-level train/validation/test splits; record permissions and checksums; and avoid depositing identifiable clinical MRI into this public repository.

Do not replace original source data silently. Select physical target-plane constraints, normalize optical power, include out-of-plane leakage metrics, and report all computational and physical limitations.
