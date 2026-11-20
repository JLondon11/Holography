# Source Table I numerical consistency audit

Source: *Fourier Optics and Deep Learning Methods for Fast 3D Reconstruction in Digital Holography*, arXiv:2508.06703v1, p. 14.

Audit script: `python audit/audit_table1.py`.

Results from locally executed audit on 2026-10-30:
- 14 reported algorithm configurations, each evaluated with and without filtering (28 metric sets).
- With a 0.001 absolute tolerance between reported RMSE and sqrt(reported MSE), 2 flagged values:
  1. AP / SP-3D-AS-CH / filtered: MSE 0.009, RMSE 0.0304; sqrt(MSE) = 0.094868.
  2. SGD / G-3D-AS-CH / filtered: MSE 0.0637, RMSE 0.2510; sqrt(MSE) = 0.252389.
- The first is substantial; the second may reflect transcription, rounding, or differences between evaluation domains.
- PSNR is **not** assessed as an inconsistency without a declared peak, normalization and ROI. Script computes implied peak for each row as a diagnostic.

Local regression checks for the standalone audit script: 3 tests passed. GitHub CI results remain to be inspected.

**Do not modify historical values without original reconstruction arrays or experimental logs.**
