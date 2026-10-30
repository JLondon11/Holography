"""Explicit, auditable reconstruction metrics for real-valued intensity arrays."""
from __future__ import annotations
import numpy as np

def reconstruction_metrics(reference, estimate, peak: float = 1.0) -> dict:
    """Return MSE, RMSE and PSNR on the *same* samples and intensity scale.

    peak is the data-range peak used for PSNR, not estimated implicitly.
    NaN and mismatched array shapes raise exceptions. PSNR is infinite for
    zero MSE; no per-image normalization or clipping is performed.
    """
    a, b = np.asarray(reference, dtype=np.float64), np.asarray(estimate, dtype=np.float64)
    if a.shape != b.shape or not a.size or a.ndim == 0:
        raise ValueError("reference and estimate must have matching nonempty array shapes")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("arrays must contain finite data")
    if peak <= 0 or not np.isfinite(peak):
        raise ValueError("peak must be finite and positive")
    mse = float(np.mean(np.square(a - b)))
    rmse = float(np.sqrt(mse))
    psnr = float("inf") if mse == 0 else float(10 * np.log10(peak**2 / mse))
    return {"mse": mse, "rmse": rmse, "psnr_db": psnr, "peak": float(peak)}
