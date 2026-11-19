"""Evaluate one frozen checkpoint on the complete real-MRI validation cohort.

No training, no synthetic data, no held-out test access.
Metrics use the exact mean-normalized intensity objective of train_mri_subjects.py.
Run after installing the project and MRI dependencies:
  python experiments/evaluate_mri_validation.py \
    --manifest '/Volumes/BioMedAI Data Lake/IXI-T1/ixi_t1_manifest.json' \
    --checkpoint results/ixi_t1_phasecnn/best_checkpoint.pt \
    --out results/ixi_t1_phasecnn/validation_cohort
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from mri_pipeline import build_target_planes, load_volume
from subject_manifest import read_manifest
from train_mri_subjects import make_model, reconstructed_intensity, DEPTHS, PITCH_M, WAVELENGTH_M


def evaluate(manifest, checkpoint, out, axis=2):
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1 or 2")
    records = [row for row in read_manifest(manifest) if row["split"] == "validation"]
    if not records:
        raise ValueError("No validation subjects in verified manifest")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if ckpt.get("model_name") not in ("phase_cnn", "unrolled"):
        raise ValueError("Unsupported checkpoint")
    if tuple(float(z) for z in ckpt["depths_m"]) != tuple(DEPTHS):
        raise ValueError("Checkpoint depth mismatch")
    if not np.isclose(float(ckpt["pixel_pitch_m"]), PITCH_M) or not np.isclose(
            float(ckpt["wavelength_m"]), WAVELENGTH_M):
        raise ValueError("Checkpoint optical parameters mismatch")
    model = make_model(ckpt["model_name"])
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    n = int(ckpt["size"])
    results = []
    for row in records:
        volume, spacing = load_volume(row["path"])
        planes, indices = build_target_planes(volume, axis=axis, size=n)
        x = torch.from_numpy(planes)[None]
        # The unrolled network needs autograd internally even at evaluation.
        with torch.enable_grad():
            reconstructed, _ = reconstructed_intensity(model, x)
        pred = reconstructed[0].detach().cpu().numpy().astype(np.float64)
        target = planes.astype(np.float64)
        target = target / np.maximum(target.mean(axis=(-2, -1), keepdims=True), 1e-5)
        model_plane = ((pred - target)**2).mean(axis=(1, 2))
        flat_plane = ((np.ones_like(target) - target)**2).mean(axis=(1, 2))
        model_mse = float(model_plane.mean())
        flat_mse = float(flat_plane.mean())
        results.append({
            "subject_id": row["subject_id"],
            "scan_sha256": row["sha256"],
            "model_mse": model_mse,
            "flat_reference_mse": flat_mse,
            "relative_mse_reduction": float((flat_mse - model_mse) / flat_mse) if flat_mse > 0 else None,
            "model_better_than_flat": bool(model_mse < flat_mse),
            "model_plane_mse": [float(v) for v in model_plane],
            "flat_plane_mse": [float(v) for v in flat_plane],
            "selected_indices": indices,
            "voxel_spacing_mm": spacing
        })

    output = Path(out)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "per_subject.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "subject_id", "scan_sha256", "model_mse", "flat_reference_mse",
            "relative_mse_reduction", "model_better_than_flat"])
        writer.writeheader()
        for row in results:
            writer.writerow({k: row[k] for k in writer.fieldnames})

    models = np.asarray([r["model_mse"] for r in results], dtype=np.float64)
    flats = np.asarray([r["flat_reference_mse"] for r in results], dtype=np.float64)
    reduction = (flats - models) / np.maximum(flats, 1e-12)
    summary = {
        "status": "real IXI validation MRI, simulated optical propagation",
        "cohort": "validation_only",
        "n_subjects": len(records),
        "checkpoint": str(checkpoint),
        "model_name": ckpt["model_name"],
        "size": n,
        "depths_m": list(DEPTHS),
        "model_mse_mean": float(models.mean()),
        "model_mse_median": float(np.median(models)),
        "model_mse_std": float(models.std(ddof=1)) if len(models)>1 else None,
        "uniform_mse_mean": float(flats.mean()),
        "uniform_mse_median": float(np.median(flats)),
        "relative_reduction_from_cohort_means": float((flats.mean()-models.mean()) / flats.mean()),
        "median_subject_relative_reduction": float(np.median(reduction)),
        "fraction_subjects_better_than_uniform": float((models < flats).mean()),
        "per_subject": results,
        "limitations": "Incoherent MRI intensity targets are prescribed for a simulated coherent optical display; not a tissue optical transmission model or measured hologram. Uniform baseline is trivial."
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k!="per_subject"}, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--axis", type=int, default=2)
    args = parser.parse_args()
    evaluate(args.manifest, args.checkpoint, args.out, axis=args.axis)
