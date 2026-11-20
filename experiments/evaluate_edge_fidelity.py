"""Edge fidelity on real acquired IXI MRI validation intensity targets.

Follows the *existing* frozen PhaseCNN + 20-step optimizer; NO retraining,
no synthetic MRI data and no held-out test MRI usage. MRI intensities serve
as prescribed optical display targets, not tissue optical phase.

Edges are measured by finite-difference gradient magnitude of the reconstructed
and prescribed intensity images. A gradient-magnitude Pearson correlation is
an image-detail proxy, not a clinical or histological validation metric.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from mri_pipeline import load_volume, build_target_planes
from subject_manifest import read_manifest
from train_mri_subjects import make_model, DEPTHS, WAVELENGTH_M, PITCH_M
from compare_mri_baselines import propagate_planes, target_scaled
from inspect_phase_and_depth import refine, pearson


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gradient_magnitude(image):
    """Finite-difference 2D gradient magnitude (natural intensity units/pixel)."""
    array = np.asarray(image, dtype=np.float64)
    if array.ndim != 2 or min(array.shape) < 3 or not np.isfinite(array).all():
        raise ValueError("Expected finite two-dimensional image")
    dy, dx = np.gradient(array)
    return np.hypot(dy, dx)


def structural_metrics(predicted, target):
    """Evaluate optical intensity fidelity; do not imply clinical anatomy scores."""
    pred = np.asarray(predicted, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    if pred.shape != truth.shape:
        raise ValueError("Predicted and reference intensity shapes differ")
    p_edge, t_edge = gradient_magnitude(pred), gradient_magnitude(truth)
    edge_mae = float(np.mean(np.abs(p_edge - t_edge)))
    target_edge_mean = float(np.mean(t_edge))
    return {
        "mse": float(np.mean((pred-truth)**2)),
        "pixel_pearson": pearson(pred, truth),
        "gradient_magnitude_pearson": pearson(p_edge, t_edge),
        "gradient_magnitude_mae": edge_mae,
        "relative_gradient_mae": (
            edge_mae / target_edge_mean if target_edge_mean > 1e-10 else None
        ),
        "target_gradient_mean": target_edge_mean,
        "reconstruction_gradient_mean": float(p_edge.mean()),
    }


def run(manifest, checkpoint, out, steps=20, lr=0.05, axis=2, max_subjects=None):
    if steps < 1 or lr <= 0 or axis not in (0, 1, 2):
        raise ValueError("Invalid optimization or MRI axis parameters")
    if max_subjects is not None and max_subjects < 1:
        raise ValueError("max-subjects must be >= 1")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    selected_all = [r for r in read_manifest(manifest) if r["split"] == "validation"]
    if not selected_all:
        raise ValueError("No acquired validation MRI scans")
    selected = selected_all[:max_subjects] if max_subjects is not None else selected_all

    checkpoint = Path(checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved["model_name"] != "phase_cnn":
        raise ValueError("Expected PhaseCNN checkpoint")
    if tuple(float(z) for z in saved["depths_m"]) != tuple(DEPTHS):
        raise ValueError("Depth mismatch with checkpoint")
    for name, expected in (("wavelength_m", WAVELENGTH_M),
                           ("pixel_pitch_m", PITCH_M)):
        if not np.isclose(float(saved[name]), expected, rtol=1e-8, atol=0):
            raise ValueError("Mismatch in optical parameter: " + name)
    model = make_model("phase_cnn").cpu().eval()
    model.load_state_dict(saved["model"], strict=True)
    size = int(saved["size"])
    results = []
    for index, record in enumerate(selected, 1):
        volume, _ = load_volume(record["path"])
        planes, indices = build_target_planes(volume, axis=axis, size=size)
        target = target_scaled(planes)
        with torch.no_grad():
            cnn_phase = model(torch.from_numpy(planes)[None])[0].detach()
        final_phase = refine(cnn_phase, target, steps, lr)
        with torch.no_grad():
            original = propagate_planes(cnn_phase).numpy()
            refined = propagate_planes(final_phase).numpy()
        truth = target.numpy()
        for k in range(len(DEPTHS)):
            for name, prediction in (("phase_cnn", original[k]),
                                     ("hybrid_refined", refined[k])):
                result = structural_metrics(prediction, truth[k])
                results.append({
                    "subject_id": record["subject_id"],
                    "scan_sha256": record["sha256"],
                    "plane": k + 1,
                    "depth_m": DEPTHS[k],
                    "method": name,
                    **result
                })
        print(f"{index}/{len(selected)} {record['subject_id']} "
              f"completed edge fidelity", flush=True)

    output = Path(out)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "per_subject_plane_edges.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    groups = []
    for method in ("phase_cnn", "hybrid_refined"):
        for k, z in enumerate(DEPTHS, 1):
            subset = [r for r in results if r["method"] == method and r["plane"] == k]
            groups.append({
                "method": method,
                "depth_m": z,
                "n_subjects": len(subset),
                "mean_mse": float(np.mean([r["mse"] for r in subset])),
                "mean_gradient_magnitude_pearson": float(np.nanmean(
                    [r["gradient_magnitude_pearson"] for r in subset])),
                "median_gradient_magnitude_pearson": float(np.nanmedian(
                    [r["gradient_magnitude_pearson"] for r in subset])),
                "mean_relative_gradient_mae": float(np.nanmean(
                    [r["relative_gradient_mae"] for r in subset])),
            })
    wins = 0
    total = 0
    for record in selected:
        for k in range(1, 4):
            cnn = next(r for r in results if r["subject_id"] == record["subject_id"]
                       and r["plane"] == k and r["method"] == "phase_cnn")
            refined = next(r for r in results if r["subject_id"] == record["subject_id"]
                           and r["plane"] == k and r["method"] == "hybrid_refined")
            c = cnn["gradient_magnitude_pearson"]
            h = refined["gradient_magnitude_pearson"]
            if np.isfinite(c) and np.isfinite(h):
                total += 1
                wins += int(h > c)
    report = {
        "status": "acquired IXI MRI validation, simulated CGH structural fidelity",
        "n_subjects": len(selected),
        "validation_total": len(selected_all),
        "pilot_subset": len(selected) != len(selected_all),
        "test_set_accessed": False,
        "checkpoint_sha256": sha256(checkpoint),
        "manifest_sha256": sha256(manifest),
        "steps": steps,
        "learning_rate": lr,
        "edge_correlation_improved_planes": wins,
        "edge_correlation_evaluable_planes": total,
        "method_depth_metrics": groups,
        "limitations": [
            "Gradient magnitude correlation is an image structure proxy, not quantified clinical tissue anatomy.",
            "The model was selected on the same validation cohort; no independent test result.",
            "No synthetic MRI inputs; all reconstructed optical fields are numerical simulations.",
            "No measured optical cross-talk, SLM calibration, or optical bench validation."
        ],
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--axis", type=int, default=2)
    parser.add_argument("--max-subjects", type=int)
    args = parser.parse_args()
    run(args.manifest, args.checkpoint, args.out, args.steps,
        args.lr, args.axis, args.max_subjects)
