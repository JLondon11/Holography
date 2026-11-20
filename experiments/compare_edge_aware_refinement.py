"""Paired edge-aware versus original intensity-only hybrid CGH refinement.

Only acquired IXI T1 *validation* MRI scans. Same frozen CNN, optical model,
20 Adam updates, learning rate and starting phase for both methods.
No network retraining or use of test subjects. Propagation is simulated,
not an optical-bench measurement or physical tissue optical model.

Edge-aware objective: MSE + alpha * mean(abs(gradmag(I)-gradmag(target)))
 / mean(gradmag(target)). Gradmag uses centered interior finite differences.
The control is EXACTLY the original intensity-only loss at alpha=0.
Use a pilot, then all 88 validation subjects; do not claim better results
before execution.
"""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from mri_pipeline import build_target_planes, load_volume
from subject_manifest import read_manifest
from train_mri_subjects import make_model, DEPTHS, WAVELENGTH_M, PITCH_M
from compare_mri_baselines import propagate_planes, target_scaled
from evaluate_edge_fidelity import structural_metrics


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gradient_magnitude_torch(x):
    """Spatial gradient magnitude, preserving autograd on real intensities."""
    if x.ndim != 3 or min(x.shape[-2:]) < 3:
        raise ValueError("Expected [planes,H,W] images with H,W>=3")
    dx = (x[:, 1:-1, 2:] - x[:, 1:-1, :-2]) * 0.5
    dy = (x[:, 2:, 1:-1] - x[:, :-2, 1:-1]) * 0.5
    return torch.sqrt(dx.square() + dy.square() + 1e-12)


def optimize(initial_phase, target, steps, lr, edge_weight):
    if edge_weight < 0 or not np.isfinite(edge_weight):
        raise ValueError("Edge weight must be a finite nonnegative number")
    variable = torch.nn.Parameter(initial_phase.detach().clone())
    optimizer = torch.optim.Adam([variable], lr=lr)
    target_grad = gradient_magnitude_torch(target).detach()
    normalizer = target_grad.mean().clamp_min(1e-6)
    start = time.perf_counter()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        intensity = propagate_planes(variable)
        mse = (intensity - target).square().mean()
        if edge_weight:
            edge = (gradient_magnitude_torch(intensity) - target_grad).abs().mean()
            loss = mse + edge_weight * (edge / normalizer)
        else:
            loss = mse
        if not torch.isfinite(loss).item():
            raise RuntimeError("Nonfinite optical reconstruction loss")
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        output = propagate_planes(variable).detach().cpu().numpy()
    elapsed = time.perf_counter() - start
    return output, elapsed


def run(manifest, checkpoint, out, alpha=0.1, steps=20, lr=0.05,
        axis=2, max_subjects=None):
    if not (np.isfinite(alpha) and alpha > 0 and np.isfinite(lr) and lr > 0):
        raise ValueError("alpha and learning rate must be finite and positive")
    if steps < 1 or axis not in (0, 1, 2) or (
            max_subjects is not None and max_subjects < 1):
        raise ValueError("Invalid steps, axis or max-subjects")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    records = [r for r in read_manifest(manifest) if r["split"] == "validation"]
    if not records:
        raise ValueError("No acquired MRI validation subjects")
    chosen = records[:max_subjects] if max_subjects is not None else records
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved["model_name"] != "phase_cnn":
        raise ValueError("Expected frozen PhaseCNN checkpoint")
    if tuple(float(x) for x in saved["depths_m"]) != tuple(DEPTHS):
        raise ValueError("Optical depth mismatch")
    for key, expected in (("wavelength_m", WAVELENGTH_M),
                          ("pixel_pitch_m", PITCH_M)):
        if not np.isclose(float(saved[key]), expected, rtol=1e-8, atol=0):
            raise ValueError("Optical mismatch: " + key)
    size = int(saved["size"])
    model = make_model("phase_cnn").cpu().eval()
    model.load_state_dict(saved["model"], strict=True)
    measurements = []
    for index, record in enumerate(chosen, 1):
        volume, _ = load_volume(record["path"])
        raw, _ = build_target_planes(volume, axis=axis, size=size)
        target = target_scaled(raw)
        with torch.no_grad():
            cnn_start = time.perf_counter()
            phase = model(torch.from_numpy(raw)[None])[0].detach()
            inference_seconds = time.perf_counter() - cnn_start
        source = target.numpy()
        predictions = {}
        for method, weight in (("intensity_only", 0.0),
                               ("edge_aware", alpha)):
            output, opt_seconds = optimize(phase, target, steps, lr, weight)
            predictions[method] = output
            for k, depth in enumerate(DEPTHS):
                metrics = structural_metrics(output[k], source[k])
                measurements.append({
                    "subject_id": record["subject_id"],
                    "scan_sha256": record["sha256"],
                    "method": method,
                    "plane": k + 1,
                    "depth_m": depth,
                    "inference_plus_optimization_seconds": inference_seconds + opt_seconds,
                    **metrics,
                })
        baseline_mse = float(np.mean((predictions["intensity_only"] - source)**2))
        candidate_mse = float(np.mean((predictions["edge_aware"] - source)**2))
        print(f"{index}/{len(chosen)} {record['subject_id']} "
              f"control_mse={baseline_mse:.5f} edge_mse={candidate_mse:.5f}",
              flush=True)

    output_dir = Path(out)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "per_subject_plane_methods.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(measurements[0]))
        writer.writeheader()
        writer.writerows(measurements)

    summary = []
    for method in ("intensity_only", "edge_aware"):
        for plane, depth in enumerate(DEPTHS, 1):
            matched = [m for m in measurements
                       if m["method"] == method and m["plane"] == plane]
            summary.append({
                "method": method,
                "plane": plane,
                "depth_m": depth,
                "subjects": len(matched),
                "mean_mse": float(np.mean([m["mse"] for m in matched])),
                "mean_gradient_magnitude_pearson": float(np.mean(
                    [m["gradient_magnitude_pearson"] for m in matched])),
                "mean_relative_gradient_mae": float(np.mean(
                    [m["relative_gradient_mae"] for m in matched])),
                "mean_cpu_seconds": float(np.mean(
                    [m["inference_plus_optimization_seconds"] for m in matched]))
            })
    pairs = {}
    for m in measurements:
        pairs[(m["subject_id"], m["plane"], m["method"])] = m
    edge_corr_wins = 0
    intensity_mse_wins = 0
    edge_mae_wins = 0
    worst_relative_mse_increase = -float("inf")
    for row in chosen:
        for plane in range(1, len(DEPTHS) + 1):
            a = pairs[(row["subject_id"], plane, "intensity_only")]
            b = pairs[(row["subject_id"], plane, "edge_aware")]
            edge_corr_wins += int(b["gradient_magnitude_pearson"] > a["gradient_magnitude_pearson"])
            intensity_mse_wins += int(b["mse"] < a["mse"])
            edge_mae_wins += int(b["relative_gradient_mae"] < a["relative_gradient_mae"])
            worst_relative_mse_increase = max(
                worst_relative_mse_increase,
                (b["mse"] - a["mse"]) / max(a["mse"], 1e-12))
    all_control = [m for m in measurements if m["method"] == "intensity_only"]
    all_candidate = [m for m in measurements if m["method"] == "edge_aware"]
    control_mse = float(np.mean([m["mse"] for m in all_control]))
    candidate_mse = float(np.mean([m["mse"] for m in all_candidate]))
    report = {
        "status": "real acquired IXI validation MRI, simulated phase-only CGH",
        "n_subjects": len(chosen),
        "validation_total": len(records),
        "pilot_subset": len(chosen) != len(records),
        "test_set_accessed_by_this_script": False,
        "note_test_history": "The original training script previously reported a test MSE; the test cohort is not strictly untouched.",
        "checkpoint_sha256": sha256(checkpoint),
        "manifest_sha256": sha256(manifest),
        "optical_depths_m": list(DEPTHS),
        "wavelength_m": WAVELENGTH_M,
        "pitch_m": PITCH_M,
        "size": size,
        "optimizer_steps": steps,
        "learning_rate": lr,
        "edge_weight": alpha,
        "loss": "intensity_MSE + edge_weight * gradient_magnitude_MAE / target_gradient_mean",
        "method_depth_metrics": summary,
        "paired_plane_comparisons": {
            "evaluable_planes": len(chosen) * 3,
            "edge_correlation_wins": edge_corr_wins,
            "relative_gradient_mae_wins": edge_mae_wins,
            "intensity_mse_wins": intensity_mse_wins,
            "worst_relative_intensity_mse_increase": worst_relative_mse_increase,
            "control_cohort_mean_mse": control_mse,
            "edge_aware_cohort_mean_mse": candidate_mse,
            "relative_cohort_mse_change": candidate_mse / control_mse - 1.0,
        },
        "limitations": [
            "One candidate edge-loss weight tested; it is a validation experiment, not proven superiority.",
            "Intensity-only control and candidate share CNN phase initialization and Adam iteration budget.",
            "Gradient magnitude correlation is not a clinical anatomical accuracy metric.",
            "Simulated optics from real acquired MRI intensity targets, not physical optical measurements.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--axis", type=int, default=2)
    p.add_argument("--max-subjects", type=int)
    args = p.parse_args()
    run(args.manifest, args.checkpoint, args.out, args.alpha,
        args.steps, args.lr, args.axis, args.max_subjects)
