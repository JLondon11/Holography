"""Cohort-level depth fidelity for acquired IXI MRI-derived holographic targets.

This is numerical coherent-wave display simulation, not measured optical data.
Only checksum-validated acquired IXI *validation* MRI subjects are processed.
The 93 test subjects are never loaded. A frozen CNN and fixed 20-step
physics-only refinement are evaluated; no network training is performed.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from mri_pipeline import build_target_planes, load_volume
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


def corr_matrix(images, targets):
    if images.shape != targets.shape or images.ndim != 3:
        raise ValueError("Expected matched [3, H, W] images")
    return np.array([[pearson(images[i], targets[j])
                      for j in range(images.shape[0])]
                     for i in range(images.shape[0])], dtype=np.float64)


def depth_metrics(pred, target):
    """Return descriptive matched-depth correlations, not physical crosstalk."""
    matrix = corr_matrix(pred, target)
    native = corr_matrix(target, target)
    margins = []
    correct = []
    ambiguous = []
    for i in range(matrix.shape[0]):
        diag = matrix[i, i]
        off = [matrix[i, j] for j in range(matrix.shape[1]) if j != i]
        if not np.isfinite(diag) or not np.isfinite(off).all():
            margins.append(float("nan"))
            correct.append(None)
            ambiguous.append(None)
            continue
        margins.append(float(diag - max(off)))
        correct.append(bool(diag > max(off)))
        # If target slices are highly mutually correlated, a depth correlation
        # margin by itself may not imply optical axial selectivity.
        ambiguous.append(bool(np.max([native[i,j] for j in range(3) if j != i]) > 0.9))
    return matrix, native, margins, correct, ambiguous


def run(manifest, checkpoint, out, steps=20, lr=0.05, axis=2, max_subjects=None):
    if steps < 1 or lr <= 0 or axis not in (0, 1, 2):
        raise ValueError("Invalid steps/lr/axis")
    if max_subjects is not None and max_subjects < 1:
        raise ValueError("max-subjects must be positive")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    validation = [r for r in read_manifest(manifest) if r["split"] == "validation"]
    if not validation:
        raise ValueError("No validation records")
    selected = validation[:max_subjects] if max_subjects else validation

    checkpoint = Path(checkpoint)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if ckpt["model_name"] != "phase_cnn":
        raise ValueError("Expected frozen PhaseCNN")
    if tuple(float(z) for z in ckpt["depths_m"]) != tuple(DEPTHS):
        raise ValueError("Depths differ from pretrained model")
    for key, expected in (("wavelength_m", WAVELENGTH_M), ("pixel_pitch_m", PITCH_M)):
        if not np.isclose(float(ckpt[key]), expected, rtol=1e-8, atol=0):
            raise ValueError("Optical parameter mismatch: " + key)

    model = make_model("phase_cnn").cpu().eval()
    model.load_state_dict(ckpt["model"], strict=True)
    size = int(ckpt["size"])
    rows = []
    per_subject = []
    for idx, record in enumerate(selected, 1):
        volume, _ = load_volume(record["path"])
        planes, indices = build_target_planes(volume, axis=axis, size=size)
        target = target_scaled(planes)
        with torch.no_grad():
            initial_phase = model(torch.from_numpy(planes)[None])[0].detach()
        final_phase = refine(initial_phase, target, steps, lr)
        with torch.no_grad():
            cnn = propagate_planes(initial_phase).numpy()
            hybrid = propagate_planes(final_phase).numpy()
        desired = target.numpy()
        matrix, target_matrix, margins, correct, ambiguous = depth_metrics(hybrid, desired)
        subject_mse_cnn = float(np.mean((cnn - desired) ** 2))
        subject_mse_hybrid = float(np.mean((hybrid - desired) ** 2))
        per_subject.append({
            "subject_id": record["subject_id"],
            "cnn_mse": subject_mse_cnn,
            "hybrid_mse": subject_mse_hybrid,
            "matched_depth_wins": int(sum(x is True for x in correct)),
            "valid_depth_planes": int(sum(x is not None for x in correct)),
            "mean_diagonal_margin": float(np.nanmean(margins))
              if np.isfinite(margins).any() else None,
            "ambiguous_high_target_similarity_planes": int(sum(x is True for x in ambiguous)),
            "target_slice_indices": indices,
        })
        for i in range(3):
            for j in range(3):
                rows.append({
                    "subject_id": record["subject_id"],
                    "reconstruction_plane": i + 1,
                    "target_plane": j + 1,
                    "reconstruction_z_m": DEPTHS[i],
                    "target_z_m": DEPTHS[j],
                    "matched_depth": bool(i == j),
                    "hybrid_target_pearson": float(matrix[i,j]),
                    "native_target_pearson": float(target_matrix[i,j]),
                    "diagonal_minus_best_other": margins[i],
                    "best_match_is_intended": correct[i],
                    "high_target_similarity_flag": ambiguous[i],
                    "hybrid_plane_mse": float(np.mean((hybrid[i]-desired[i])**2)),
                    "cnn_plane_mse": float(np.mean((cnn[i]-desired[i])**2)),
                })
        print(f"{idx}/{len(selected)} {record['subject_id']} "
              f"CNN={subject_mse_cnn:.5f} hybrid={subject_mse_hybrid:.5f} "
              f"matched={sum(x is True for x in correct)}/3", flush=True)

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for filename, data, fields in (
        ("per_subject_depth.csv", per_subject,
         ("subject_id", "cnn_mse", "hybrid_mse", "matched_depth_wins",
          "valid_depth_planes", "mean_diagonal_margin",
          "ambiguous_high_target_similarity_planes", "target_slice_indices")),
        ("depth_matrix_long.csv", rows,
         ("subject_id", "reconstruction_plane", "target_plane",
          "reconstruction_z_m", "target_z_m", "matched_depth",
          "hybrid_target_pearson", "native_target_pearson",
          "diagonal_minus_best_other", "best_match_is_intended",
          "high_target_similarity_flag", "hybrid_plane_mse", "cnn_plane_mse")),
    ):
        with (out / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(data)

    diagonal = [r for r in rows if r["matched_depth"]]
    valid = [r for r in diagonal if r["best_match_is_intended"] is not None]
    marg = [r["diagonal_minus_best_other"] for r in valid]
    report = {
        "status": "real IXI MRI-derived intensity target, numerical holography simulation",
        "n_subjects": len(selected),
        "validation_total": len(validation),
        "pilot_subset": len(selected) != len(validation),
        "test_set_accessed": False,
        "manifest_sha256": sha256(manifest),
        "checkpoint_sha256": sha256(checkpoint),
        "optical_params": {
            "depths_m": list(DEPTHS),
            "wavelength_m": WAVELENGTH_M,
            "pitch_m": PITCH_M,
            "size": size
        },
        "refinement_steps": steps,
        "refinement_lr": lr,
        "phasecnn_cohort_mse": float(np.mean([r["cnn_mse"] for r in per_subject])),
        "hybrid_cohort_mse": float(np.mean([r["hybrid_mse"] for r in per_subject])),
        "depth_match_correct": int(sum(r["best_match_is_intended"] is True for r in valid)),
        "depth_match_evaluable": len(valid),
        "depth_match_fraction": float(np.mean(
            [r["best_match_is_intended"] for r in valid])) if valid else None,
        "median_matched_depth_margin": float(np.median(marg)) if marg else None,
        "high_target_similarity_plane_count": int(sum(
            r["high_target_similarity_flag"] is True for r in diagonal)),
        "limitations": [
            "Depth-match correlation compares output with MRI-derived target planes; it is not a physical axial resolution or optical cross-talk measurement.",
            "Different acquired MRI slices may be highly similar; inspect native target-to-target correlations.",
            "Validation-only exploratory diagnostic. No independent held-out performance claim.",
            "MRI intensity is a prescribed optical display target, not a measured biological optical field.",
            "Refinement is per-case target-aware numerical optimization, not instantaneous neural inference."
        ],
    }
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--axis", type=int, default=2)
    p.add_argument("--max-subjects", type=int)
    args = p.parse_args()
    run(args.manifest, args.checkpoint, args.out, args.steps, args.lr,
        args.axis, args.max_subjects)
