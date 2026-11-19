"""Real MRI validation: export phase-only CGHs and depth-correlation diagnostics.

Uses the acquired IXI validation subjects and a frozen PhaseCNN checkpoint.
Numerically simulates optical propagation only; no fabricated MRI or optical
hardware measurements. Does not read held-out test subjects.

Example:
 python experiments/inspect_phase_and_depth.py \
 --manifest '/Volumes/BioMedAI Data Lake/IXI-T1/ixi_t1_manifest.json' \
 --checkpoint results/ixi_t1_phasecnn/best_checkpoint.pt \
 --out results/ixi_t1_phasecnn/phase_depth --subjects IXI-002 IXI-337 IXI-373
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from mri_pipeline import load_volume, build_target_planes
from subject_manifest import read_manifest
from train_mri_subjects import make_model, DEPTHS, WAVELENGTH_M, PITCH_M
from compare_mri_baselines import propagate_planes, target_scaled


def pearson(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.std() < 1e-10 or b.std() < 1e-10:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def refine(phase, target, steps, lr):
    variable = torch.nn.Parameter(phase.detach().clone())
    opt = torch.optim.Adam([variable], lr=lr)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = (propagate_planes(variable) - target).square().mean()
        loss.backward()
        opt.step()
    return variable.detach()


def run(manifest, checkpoint, out, subjects, steps=20, lr=0.05, axis=2):
    if not subjects or len(subjects) != len(set(subjects)):
        raise ValueError("Specify unique validation subject IDs")
    if steps < 1 or lr <= 0 or axis not in (0, 1, 2):
        raise ValueError("Invalid steps, learning rate, or MRI slice axis")
    records = {r["subject_id"]: r for r in read_manifest(manifest)
               if r["split"] == "validation"}
    if any(subject not in records for subject in subjects):
        raise ValueError("Every subject must belong to the validation split")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if ckpt["model_name"] != "phase_cnn":
        raise ValueError("Requires the frozen PhaseCNN checkpoint")
    if tuple(float(z) for z in ckpt["depths_m"]) != tuple(DEPTHS):
        raise ValueError("Depth mismatch")
    for key, value in (("wavelength_m", WAVELENGTH_M), ("pixel_pitch_m", PITCH_M)):
        if not np.isclose(float(ckpt[key]), value, rtol=1e-8, atol=0):
            raise ValueError("Optical sampling mismatch: " + key)
    model = make_model("phase_cnn").cpu().eval()
    model.load_state_dict(ckpt["model"], strict=True)
    size = int(ckpt["size"])
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for subject in subjects:
        volume, spacing = load_volume(records[subject]["path"])
        planes, indices = build_target_planes(volume, axis=axis, size=size)
        target = target_scaled(planes)
        with torch.no_grad():
            original_phase = model(torch.from_numpy(planes)[None])[0].detach()
        optimized_phase = refine(original_phase, target, steps, lr)
        with torch.no_grad():
            cnn_intensity = propagate_planes(original_phase).numpy()
            hybrid_intensity = propagate_planes(optimized_phase).numpy()
            wrapped_cnn = torch.angle(torch.polar(
                torch.ones_like(original_phase), original_phase)).numpy()
            wrapped_hybrid = torch.angle(torch.polar(
                torch.ones_like(optimized_phase), optimized_phase)).numpy()
        wanted = target.numpy()

        # Export both actual computed phase masks independently of image intensities.
        figure, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
        for axis_item, phase, title in zip(axes,
                (wrapped_cnn, wrapped_hybrid),
                ("PhaseCNN phase-only CGH", "Hybrid phase-only CGH")):
            heat = axis_item.imshow(phase, cmap="twilight", vmin=-np.pi, vmax=np.pi)
            axis_item.set_title(title)
            axis_item.set_xticks([])
            axis_item.set_yticks([])
        figure.colorbar(heat, ax=axes, shrink=0.78, label="Wrapped phase (radians)")
        figure.suptitle(subject + " — simulated phase holograms, not reconstructed anatomy")
        figure.savefig(out / (subject + "_phase_cgh.png"), dpi=200)
        plt.close(figure)
        np.save(out / (subject + "_phasecnn_phase_radians.npy"), wrapped_cnn)
        np.save(out / (subject + "_hybrid_phase_radians.npy"), wrapped_hybrid)

        # This matrix is a depth selectivity *proxy*. Adjacent MRI target slices
        # may themselves be similar, so off-diagonal correlations must be viewed
        # alongside the target-to-target similarity matrix.
        matrix = np.array([[pearson(hybrid_intensity[i], wanted[j])
                            for j in range(3)] for i in range(3)])
        target_matrix = np.array([[pearson(wanted[i], wanted[j])
                                   for j in range(3)] for i in range(3)])
        figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
        for axis_item, values, title in zip(axes, (matrix, target_matrix),
                ("Hybrid output vs target", "Target vs target")):
            heat = axis_item.imshow(values, vmin=-1, vmax=1, cmap="coolwarm")
            axis_item.set_title(title)
            axis_item.set_xticks(range(3), [f"{z*1000:g} mm" for z in DEPTHS])
            axis_item.set_yticks(range(3), [f"{z*1000:g} mm" for z in DEPTHS])
            axis_item.set_xlabel("Target plane")
            axis_item.set_ylabel("Reconstruction plane" if title.startswith("Hybrid")
                                 else "MRI target plane")
            for i in range(3):
                for j in range(3):
                    axis_item.text(j, i, f"{values[i,j]:.3f}",
                                   ha="center", va="center", fontsize=9)
        figure.colorbar(heat, ax=axes, shrink=0.78, label="Pearson r")
        figure.suptitle(subject + " — depth-correlation proxy (not optical crosstalk)")
        figure.savefig(out / (subject + "_depth_correlations.png"), dpi=200)
        plt.close(figure)

        for i in range(3):
            off = [matrix[i,j] for j in range(3) if i != j]
            margin = (float(matrix[i,i] - max(off))
                      if np.isfinite(matrix[i,i]) and all(np.isfinite(off))
                      else None)
            for j in range(3):
                rows.append({
                    "subject_id":subject,
                    "reconstruction_depth_m":DEPTHS[i],
                    "target_depth_m":DEPTHS[j],
                    "matched_depth":bool(i==j),
                    "hybrid_target_pearson":float(matrix[i,j]),
                    "target_target_pearson":float(target_matrix[i,j]),
                    "matched_vs_best_other_margin":margin,
                    "hybrid_plane_mse":float(np.mean((hybrid_intensity[i]-wanted[i])**2)),
                    "cnn_plane_mse":float(np.mean((cnn_intensity[i]-wanted[i])**2))
                })
        print(f"{subject}: hybrid MSE {float(np.mean((hybrid_intensity-wanted)**2)):.5f}; "
              f"phase and depth figures saved", flush=True)

    with (out / "depth_correlation_matrix.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "status": "real acquired IXI MRI validation intensity targets, numerically propagated CGH",
        "subjects": list(subjects),
        "depths_m": list(DEPTHS),
        "steps": steps,
        "learning_rate": lr,
        "test_set_accessed": False,
        "phase_masks_are_wrapped_radians": True,
        "caveats": [
            "An SLM phase pattern need not resemble brain anatomy; the reconstructed intensity should.",
            "These correlations are not a measured or calibrated axial cross-talk ratio.",
            "High target-to-target similarities limit depth-selectivity interpretation.",
            "No holographic optical hardware or physical tissue optical measurements."
        ]
    }
    (out / "run_info.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--subjects", nargs="+", default=["IXI-002","IXI-337","IXI-373"])
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--axis", type=int, default=2)
    args = p.parse_args()
    run(args.manifest, args.checkpoint, args.out, args.subjects, args.steps,
        args.lr, args.axis)
