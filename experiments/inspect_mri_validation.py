"""Inspect a trained model on one real MRI validation subject.

Produces numerical diagnostics and a figure, not optical-bench measurements.
No synthetic inputs and no use of held-out test subjects.
Usage:
 python experiments/inspect_mri_validation.py \
   --manifest '/Volumes/BioMedAI Data Lake/IXI-T1/ixi_t1_manifest.json' \
   --checkpoint results/ixi_t1_phasecnn/best_checkpoint.pt \
   --out results/ixi_t1_phasecnn/validation_diagnostic
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mri_pipeline import build_target_planes, load_volume
from subject_manifest import read_manifest
from train_mri_subjects import make_model, reconstructed_intensity


def validate_subject(manifest, checkpoint, out, subject_id=None, axis=2):
    records = [r for r in read_manifest(manifest) if r["split"] == "validation"]
    if not records:
        raise ValueError("No validation subjects found")
    by_id = {r["subject_id"]: r for r in records}
    if subject_id is not None and subject_id not in by_id:
        raise ValueError("Subject must belong to the validation split")
    row = by_id[subject_id] if subject_id else records[0]

    # The checkpoint contains only tensors and primitive numeric metadata.
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model_name = ckpt["model_name"]
    if model_name not in ("phase_cnn", "unrolled"):
        raise ValueError(f"Unsupported checkpoint model: {model_name}")
    size = int(ckpt["size"])
    if tuple(float(x) for x in ckpt["depths_m"]) != (.004, .006, .008):
        raise ValueError("Depth settings do not match the current reconstruction implementation")
    model = make_model(model_name)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    volume, spacing = load_volume(row["path"])
    original, indices = build_target_planes(volume, axis=axis, size=size)
    input_tensor = torch.from_numpy(original)[None]

    # For consistency with the training objective, target normalization is
    # performed *after* generating the phase estimate from raw input slices.
    with torch.enable_grad():
        predicted, phase = reconstructed_intensity(model, input_tensor)
    actual = predicted[0].detach().cpu().numpy()
    phase = phase[0].detach().cpu().numpy()
    target = original / np.maximum(original.mean(axis=(-2, -1), keepdims=True), 1e-5)
    # A fixed unit-intensity plane wave at the detector is a simple reference.
    # It is NOT a matched optimization baseline and is NOT GS or HoloNet.
    flat = np.ones_like(target)
    model_mse = float(np.mean((actual - target) ** 2))
    flat_mse = float(np.mean((flat - target) ** 2))
    per_plane = [float(np.mean((actual[i] - target[i]) ** 2)) for i in range(target.shape[0])]
    output = Path(out)
    output.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(4, 3, figsize=(11, 11), layout="constrained")
    for k in range(3):
        vmax = float(max(np.percentile(target[k], 99), np.percentile(actual[k], 99), 1e-5))
        ax[0, k].imshow(original[k], cmap="gray", vmin=0, vmax=1)
        ax[0, k].set_title(f"Acquired MRI slice index {indices[k]}")
        ax[1, k].imshow(target[k], cmap="gray", vmin=0, vmax=vmax)
        ax[1, k].set_title(f"Optical intensity target plane {k + 1}")
        ax[2, k].imshow(actual[k], cmap="gray", vmin=0, vmax=vmax)
        ax[2, k].set_title(f"Simulated reconstruction plane {k + 1}")
        ax[3, k].imshow(np.abs(actual[k] - target[k]), cmap="magma")
        ax[3, k].set_title(f"Absolute intensity error: MSE {per_plane[k]:.4f}")
    for axis_item in ax.flat:
        axis_item.set_xticks([])
        axis_item.set_yticks([])
    fig.suptitle("REAL IXI MRI input — simulated CGH reconstruction (not optical measurement)")
    fig.savefig(output / "validation_depth_planes.png", dpi=220)
    fig.savefig(output / "validation_depth_planes.pdf")
    plt.close(fig)

    plt.figure(figsize=(5, 4))
    plt.imshow(phase, cmap="twilight", vmin=-np.pi, vmax=np.pi)
    plt.colorbar(label="SLM phase (rad)")
    plt.title("Predicted phase-only hologram")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output / "predicted_slm_phase.png", dpi=220)
    plt.close()

    report = {
        "status": "real MRI input, simulated optics, validation subject only",
        "subject_id": row["subject_id"],
        "scan_sha256": row["sha256"],
        "model": model_name,
        "checkpoint": str(checkpoint),
        "selected_slice_indices": indices,
        "voxel_spacing_mm": spacing,
        "intensity_target_mean_normalized": True,
        "model_validation_subject_mse": model_mse,
        "flat_unit_intensity_reference_mse": flat_mse,
        "per_plane_mse": per_plane,
        "physical_measurement": False,
        "note": "Single-subject diagnostic only. Baseline is trivial, not a published algorithm."
    }
    (output / "diagnostic.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--subject-id", default=None)
    parser.add_argument("--axis", type=int, default=2)
    args = parser.parse_args()
    validate_subject(args.manifest, args.checkpoint, args.out,
                     subject_id=args.subject_id, axis=args.axis)


if __name__ == "__main__":
    main()
