"""Real-IXI-only, matched multi-plane hologram baseline comparison.

Compares frozen PhaseCNN, Gerchberg-Saxton-style simultaneous multi-plane
alternating projections (MP-GS), and direct phase-only Adam optimization.
NO model fitting on test subjects; ONLY IXI validation records.
Inputs must pass manifest checksum and provenance validation.

Run on Intel Mac (CPU):
 python experiments/compare_mri_baselines.py \
 --manifest '/Volumes/BioMedAI Data Lake/IXI-T1/ixi_t1_manifest.json' \
 --checkpoint results/ixi_t1_phasecnn/best_checkpoint.pt \
 --out results/ixi_t1_phasecnn/baselines --iterations 40

Full 88-subject run is default. --max-subjects is for explicit pilot runs
and is reported as a subset. No synthetic/anatomical proxy data allowed.
"""
import argparse
import csv
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from holography.torch_optics import propagate
from mri_pipeline import load_volume, build_target_planes
from subject_manifest import read_manifest
from train_mri_subjects import make_model, DEPTHS, WAVELENGTH_M, PITCH_M


def checksum(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def target_scaled(planes):
    t = torch.as_tensor(planes, dtype=torch.float32)
    if t.ndim != 3 or t.shape[0] != len(DEPTHS):
        raise ValueError("Expected [3,height,width] acquired MRI slice intensities")
    return t / t.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-5)


def propagate_planes(phase):
    """Simulated optical intensities from one phase-only mask."""
    field = torch.polar(torch.ones_like(phase), phase).to(torch.complex64)
    return torch.stack([propagate(field, WAVELENGTH_M, PITCH_M, z).abs().square()
                        for z in DEPTHS])


def mse_per_plane(predicted, target):
    return (predicted - target).square().mean(dim=(-2, -1)).detach().cpu().tolist()


def initialize_phase(height, width, seed):
    """Algorithm initialization only, not generated anatomical research data."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.rand((height, width), generator=generator) * (2 * torch.pi) - torch.pi)


def direct_adam(target, iterations, seed, lr):
    if iterations < 1 or lr <= 0:
        raise ValueError("iterations >=1 and lr >0 required")
    phase = torch.nn.Parameter(initialize_phase(target.shape[-2], target.shape[-1], seed))
    optimizer = torch.optim.Adam([phase], lr=lr)
    for _ in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        predicted = propagate_planes(phase)
        loss = (predicted - target).square().mean()
        loss.backward()
        optimizer.step()
    return phase.detach()


def multiplane_gs(target, iterations, seed):
    """Gerchberg-Saxton-style, simultaneous multi-plane amplitude projections.

    Backpropagates each constrained field to SLM; averages complex fields and
    projects onto unit-amplitude SLM. This is a named implementation variant,
    not a reproduction of a particular published paper or guaranteed optimum.
    """
    if iterations < 1:
        raise ValueError("iterations >=1 required")
    phase = initialize_phase(target.shape[-2], target.shape[-1], seed)
    desired_amplitudes = torch.sqrt(target.clamp_min(0))
    with torch.no_grad():
        for _ in range(iterations):
            source = torch.polar(torch.ones_like(phase), phase).to(torch.complex64)
            fields = []
            for i, z in enumerate(DEPTHS):
                plane = propagate(source, WAVELENGTH_M, PITCH_M, z)
                constrained = torch.polar(desired_amplitudes[i], torch.angle(plane))
                fields.append(propagate(constrained, WAVELENGTH_M, PITCH_M, -z))
            phase = torch.angle(torch.stack(fields).mean(dim=0))
    return phase


def run(manifest, checkpoint, out, iterations=40, lr=0.05, max_subjects=None,
        seed=20261119, axis=2):
    if iterations < 1 or lr <= 0 or axis not in (0, 1, 2):
        raise ValueError("Invalid iteration count, learning rate, or slice axis")
    if max_subjects is not None and max_subjects < 1:
        raise ValueError("--max-subjects must be >=1")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    records = [r for r in read_manifest(manifest) if r["split"] == "validation"]
    if not records:
        raise ValueError("No verified validation MRI volumes")
    selected = records[:max_subjects] if max_subjects is not None else records
    checkpoint = Path(checkpoint)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if ckpt["model_name"] != "phase_cnn":
        raise ValueError("This comparison requires the frozen PhaseCNN checkpoint")
    if tuple(float(z) for z in ckpt["depths_m"]) != tuple(DEPTHS):
        raise ValueError("Checkpoint depth geometry disagrees with benchmark")
    for key, expected in (("wavelength_m", WAVELENGTH_M), ("pixel_pitch_m", PITCH_M)):
        if not np.isclose(float(ckpt[key]), expected, rtol=1e-8, atol=0):
            raise ValueError(f"Checkpoint mismatch: {key}")
    size = int(ckpt["size"])
    model = make_model("phase_cnn").cpu().eval()
    model.load_state_dict(ckpt["model"], strict=True)

    rows = []
    for index, record in enumerate(selected):
        volume, _ = load_volume(record["path"])
        planes, _ = build_target_planes(volume, axis=axis, size=size)
        raw = torch.from_numpy(planes)
        scaled = target_scaled(planes)
        local_seed = seed + index
        estimates = {}
        timings = {}

        with torch.no_grad():
            start = time.perf_counter()
            cnn_phase = model(raw[None])[0]
            estimates["phase_cnn"] = propagate_planes(cnn_phase).detach()
            timings["phase_cnn"] = time.perf_counter() - start

        start = time.perf_counter()
        adam_phase = direct_adam(scaled, iterations, local_seed, lr)
        with torch.no_grad():
            estimates["direct_adam"] = propagate_planes(adam_phase)
        timings["direct_adam"] = time.perf_counter() - start

        start = time.perf_counter()
        gs_phase = multiplane_gs(scaled, iterations, local_seed)
        with torch.no_grad():
            estimates["multiplane_gs"] = propagate_planes(gs_phase)
        timings["multiplane_gs"] = time.perf_counter() - start

        for method, prediction in estimates.items():
            plane_mse = mse_per_plane(prediction, scaled)
            rows.append({"subject_id": record["subject_id"],
                         "scan_sha256":record["sha256"], "method":method,
                         "mse":float(np.mean(plane_mse)),
                         "plane1_mse":plane_mse[0], "plane2_mse":plane_mse[1],
                         "plane3_mse":plane_mse[2],
                         "runtime_seconds":timings[method]})
        print(f"{index+1}/{len(selected)} {record['subject_id']} "
              + " ".join(f"{method}:{np.mean(mse_per_plane(pred,scaled)):.5f}"
                         for method,pred in estimates.items()),flush=True)

    directory = Path(out)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/"per_subject_methods.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    methods = ("phase_cnn","direct_adam","multiplane_gs")
    metrics = {}
    for method in methods:
        subset = [r for r in rows if r["method"] == method]
        metrics[method] = {
            "mean_mse": float(np.mean([r["mse"] for r in subset])),
            "median_mse": float(np.median([r["mse"] for r in subset])),
            "mean_runtime_seconds": float(np.mean([r["runtime_seconds"] for r in subset])),
            "median_runtime_seconds": float(np.median([r["runtime_seconds"] for r in subset]))
        }
    result = {
        "status": "real IXI validation anatomical intensity targets, simulated optics",
        "n_validation_subjects":len(selected),
        "full_validation_cohort":len(records),
        "pilot_subset":len(selected)!=len(records),
        "test_set_accessed":False,
        "model_checkpoint_sha256":checksum(checkpoint),
        "manifest_sha256":checksum(manifest),
        "physics":{"wavelength_m":WAVELENGTH_M,"pixel_pitch_m":PITCH_M,
                   "depths_m":DEPTHS,"resolution":size},
        "iterations":iterations,"adam_learning_rate":lr,"seed":seed,
        "device":"cpu","torch_version":torch.__version__,
        "python_version":platform.python_version(),
        "methods":metrics,
        "warnings":[
            "Per-subject direct optimization receives that subject's target intensity: it is not amortized inference.",
            "CNN timing excludes training; all timings include each method's reconstruction.",
            "MP-GS is an explicitly implemented variant, not a faithful external-paper reproduction.",
            "All target intensity planes derive from real acquired MRI; propagation is simulated, not optically measured.",
            "MSE uses mean-normalized intensity; do not convert to PSNR without a consistent defined peak.",
            "No axial crosstalk or optical calibration has yet been assessed."
        ]
    }
    (directory/"summary.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--manifest",required=True)
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--iterations",type=int,default=40)
    p.add_argument("--lr",type=float,default=0.05)
    p.add_argument("--seed",type=int,default=20261119)
    p.add_argument("--axis",type=int,default=2)
    p.add_argument("--max-subjects",type=int)
    args=p.parse_args()
    run(args.manifest,args.checkpoint,args.out,args.iterations,args.lr,args.max_subjects,args.seed,args.axis)

if __name__=="__main__":
    main()
