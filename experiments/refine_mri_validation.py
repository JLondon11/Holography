"""Compare frozen PhaseCNN with physics-only phase refinement on REAL IXI validation MRIs.

No learned weights updated. No synthetic inputs. No test subjects.
CNN-initialized Adam and random-initialized Adam get exactly the same
gradient-step budgets. All phases use the same simulated ASM.

Usage:
 python experiments/refine_mri_validation.py \
 --manifest '/Volumes/BioMedAI Data Lake/IXI-T1/ixi_t1_manifest.json' \
 --checkpoint results/ixi_t1_phasecnn/best_checkpoint.pt \
 --out results/ixi_t1_phasecnn/refinement_pilot --max-subjects 3
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
from compare_mri_baselines import propagate_planes, target_scaled, initialize_phase


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def optimize_checkpoints(initial_phase, target, steps, learning_rate):
    """Record MSE/time at each budget without reinitializing between budgets."""
    if not steps or any(n < 1 for n in steps):
        raise ValueError("positive refinement step budgets required")
    phase = torch.nn.Parameter(initial_phase.detach().clone().cpu())
    opt = torch.optim.Adam([phase], lr=learning_rate)
    results = {}
    start = time.perf_counter()
    max_steps = max(steps)
    for i in range(1, max_steps + 1):
        opt.zero_grad(set_to_none=True)
        predicted = propagate_planes(phase)
        loss = (predicted - target).square().mean()
        loss.backward()
        opt.step()
        if i in steps:
            with torch.no_grad():
                estimate = propagate_planes(phase)
                per_plane = (estimate - target).square().mean(dim=(-2, -1)).cpu().tolist()
            results[i] = {
                "mse": float(np.mean(per_plane)),
                "plane_mse": [float(x) for x in per_plane],
                "seconds": time.perf_counter() - start,
            }
    return results


def run(manifest, checkpoint, out, steps=(5, 10, 20), lr=0.05,
        seed=20261119, axis=2, max_subjects=None):
    if axis not in (0, 1, 2) or lr <= 0:
        raise ValueError("Invalid axis or learning rate")
    if max_subjects is not None and max_subjects < 1:
        raise ValueError("max-subjects must be positive")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    records = [r for r in read_manifest(manifest) if r["split"] == "validation"]
    if not records:
        raise ValueError("No acquired MRI validation data available")
    chosen = records[:max_subjects] if max_subjects is not None else records
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved["model_name"] != "phase_cnn":
        raise ValueError("Requires a frozen PhaseCNN checkpoint")
    if tuple(float(z) for z in saved["depths_m"]) != tuple(DEPTHS):
        raise ValueError("Depth mismatch")
    for name, val in (("wavelength_m", WAVELENGTH_M), ("pixel_pitch_m", PITCH_M)):
        if not np.isclose(float(saved[name]), val, rtol=1e-8, atol=0):
            raise ValueError(f"Optical parameter mismatch: {name}")
    model = make_model("phase_cnn").cpu().eval()
    model.load_state_dict(saved["model"], strict=True)
    n = int(saved["size"])
    rows = []
    for index, record in enumerate(chosen):
        volume, _ = load_volume(record["path"])
        planes, _ = build_target_planes(volume, axis=axis, size=n)
        raw = torch.from_numpy(planes)
        target = target_scaled(planes)
        with torch.no_grad():
            t = time.perf_counter()
            cnn_phase = model(raw[None])[0].detach()
            cnn_pred = propagate_planes(cnn_phase)
            cnn_seconds = time.perf_counter() - t
            cnn_mse = float((cnn_pred - target).square().mean().item())
        rows.append({"subject_id":record["subject_id"], "method":"phase_cnn",
                     "steps":0, "mse":cnn_mse, "seconds":cnn_seconds})
        random_phase = initialize_phase(n, n, seed + index)
        for method, initial in (("cnn_refine_adam",cnn_phase),
                                ("random_adam",random_phase)):
            measures = optimize_checkpoints(initial,target,steps,lr)
            for budget, measurement in measures.items():
                rows.append({"subject_id":record["subject_id"],
                             "method":method,"steps":budget,
                             "mse":measurement["mse"],
                             "seconds":measurement["seconds"] +
                                       (cnn_seconds if method=="cnn_refine_adam" else 0)})
        print(f"{index+1}/{len(chosen)} {record['subject_id']} "
              f"CNN={cnn_mse:.5f} "
              f"warm{max(steps)}={measures[max(steps)]['mse']:.5f}",flush=True)

    out = Path(out)
    out.mkdir(parents=True,exist_ok=True)
    with (out/"per_subject_refinement.csv").open("w",newline="") as file:
        writer=csv.DictWriter(file, fieldnames=("subject_id","method","steps","mse","seconds"))
        writer.writeheader()
        writer.writerows(rows)
    groups = sorted(set((r["method"],r["steps"]) for r in rows))
    aggregates = []
    for method, steps_used in groups:
        matches = [r for r in rows if r["method"]==method and r["steps"]==steps_used]
        aggregates.append({"method":method,"steps":steps_used,
                           "mean_mse":float(np.mean([r["mse"] for r in matches])),
                           "mean_cpu_seconds":float(np.mean([r["seconds"] for r in matches])),
                           "median_mse":float(np.median([r["mse"] for r in matches]))})
    report={"status":"real acquired IXI MRI validation targets, simulated propagation",
            "n_subjects":len(chosen),"full_validation_size":len(records),
            "pilot_subset":len(chosen)!=len(records),"test_set_accessed":False,
            "checkpoint_sha256":sha256(checkpoint),"manifest_sha256":sha256(manifest),
            "optical_wavelength_m":WAVELENGTH_M,"pixel_pitch_m":PITCH_M,
            "depths_m":list(DEPTHS),"size":n,"seed":seed,
            "learning_rate":lr,"budgets":list(steps),
            "method_results":aggregates,
            "limitations":"This is validation-only physics optimization, not extra neural training. CPU timing includes CNN inference for its warm-start optimizer. No actual optical bench measurement. Improvement alone does not establish scientific novelty."}
    (out/"summary.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--manifest",required=True)
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--steps",type=int,nargs="+",default=[5,10,20])
    p.add_argument("--lr",type=float,default=0.05)
    p.add_argument("--seed",type=int,default=20261119)
    p.add_argument("--axis",type=int,default=2)
    p.add_argument("--max-subjects",type=int)
    a=p.parse_args()
    run(a.manifest,a.checkpoint,a.out,tuple(sorted(set(a.steps))),a.lr,
        a.seed,a.axis,a.max_subjects)
