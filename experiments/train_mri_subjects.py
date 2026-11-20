"""MRI-derived intensity CGH training, with leakage-safe subject partitions.

Uses real MRI files only when a verified real-data manifest is supplied.
Anatomical MRI intensities are target patterns, NOT tissue optical fields.
No inference of true refractive index, scattering or coherent MRI phase.
Run:
 python -m pip install -e '.[mri]'
 python experiments/train_mri_subjects.py --manifest data/private_manifest.json --out results/mri_subjects
"""
import argparse
import json
import os
import platform
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from holography.models import PhaseCNN, PhysicsUnrolledCGH
from subject_manifest import read_manifest
from mri_pipeline import load_volume, build_target_planes

DEPTHS = (0.004, 0.006, 0.008)
WAVELENGTH_M = 532e-9
PITCH_M = 3.74e-6

class MRIPlanes(Dataset):
    def __init__(self, records, size, axis):
        self.records = list(records)
        self.size = int(size)
        self.axis = int(axis)
    def __len__(self):
        return len(self.records)
    def __getitem__(self, index):
        row = self.records[index]
        volume, _ = load_volume(row["path"])
        planes, _ = build_target_planes(volume, axis=self.axis, size=self.size)
        return torch.from_numpy(planes), row["subject_id"]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def make_model(name):
    if name == "phase_cnn":
        return PhaseCNN(planes=len(DEPTHS))
    if name == "unrolled":
        return PhysicsUnrolledCGH(DEPTHS, WAVELENGTH_M, PITCH_M, steps=2)
    raise ValueError(f"Unknown model {name}")

def reconstructed_intensity(model, inputs):
    phase = model(inputs)
    if hasattr(model, "simulate"):
        return model.simulate(phase), phase
    from holography.torch_optics import propagate
    field = torch.polar(torch.ones_like(phase), phase).to(torch.complex64)
    return torch.stack([
        propagate(field, WAVELENGTH_M, PITCH_M, z).abs().square()
        for z in DEPTHS
    ], dim=1), phase

def objective(pred, target):
    # Match the available optical power within each target plane.
    # This defines a display-target optimization, not biological tissue optics.
    target = target / target.mean(dim=(-2,-1), keepdim=True).clamp_min(1e-5)
    return (pred - target).square().mean()

def epoch_pass(model, loader, optimizer, device):
    training = optimizer is not None
    model.train(training)
    sum_loss, seen = 0.0, 0
    for target, _ in loader:
        target = target.to(device)
        if training: optimizer.zero_grad(set_to_none=True)
        # Unrolled model uses autograd even during inference.
        with torch.enable_grad() if training or hasattr(model, "simulate") else torch.no_grad():
            estimate, _ = reconstructed_intensity(model, target)
            loss = objective(estimate, target)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Nonfinite CGH reconstruction loss")
            if training:
                loss.backward()
                optimizer.step()
        sum_loss += float(loss.detach()) * target.shape[0]
        seen += target.shape[0]
    if not seen: raise ValueError("Empty split")
    return sum_loss / seen

def train(manifest, output, epochs=5, size=64, axis=2, seed=20261030,
          model_name="phase_cnn", batch_size=1, device="cpu"):
    if epochs < 1 or size < 16 or batch_size < 1 or axis not in (0,1,2):
        raise ValueError("Invalid training arguments")
    rows = read_manifest(manifest)
    set_seed(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    batches = {}
    for split in ("train","validation","test"):
        group = [r for r in rows if r["split"] == split]
        if group:
            batches[split] = DataLoader(MRIPlanes(group, size, axis),
                batch_size=batch_size, shuffle=(split=="train"), num_workers=0)
    model = make_model(model_name).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    history = []
    best = float("inf")
    for epoch in range(epochs):
        train_mse = epoch_pass(model,batches["train"],optimizer,device)
        val_mse = epoch_pass(model,batches["validation"],None,device)
        history.append({"epoch":epoch+1, "train_mse":train_mse, "validation_mse":val_mse})
        if val_mse < best:
            best=val_mse
            torch.save({"model":model.state_dict(), "model_name":model_name, "epoch":epoch+1,
                        "seed":seed, "depths_m":DEPTHS,"pixel_pitch_m":PITCH_M,
                        "wavelength_m":WAVELENGTH_M, "size":size},output/"best_checkpoint.pt")
    # Assess test once after selecting checkpoint by validation score.
    checkpoint=torch.load(output/"best_checkpoint.pt",map_location=device,weights_only=True)
    model.load_state_dict(checkpoint["model"])
    test_mse=epoch_pass(model,batches["test"],None,device) if "test" in batches else None
    report={"status":"MRI-derived intensity target numerical simulation",
            "physical_hologram_measurement":False, "tissue_optical_model":False,
            "seed":seed,"epochs":epochs,"size":size,"axis":axis,
            "depths_m":DEPTHS,"wavelength_m":WAVELENGTH_M,"pixel_pitch_m":PITCH_M,
            "model":model_name,"subject_counts":{s:sum(r["split"]==s for r in rows)
                                                    for s in ("train","validation","test")},
            "manifest_subjects":[{"subject_id":r["subject_id"],"split":r["split"],
                                  "sha256":r["sha256"]} for r in rows],
            "best_validation_mse":best, "heldout_test_mse":test_mse,
            "history":history,"python":platform.python_version(),"torch":torch.__version__}
    (output/"results.json").write_text(json.dumps(report,indent=2))
    return report

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--manifest",required=True)
    p.add_argument("--out",default="results/mri_subjects")
    p.add_argument("--epochs",type=int,default=5)
    p.add_argument("--size",type=int,default=64)
    p.add_argument("--batch-size",type=int,default=1)
    p.add_argument("--axis",type=int,default=2)
    p.add_argument("--model",choices=("phase_cnn","unrolled"),default="phase_cnn")
    p.add_argument("--device",choices=("cpu","cuda","mps"),default="cpu")
    args=p.parse_args()
    result=train(args.manifest,args.out,args.epochs,args.size,args.axis,
                 model_name=args.model,batch_size=args.batch_size,device=args.device)
    print(json.dumps({k:result[k] for k in ("best_validation_mse","heldout_test_mse",
                      "subject_counts","status")},indent=2))
if __name__=="__main__":main()
