"""Deterministic multi-plane CGH smoke benchmark; NOT research validation.
Targets originate from phase-only masks propagated with the same simulated
optics used in training. No optical measurements or model-mismatch tests.
Run: python experiments/benchmark_synthetic.py --epochs 2
"""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from holography.models import PhaseCNN, PhysicsUnrolledCGH
from holography.torch_optics import propagate

DEPTHS = (0.001, 0.002, 0.003)
WAVELENGTH = 532e-9
PITCH = 3.74e-6

def targets_for_ids(ids, n, device):
    targets = []
    for identifier in ids:
        rng = np.random.default_rng(int(identifier))
        noise = torch.from_numpy(rng.standard_normal((1, 1, 8, 8)).astype('float32'))
        smooth = F.interpolate(noise, size=(n, n), mode='bicubic', align_corners=False)[0, 0].to(device)
        phase = torch.pi * torch.tanh(smooth)
        field = torch.polar(torch.ones_like(phase), phase).to(torch.complex64)
        targets.append(torch.stack([propagate(field, WAVELENGTH, PITCH, z).abs().square() for z in DEPTHS]))
    return torch.stack(targets)

def predicted_intensities(model, target):
    phase = model(target)
    field = torch.polar(torch.ones_like(phase), phase).to(torch.complex64)
    return torch.stack([propagate(field, WAVELENGTH, PITCH, z).abs().square() for z in DEPTHS], dim=1)

def measure(model, target):
    model.eval()
    # The unrolled solver computes an internal derivative: no torch.no_grad().
    with torch.enable_grad():
        return float((predicted_intensities(model, target) - target).square().mean().detach())

def train(model, target, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    history = []
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = (predicted_intensities(model, target) - target).square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite loss")
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    return history

def run(n=32, epochs=2, seed=20261030, device="cpu"):
    if n < 16 or epochs < 1:
        raise ValueError("Require size >=16 and epochs >=1")
    torch.manual_seed(seed)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    ids = {"train": [101, 102, 103, 104], "validation": [201, 202], "test": [301, 302]}
    train_target = targets_for_ids(ids["train"], n, device)
    validation_target = targets_for_ids(ids["validation"], n, device)
    test_target = targets_for_ids(ids["test"], n, device)
    outputs = {}
    for name, model in (("phase_cnn", PhaseCNN(3)),
                        ("physics_unrolled", PhysicsUnrolledCGH(DEPTHS, steps=2))):
        model = model.to(device)
        start = time.perf_counter()
        before = measure(model, validation_target)
        history = train(model, train_target, epochs)
        after = measure(model, validation_target)
        heldout = measure(model, test_target)
        outputs[name] = {
            "validation_mse_before": before, "validation_mse_after": after,
            "test_mse": heldout, "train_loss": history,
            "seconds_total": time.perf_counter() - start,
            "parameters": sum(p.numel() for p in model.parameters())
        }
    return {
        "status": "synthetic_simulator_smoke_test_only", "scientific_validation": False,
        "seed": seed, "size": n, "wavelength_m": WAVELENGTH,
        "pitch_m": PITCH, "depths_m": DEPTHS, "split_ids": ids,
        "split_seed_sha256": hashlib.sha256(json.dumps(ids, sort_keys=True).encode()).hexdigest(),
        "torch_version": torch.__version__, "python_version": platform.python_version(),
        "models": outputs
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("results/synthetic_smoke/benchmark.json"))
    args = parser.parse_args()
    result = run(args.size, args.epochs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["models"], indent=2))

if __name__ == "__main__":
    main()
