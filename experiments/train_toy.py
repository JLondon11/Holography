"""Smoke-training with explicitly synthetic target fields; NOT paper evidence.

Usage: pip install -e '.[deep,test]'
python experiments/train_toy.py --steps 5 --output results/toy_demo
"""
import argparse
import json
from pathlib import Path
import torch
from holography.models import PhysicsUnrolledCGH, reconstruction_loss

def make_toy_targets(batch, n, device):
    """Deterministic abstract depth targets solely for code-path verification."""
    coord = torch.linspace(-1, 1, n, device=device)
    y, x = torch.meshgrid(coord, coord, indexing="ij")
    layers = [torch.exp(-((x - d)**2 + (y + d)**2) / 0.12)
              for d in (-0.3, 0., 0.3)]
    return torch.stack(layers)[None].repeat(batch, 1, 1, 1)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--output", type=Path, default=Path("results/toy_demo"))
    a = p.parse_args()
    if a.steps < 1 or a.size < 16:
        p.error("steps>=1 and size>=16 required")
    torch.manual_seed(20261030)
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    model = PhysicsUnrolledCGH([.008, .012, .016]).to(device)
    target = make_toy_targets(1, a.size, device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = []
    for step in range(a.steps):
        opt.zero_grad(set_to_none=True)
        phase = model(target)
        loss = reconstruction_loss(model, target, phase)
        loss.backward()
        opt.step()
        history.append(float(loss.detach().cpu()))
        print(f"step={step+1} loss={history[-1]:.8g}")
    a.output.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), a.output / "toy_checkpoint.pt")
    (a.output / "manifest.json").write_text(json.dumps({
        "purpose": "synthetic smoke test only", "seed": 20261030,
        "model": "PhysicsUnrolledCGH", "size": a.size, "steps": a.steps,
        "depths_m": list(model.depths_m), "wavelength_m": model.wavelength_m,
        "pitch_m": model.pitch_m, "torch_version": torch.__version__,
        "device": str(device), "loss_history": history,
        "scientific_validation": False}, indent=2))
if __name__ == "__main__":
    main()
