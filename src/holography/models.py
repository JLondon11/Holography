"""Research model implementations; no pretrained weights or performance claims."""
import torch
from torch import nn
from .torch_optics import propagate

class PhaseCNN(nn.Module):
    """Compact phase prediction CNN baseline for multi-plane target intensity."""
    def __init__(self, planes=3, channels=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(planes, channels, 3, padding=1), nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.GELU(),
            nn.Conv2d(channels, 1, 1))
    def forward(self, target):
        if target.ndim != 4:
            raise ValueError("Expected [batch, depth, height, width]")
        return torch.pi * torch.tanh(self.net(target).squeeze(1))

class PhysicsUnrolledCGH(nn.Module):
    """Learnable initialization followed by differentiable phase-only descent.

    First-order unrolled projected-gradient steps use graph-based derivatives
    through optical propagation. A learned positive step per iteration and
    CNN initialization are optimized using outer reconstruction loss.
    Computationally expensive by design; not a validated novel contribution.
    """
    def __init__(self, depths_m, wavelength_m=532e-9, pitch_m=3.74e-6, steps=3):
        super().__init__()
        if not depths_m or steps < 1:
            raise ValueError("Depths and steps required")
        self.depths_m = tuple(float(z) for z in depths_m)
        self.wavelength_m, self.pitch_m = wavelength_m, pitch_m
        self.init = PhaseCNN(planes=len(depths_m))
        self.log_step = nn.Parameter(torch.full((steps,), -3.0))

    def simulate(self, phase):
        field = torch.polar(torch.ones_like(phase), phase).to(torch.complex64)
        images = [propagate(field, self.wavelength_m, self.pitch_m, z).abs().square()
                  for z in self.depths_m]
        return torch.stack(images, dim=1)

    def forward(self, target):
        phase = self.init(target)
        for log_step in self.log_step:
            with torch.enable_grad():
                if not phase.requires_grad:
                    phase = phase.detach().requires_grad_(True)
                loss = (self.simulate(phase) - target).square().mean()
                gradient = torch.autograd.grad(loss, phase, create_graph=self.training)[0]
            phase = phase - torch.nn.functional.softplus(log_step) * gradient
        return phase

def reconstruction_loss(model, target, phase):
    return (model.simulate(phase) - target).square().mean()
