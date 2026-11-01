"""Optional CPU-only deep-learning smoke tests."""
import pytest
torch = pytest.importorskip("torch")
from holography.torch_optics import propagate
from holography.models import PhaseCNN, PhysicsUnrolledCGH, reconstruction_loss

def test_zero_distance_complex_identity():
    field = torch.complex(torch.randn(2, 32, 32), torch.randn(2, 32, 32))
    out = propagate(field, 532e-9, 3.74e-6, 0.0)
    torch.testing.assert_close(out, field, atol=1e-5, rtol=1e-5)

def test_network_shapes_and_gradient():
    x = torch.rand(1, 3, 16, 16)
    cnn = PhaseCNN()
    phase = cnn(x)
    assert phase.shape == (1, 16, 16)
    assert bool(torch.isfinite(phase).all())
    phase.square().mean().backward()
    assert any(p.grad is not None for p in cnn.parameters())

def test_unrolled_optics_backward():
    model = PhysicsUnrolledCGH([0.001, 0.002, 0.003], steps=2)
    x = torch.rand(1, 3, 16, 16)
    phase = model(x)
    loss = reconstruction_loss(model, x, phase)
    loss.backward()
    assert bool(torch.isfinite(loss))
    assert model.log_step.grad is not None
    assert all(p.grad is not None for p in model.init.parameters())
