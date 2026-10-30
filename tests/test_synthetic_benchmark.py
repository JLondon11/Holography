"""Benchmark shape, split provenance and training smoke tests."""
from pathlib import Path
import sys
import pytest
torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from benchmark_synthetic import targets_for_ids, run

def test_seeded_targets_reproduce():
    a = targets_for_ids([101], 16, "cpu")
    b = targets_for_ids([101], 16, "cpu")
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert a.shape == (1, 3, 16, 16)
    assert torch.all(a >= 0)

def test_training_and_split_integrity():
    results = run(n=16, epochs=1)
    splits = results["split_ids"]
    assert len(set(splits["train"]) & set(splits["validation"])) == 0
    assert len(set(splits["train"]) & set(splits["test"])) == 0
    assert len(set(splits["validation"]) & set(splits["test"])) == 0
    assert set(results["models"]) == {"phase_cnn", "physics_unrolled"}
    for row in results["models"].values():
        assert row["test_mse"] >= 0
        assert len(row["train_loss"]) == 1
    assert results["scientific_validation"] is False
