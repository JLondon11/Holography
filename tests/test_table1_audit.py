"""Original Table I checks; distinct from physical/optical validation."""
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "audit"))
from audit_table1 import ROWS, audit

def test_transcribed_table_size():
    assert len(ROWS) == 14
    assert len(audit()) == 28

def test_major_filtered_ap_discrepancy():
    x = next(r for r in audit()
             if r["family"] == "AP" and r["model"] == "SP-3D-AS-CH" and r["filtered"])
    assert not x["rmse_consistent"]
    assert math.isclose(x["rmse_expected_from_mse"], math.sqrt(0.009))

def test_consistent_example():
    x = next(r for r in audit()
             if r["family"] == "AP" and r["model"] == "SEQ-3D-AS-CH" and not r["filtered"])
    assert x["rmse_consistent"]
