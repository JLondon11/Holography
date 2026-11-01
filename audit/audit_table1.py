"""Audit transcribed Table I metrics without altering historical results.

Source: arXiv:2508.06703v1, page 14, Table I.
Run: python audit/audit_table1.py --output results/table1_metric_audit.csv
"""
import argparse
import csv
import math
from pathlib import Path

# Family, model, depth, unfiltered (MSE, RMSE, PSNR), filtered (MSE, RMSE, PSNR).
ROWS = [
    ("AP","SP-3D-AS-CH",22,.0052,.0724,35.86,.009,.0304,44.08),
    ("AP","SP-3D-AS-POH",22,.0053,.0725,22.79,.0009,.0304,30.35),
    ("AP","SEQ-3D-AS-CH",10,.3917,.6259,4.27,.150,.3873,8.43),
    ("AP","SEQ-3D-AS-POH",10,.2706,.5202,5.87,.1051,.3242,9.98),
    ("AP","G-3D-AS-CH",80,.0942,.3069,20.63,.2481,.4981,17.31),
    ("AP","G-3D-AS-POH",80,.1261,.3551,8.99,.1096,.3310,9.60),
    ("QN","G-3D-AS-CH",40,.1604,.4006,7.95,.1594,.3993,7.98),
    ("QN","G-3D-AS-POH",40,.2093,.4575,6.79,.0894,.299,10.49),
    ("QN","SP-3D-AS-CH",22,.0826,.2874,23.63,.0229,.1512,29.20),
    ("QN","SP-3D-AS-POH",22,.0740,.2721,24.29,.0227,.1508,16.43),
    ("SGD","G-3D-AS-CH",70,.2595,.5094,5.95,.0637,.2510,12.1),
    ("SGD","G-3D-AS-POH",70,.1261,.3551,8.99,.0829,.2879,10.82),
    ("SGD","SP-3D-AS-CH",22,.0057,.0757,35.4,.0231,.1520,29.43),
    ("SGD","SP-3D-AS-POH",22,.0072,.0847,21.44,.0012,.0347,29.20),
]

def audit(rmse_tolerance=0.001):
    """Compare RMSE against sqrt(MSE), with tolerance for printed rounding.

    An implied peak is diagnostic only; different masks or normalization might
    have been used in source calculations and must be verified from raw outputs.
    """
    records = []
    for family, model, depth, mse0, rmse0, psnr0, mse1, rmse1, psnr1 in ROWS:
        for filtered, mse, rmse, psnr in (
            (False, mse0, rmse0, psnr0), (True, mse1, rmse1, psnr1)
        ):
            expected = math.sqrt(mse)
            records.append({
                "family": family, "model": model, "depth_reported": depth,
                "filtered": filtered, "mse_reported": mse,
                "rmse_reported": rmse, "psnr_db_reported": psnr,
                "rmse_expected_from_mse": expected, "rmse_delta": rmse - expected,
                "rmse_consistent": abs(rmse-expected) <= rmse_tolerance,
                "psnr_db_if_normalized_peak_1": 10*math.log10(1/mse),
                "peak_implied_by_mse_psnr": math.sqrt(mse*10**(psnr/10))
            })
    return records

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=Path("results/table1_metric_audit.csv"))
    a = p.parse_args()
    rows = audit()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    inconsistent = [r for r in rows if not r["rmse_consistent"]]
    print(f"{len(rows)} comparisons; {len(inconsistent)} discrepancies with tolerance=0.001")
    for r in inconsistent:
        print(f"{r['family']}/{r['model']} filtered={r['filtered']}: "
              f"reported={r['rmse_reported']:.4f}, expected={r['rmse_expected_from_mse']:.4f}")
    print("Historical values remain unchanged. PSNR peak scaling is unverified.")

if __name__ == "__main__":
    main()
