"""Validate and analyze paired real-IXI MRI refinement CSV. No test access."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

def analyze(csv_file, summary_file, output):
    summary = json.loads(Path(summary_file).read_text())
    if summary.get("test_set_accessed") is not False or summary.get("status") != "real acquired IXI MRI validation targets, simulated propagation":
        raise ValueError("Expected verified real-IXI validation-only refinement results")
    budgets = sorted(set(int(v) for v in summary["budgets"]))
    measurements = {}
    with Path(csv_file).open(newline="") as f:
        reader = csv.DictReader(f)
        if not {"subject_id","method","steps","mse","seconds"} <= set(reader.fieldnames or []):
            raise ValueError("Incorrect CSV columns")
        for row in reader:
            key=(row["subject_id"],row["method"],int(row["steps"]))
            mse, secs = float(row["mse"]),float(row["seconds"])
            if (not key[0].startswith("IXI-") or key[1] not in
                    ("phase_cnn","cnn_refine_adam","random_adam") or
                    not np.isfinite(mse) or mse < 0 or not np.isfinite(secs) or secs < 0):
                raise ValueError(f"Invalid observation: {key}")
            if key in measurements:
                raise ValueError(f"Duplicate measurement: {key}")
            measurements[key]=(mse,secs)
    subjects=sorted({x[0] for x in measurements})
    if len(subjects) != summary["n_subjects"]:
        raise ValueError("Subject count disagrees with run summary")
    expected={(s,"phase_cnn",0) for s in subjects}
    expected |= {(s,m,b) for s in subjects for m in ("cnn_refine_adam","random_adam") for b in budgets}
    if expected != set(measurements):
        raise ValueError("Incomplete or unexpected subject/method/budget observations")
    comparisons=[]
    for b in budgets:
        warm=np.array([measurements[(s,"cnn_refine_adam",b)][0] for s in subjects])
        for other,other_b in (("random_adam",b),("phase_cnn",0)):
            reference=np.array([measurements[(s,other,other_b)][0] for s in subjects])
            difference=reference-warm
            comparisons.append({
                "steps":b,"comparison":f"cnn_refine_adam_vs_{other}",
                "cnn_refined_mean_mse":float(warm.mean()),
                "comparator_mean_mse":float(reference.mean()),
                "relative_mean_mse_reduction":float(difference.mean()/reference.mean()) if reference.mean()>0 else None,
                "mean_paired_improvement":float(difference.mean()),
                "median_paired_improvement":float(np.median(difference)),
                "paired_25th_percentile":float(np.percentile(difference,25)),
                "paired_75th_percentile":float(np.percentile(difference,75)),
                "wins":int(np.count_nonzero(difference>0)),
                "ties":int(np.count_nonzero(difference==0)),
                "losses":int(np.count_nonzero(difference<0)),
                "refined_mean_seconds":float(np.mean([measurements[(s,"cnn_refine_adam",b)][1] for s in subjects])),
                "comparator_mean_seconds":float(np.mean([measurements[(s,other,other_b)][1] for s in subjects]))
            })
    last=max(budgets)
    worst=sorted([{
        "subject_id":s,
        "phasecnn_mse":measurements[(s,"phase_cnn",0)][0],
        "refined_mse":measurements[(s,"cnn_refine_adam",last)][0],
        "random_mse":measurements[(s,"random_adam",last)][0]
        } for s in subjects], key=lambda r:r["refined_mse"],reverse=True)[:10]
    result={
        "status":"paired analysis of real-acquired MRI validation derived intensity reconstructions",
        "n_subjects":len(subjects),
        "full_validation_cohort":summary["full_validation_size"],
        "pilot_subset":summary["pilot_subset"],
        "test_set_accessed":False,
        "checkpoint_sha256":summary["checkpoint_sha256"],
        "manifest_sha256":summary["manifest_sha256"],
        "comparisons":comparisons,
        "highest_refined_mse_subjects":worst,
        "limitations":"Validation data were used to select the CNN checkpoint. Descriptive paired analyses only; no statistical significance, anatomical-fidelity or optical measurement claims."
    }
    out=Path(output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--csv",required=True)
    p.add_argument("--run-summary",required=True)
    p.add_argument("--out",required=True)
    a=p.parse_args()
    analyze(a.csv,a.run_summary,a.out)
