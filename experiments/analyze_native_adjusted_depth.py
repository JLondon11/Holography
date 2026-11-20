"""Quantify native-slice-adjusted selectivity from acquired IXI validation CSV.

Reads the already-computed depth_matrix_long.csv; no new MRI or held-out
test access, no newly generated images. Both metrics are diagnostic proxies:
raw margin = r(output_i,target_i) - max_j!=i r(output_i,target_j)
native contrast = 1 - max_j!=i r(target_i,target_j)
contrast-normalized margin = raw margin / native contrast.
The normalized ratio is NOT a probability, physical crosstalk, or depth resolution.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def analyze(input_path, output_dir, min_native_contrast=0.05):
    if not 0 < min_native_contrast < 1:
        raise ValueError("min_native_contrast must be strictly between 0 and 1")
    raw = list(csv.DictReader(Path(input_path).open(newline="")))
    by_subject = {}
    for row in raw:
        sid = row["subject_id"]
        i, j = int(row["reconstruction_plane"]), int(row["target_plane"])
        if i not in (1, 2, 3) or j not in (1, 2, 3):
            raise ValueError("Expected 3x3 plane pairs")
        key = (i, j)
        by_subject.setdefault(sid, {})
        if key in by_subject[sid]:
            raise ValueError("Duplicate plane pair: " + sid)
        by_subject[sid][key] = (
            float(row["hybrid_target_pearson"]),
            float(row["native_target_pearson"]),
        )
    if not by_subject:
        raise ValueError("Empty depth correlation input")
    details = []
    for sid, pairs in sorted(by_subject.items()):
        if len(pairs) != 9:
            raise ValueError("Incomplete matrix for " + sid)
        for i in (1, 2, 3):
            diag = pairs[(i, i)][0]
            competitors = [(j, pairs[(i, j)][0]) for j in (1, 2, 3) if j != i]
            best_j, best_corr = max(competitors, key=lambda pair: pair[1])
            native_best = max(pairs[(i, j)][1] for j in (1, 2, 3) if j != i)
            native_contrast = 1.0 - native_best
            margin = diag - best_corr
            eligible = native_contrast >= min_native_contrast
            if not np.isfinite([diag, best_corr, native_best, margin]).all():
                raise ValueError("Nonfinite correlation: " + sid)
            details.append({
                "subject_id": sid,
                "plane": i,
                "best_competing_target_plane": best_j,
                "matched_correlation": diag,
                "best_competing_correlation": best_corr,
                "raw_margin": margin,
                "native_target_max_offdiagonal": native_best,
                "native_target_contrast": native_contrast,
                "native_adjusted_margin": margin / native_contrast if eligible else "",
                "native_adjusted_evaluable": eligible,
            })
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "per_plane_native_adjusted.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(details[0]))
        writer.writeheader()
        writer.writerows(details)
    groups = {}
    for i in (1, 2, 3):
        subset = [d for d in details if d["plane"] == i]
        usable = [d for d in subset if d["native_adjusted_evaluable"]]
        groups[str(i)] = {
            "n": len(subset),
            "eligible_n": len(usable),
            "raw_margin_mean": float(np.mean([d["raw_margin"] for d in subset])),
            "raw_margin_median": float(np.median([d["raw_margin"] for d in subset])),
            "native_target_contrast_median": float(np.median(
                [d["native_target_contrast"] for d in subset])),
            "native_adjusted_margin_median": float(np.median(
                [d["native_adjusted_margin"] for d in usable])) if usable else None,
            "native_adjusted_margin_mean": float(np.mean(
                [d["native_adjusted_margin"] for d in usable])) if usable else None,
            "correct_matches": int(sum(d["raw_margin"] > 0 for d in subset)),
        }
    summary = {
        "status": "IXI validation existing correlation matrix postprocessing",
        "subjects": len(by_subject),
        "planes": len(details),
        "minimum_native_target_contrast": min_native_contrast,
        "depth_groups": groups,
        "limitations": [
            "This normalized margin divides two correlation contrasts and is an exploratory similarity-adjusted index.",
            "It is not calibrated optical crosstalk, physical depth resolution, an independent target, or an estimate of clinical fidelity.",
            "Near-identical target slices yield unstable ratios; these are excluded by the documented contrast threshold.",
            "No patient-level clinical claims or optical bench measurements are supported.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-native-contrast", type=float, default=0.05)
    args = parser.parse_args()
    analyze(args.csv, args.out, args.min_native_contrast)
