"""Create a strict subject-level manifest from *downloaded real* IXI T1 NIfTI files.

No phantom generation, no test data creation, no silent dataset download.
Usage:
  python data/build_ixi_manifest.py --input-dir /path/to/IXI-T1 \
    --output /private/ixi_t1_manifest.json

The output contains scan paths and hashes; keep it out of public Git.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

SOURCE = "https://brain-development.org/ixi-dataset/"
PATTERN = re.compile(r"^IXI(?P<id>[0-9]{3})-(?P<site>Guys|HH|IOP)-(?P<sequence>[^/]+)-T1\.nii(\.gz)?$", re.IGNORECASE)

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def split_for_subject(subject_id):
    """Stable subject-level split; never divide slices from a subject."""
    value = int.from_bytes(hashlib.sha256(("IXI-T1:" + subject_id).encode()).digest()[:8], "big") % 100
    return "train" if value < 70 else ("validation" if value < 85 else "test")

def create_manifest(source_dir):
    source_dir = Path(source_dir).resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(source_dir)
    scans = sorted(p for p in source_dir.rglob("*") if p.is_file()
                   and (p.name.endswith(".nii") or p.name.endswith(".nii.gz")))
    if not scans:
        raise ValueError("No downloaded NIfTI scans found; no training can proceed")
    rows = []
    seen = set()
    for path in scans:
        match = PATTERN.fullmatch(path.name)
        if match is None:
            raise ValueError(f"Unrecognized IXI-T1 filename: {path.name}")
        subject_id = "IXI-" + match.group("id")
        if subject_id in seen:
            raise ValueError(f"Duplicate subject across scans: {subject_id}")
        seen.add(subject_id)
        # Validate that the bytes really parse as a finite NIfTI volume.
        import numpy as np
        import nibabel as nib
        image = nib.load(str(path))
        if len(image.shape) != 3 or min(image.shape) < 4:
            raise ValueError(f"Not a 3D MRI: {path.name}")
        zooms = image.header.get_zooms()[:3]
        if len(zooms) != 3 or not all(np.isfinite(zooms)) or not all(z > 0 for z in zooms):
            raise ValueError(f"Invalid MRI voxel spacing: {path.name}")
        if not np.isfinite(image.affine).all() or np.linalg.det(image.affine[:3,:3]) == 0:
            raise ValueError(f"Invalid MRI affine: {path.name}")
        # Data are read once for finite validation; expensive by design.
        if not np.all(np.isfinite(np.asarray(image.dataobj))):
            raise ValueError(f"Nonfinite voxel intensities: {path.name}")
        rows.append({
            "subject_id":subject_id,"path":str(path),
            "split":split_for_subject(subject_id),"sha256":sha256_file(path),
            "acquisition_site":match.group("site").upper(),
            "volume_shape":list(image.shape),
            "voxel_spacing_mm":[float(x) for x in zooms]
        })
    splits = {r["split"] for r in rows}
    if splits != {"train","validation","test"}:
        raise ValueError(f"Insufficient subjects for all three splits: {sorted(splits)}")
    return {
        "dataset":{"origin":"real_acquisition","name":"IXI T1 MRI","source_url":SOURCE,
            "license":"CC BY-SA 3.0","citation":"IXI Dataset, Biomedical Image Analysis Group, Imperial College London, https://brain-development.org/ixi-dataset/"},
        "split_method":"SHA-256(IXI-T1:subject_id) modulo 100; train 0-69, validation 70-84, test 85-99",
        "subjects":rows
    }

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args=parser.parse_args()
    report=create_manifest(args.input_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+"
")
    print(json.dumps({"verified_real_mri":len(report["subjects"]),
                      "split_counts":{s:sum(r["split"]==s for r in report["subjects"])
                                      for s in ("train","validation","test")},
                      "manifest":str(args.output)},indent=2))
if __name__=="__main__":
    main()
