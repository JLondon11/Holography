"""Subject-level partition and checksum validation for MRI studies."""
import hashlib
import json
from pathlib import Path

REQUIRED = ("subject_id", "path", "split", "sha256")
SPLITS = {"train", "validation", "test"}

def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def read_manifest(path, verify_files=True):
    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text())
    if not isinstance(payload, dict) or not isinstance(payload.get("subjects"), list):
        raise ValueError("JSON must have a subjects array")
    provenance = payload.get("dataset")
    if not isinstance(provenance, dict) or provenance.get("origin") != "real_acquisition":
        raise ValueError("Real acquired dataset provenance declaration required")
    if not all(isinstance(provenance.get(k),str) and provenance[k].strip() for k in ("name","source_url","license","citation")):
        raise ValueError("Dataset name, source URL, license and citation are required")
    subjects = payload["subjects"]
    if not subjects:
        raise ValueError("Manifest is empty")
    by_id = {}
    by_digest = {}
    for row in subjects:
        if any(k not in row for k in REQUIRED):
            raise ValueError("Each row requires subject_id, path, split, sha256")
        key, split = str(row["subject_id"]), row["split"]
        if not key.strip() or split not in SPLITS:
            raise ValueError("Invalid subject ID or split")
        if key in by_id:
            raise ValueError(f"Duplicate subject ID, possible leakage: {key}")
        source = Path(row["path"])
        if not source.is_absolute():
            source = manifest_path.parent / source
        source = source.resolve()
        if verify_files:
            if not source.is_file():
                raise FileNotFoundError(source)
            if sha256_file(source).lower() != str(row["sha256"]).lower():
                raise ValueError(f"Checksum mismatch for {key}")
        digest = str(row["sha256"]).lower()
        if digest in by_digest:
            raise ValueError(f"Duplicate input content between {by_digest[digest]} and {key}")
        by_digest[digest] = key
        by_id[key] = {"subject_id":key, "split":split, "path":str(source),
                      "sha256":str(row["sha256"]).lower()}
    if {r["split"] for r in by_id.values()} != SPLITS:
        raise ValueError("Train, validation, and held-out test subjects are all required")
    return [by_id[k] for k in sorted(by_id)]
