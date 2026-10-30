import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from subject_manifest import read_manifest

def build_manifest(tmp_path, duplicate=False):
    rows=[]
    for i,split in enumerate(("train","validation","test")):
        filename=f"subject_{i}.npy"
        source=tmp_path/filename
        np.save(source,np.arange(16*16*16,dtype=np.float32).reshape(16,16,16)+i)
        digest=hashlib.sha256(source.read_bytes()).hexdigest()
        rows.append({"subject_id":f"s{i}", "path":filename, "split":split, "sha256":digest})
    if duplicate:
        rows[2]["subject_id"]="s0"
    path=tmp_path/"manifest.json"
    path.write_text(json.dumps({"subjects":rows}))
    return path

def test_manifest_subject_splits(tmp_path):
    records=read_manifest(build_manifest(tmp_path))
    assert len(records)==3
    assert {r["split"] for r in records}=={"train","validation","test"}

def test_duplicate_subject_rejected(tmp_path):
    with pytest.raises(ValueError,match="Duplicate"):
        read_manifest(build_manifest(tmp_path,True))

def test_digest_corruption_rejected(tmp_path):
    path=build_manifest(tmp_path)
    (tmp_path/"subject_0.npy").write_bytes(b"changed")
    with pytest.raises(ValueError,match="Checksum"):
        read_manifest(path)

def test_same_file_under_different_ids_rejected(tmp_path):
    path=build_manifest(tmp_path)
    payload=json.loads(path.read_text())
    payload["subjects"][2]["path"]=payload["subjects"][0]["path"]
    payload["subjects"][2]["sha256"]=payload["subjects"][0]["sha256"]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError,match="Duplicate input content"):
        read_manifest(path)

def test_test_split_required(tmp_path):
    path=build_manifest(tmp_path)
    payload=json.loads(path.read_text())
    payload["subjects"][2]["split"]="train"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError,match="held-out test"):
        read_manifest(path)
