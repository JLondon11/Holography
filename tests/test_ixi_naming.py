"""IXI filename syntax tests only; no imaging data generated or used."""
import importlib.util
from pathlib import Path

module_path=Path(__file__).resolve().parents[1]/"data"/"build_ixi_manifest.py"
spec=importlib.util.spec_from_file_location("build_ixi_manifest",module_path)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def test_official_ixi_filename():
    for name in ("IXI002-Guys-0828-T1.nii.gz","IXI012-HH-1211-T1.nii.gz",
                 "IXI013-HH-1212-T1.nii"):
        assert module.PATTERN.fullmatch(name),name

def test_invalid_modalities_and_anatomical_substitutes_rejected():
    for name in ("IXI002-Guys-0828-T2.nii.gz","fake_brain.nii.gz",
                 "IXI002-Guys-0828-T1.npy"):
        assert module.PATTERN.fullmatch(name) is None

def test_stable_subject_level_splits():
    subject="IXI-002"
    assert module.split_for_subject(subject)==module.split_for_subject(subject)
    assert module.split_for_subject(subject) in {"train","validation","test"}
