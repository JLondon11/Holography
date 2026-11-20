import numpy as np
import pytest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from mri_pipeline import load_volume,build_target_planes

def test_npy_volume_roundtrip(tmp_path):
    volume=np.arange(20*24*30,dtype=np.float32).reshape(20,24,30)
    path=tmp_path/'volume.npy';np.save(path,volume)
    loaded,spacing=load_volume(path)
    np.testing.assert_array_equal(loaded,volume)
    assert spacing is None
    target,indices=build_target_planes(loaded,axis=2,size=32)
    assert target.shape==(3,32,32)
    assert indices==[10,14,19]
    assert np.min(target)>=0 and np.max(target)<=1

def test_missing_contrast_rejected():
    with pytest.raises(ValueError):
        build_target_planes(np.zeros((12,12,12)),size=32)

def test_invalid_shape_rejected(tmp_path):
    path=tmp_path/'bad.npy';np.save(path,np.zeros((10,10)))
    with pytest.raises(ValueError):
        load_volume(path)
