"""MRI volume -> target intensity planes (NOT a tissue optical phase model)."""
from pathlib import Path
import numpy as np
from scipy.ndimage import zoom

def load_volume(path):
    path = Path(path)
    suffixes = ''.join(path.suffixes).lower()
    if suffixes.endswith(('.nii', '.nii.gz')):
        try:
            import nibabel as nib
        except ImportError as ex:
            raise ImportError('For NIfTI install nibabel: pip install nibabel') from ex
        image = nib.load(str(path))
        volume = np.asarray(image.get_fdata(dtype=np.float32))
        spacing_mm = tuple(float(x) for x in image.header.get_zooms()[:3])
    elif suffixes.endswith('.npy'):
        volume = np.load(path, allow_pickle=False)
        spacing_mm = None
    elif suffixes.endswith('.npz'):
        with np.load(path, allow_pickle=False) as archive:
            if 'volume' not in archive.files:
                raise ValueError('NPZ must contain array named volume')
            volume = archive['volume']
            spacing_mm = tuple(float(x) for x in archive['spacing_mm']) if 'spacing_mm' in archive.files else None
    else:
        raise ValueError('Use .nii, .nii.gz, .npy, or .npz')
    if volume.ndim != 3 or min(volume.shape) < 4 or not np.isfinite(volume).all():
        raise ValueError('Expected finite 3-D anatomical volume, at least 4 voxels per axis')
    return volume.astype(np.float32), spacing_mm

def build_target_planes(volume, axis=2, fractions=(.35, .5, .65), size=128):
    """Normalize anatomical intensity targets; not tissue optical transmittance."""
    arr = np.asarray(volume, dtype=np.float32)
    if arr.ndim != 3 or axis not in (0,1,2) or size < 16 or not np.isfinite(arr).all():
        raise ValueError('Invalid volume, axis, or requested output size')
    if not fractions or any(not 0 <= f <= 1 for f in fractions):
        raise ValueError('Slice fraction must be within [0,1]')
    indices = [int(round(f*(arr.shape[axis]-1))) for f in fractions]
    selected = np.stack([np.take(arr, k, axis=axis) for k in indices])
    lower,upper=np.percentile(selected,[1,99.5])
    if upper<=lower: raise ValueError('Selected slices lack contrast')
    selected=np.clip((selected-lower)/(upper-lower),0,1)
    output=np.stack([zoom(image,(size/image.shape[0],size/image.shape[1]),order=1) for image in selected])
    if output.shape != (len(fractions),size,size):
        raise RuntimeError(f'Unexpected resample shape: {output.shape}')
    return output.astype(np.float32),indices
