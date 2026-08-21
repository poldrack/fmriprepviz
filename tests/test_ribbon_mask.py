import nibabel as nib
import numpy as np
import pytest

from conftest import PIAL_RADIUS_MM, RIBBON_VOXEL_SIZE, WHITE_RADIUS_MM
from fmriprepviz.discovery import AmbiguousAnatError
from fmriprepviz.ribbon import MASK_THRESHOLD, load_ribbon_mask

TOLERANCE_MM = 2 * RIBBON_VOXEL_SIZE


def _foreground_radii(mask_img):
    data = np.asarray(mask_img.dataobj)
    ijk = np.argwhere(data >= MASK_THRESHOLD)
    world = nib.affines.apply_affine(mask_img.affine, ijk)
    return np.linalg.norm(world, axis=1)


def test_fast_path_reads_the_fmriprep_ribbon_mask(derivatives):
    mask = load_ribbon_mask(derivatives, "01")
    expected = nib.load(
        derivatives / "sub-01" / "anat" / "sub-01_desc-ribbon_mask.nii.gz"
    )
    assert np.allclose(mask.affine, expected.affine)
    assert mask.shape == expected.shape


def test_mask_spans_the_cortical_ribbon_only(derivatives):
    """The mask is gray matter alone, so one contour at 0.5 gives the white
    surface as its inner level set and the pial as its outer. A mask that
    included white matter would collapse the two into one curve.
    """
    radii = _foreground_radii(load_ribbon_mask(derivatives, "01"))
    assert radii.max() < PIAL_RADIUS_MM
    assert radii.max() > PIAL_RADIUS_MM - TOLERANCE_MM
    assert radii.min() > WHITE_RADIUS_MM - TOLERANCE_MM


def test_mask_lands_on_the_t1w_grid(derivatives):
    """Everything downstream assumes the mask shares the boldrefs' space, and
    that assumption is the one this whole change exists to establish.
    """
    mask = load_ribbon_mask(derivatives, "01")
    t1w = nib.load(
        derivatives / "sub-01" / "anat" / "sub-01_desc-preproc_T1w.nii.gz"
    )
    assert mask.shape == t1w.shape
    assert np.allclose(mask.affine, t1w.affine)


def test_explicit_ribbon_path_wins(derivatives, tmp_path):
    explicit = tmp_path / "mine_desc-ribbon_mask.nii.gz"
    data = np.zeros((8, 8, 8), dtype=np.uint8)
    data[2:6, 2:6, 2:6] = 1
    nib.Nifti1Image(data, np.eye(4)).to_filename(explicit)
    mask = load_ribbon_mask(derivatives, "01", ribbon=explicit)
    assert mask.shape == (8, 8, 8)


def test_ambiguous_ribbon_mask_is_not_silently_resolved(tmp_path):
    """Falling through to FreeSurfer here would hide a real ambiguity behind
    a path that happens to work, so the error must propagate.
    """
    for session in ("a", "b"):
        anat = tmp_path / "sub-01" / f"ses-{session}" / "anat"
        anat.mkdir(parents=True)
        (anat / f"sub-01_ses-{session}_desc-ribbon_mask.nii.gz").touch()
    with pytest.raises(AmbiguousAnatError):
        load_ribbon_mask(tmp_path, "01")
