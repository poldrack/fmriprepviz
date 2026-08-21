import nibabel as nib
import numpy as np
import pytest

from conftest import (
    ANAT_FRACTIONAL_OFFSET_MM,
    ANAT_OFFSET_MM,
    PIAL_RADIUS_MM,
    RIBBON_VOXEL_SIZE,
    WHITE_RADIUS_MM,
)
from fmriprepviz.discovery import AmbiguousAnatError
from fmriprepviz.ribbon import (
    MASK_THRESHOLD,
    RIBBON_LABELS,
    FreeSurferNotFoundError,
    load_ribbon_mask,
)

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


def _centroid(mask_img):
    data = np.asarray(mask_img.dataobj)
    ijk = np.argwhere(data >= MASK_THRESHOLD)
    return nib.affines.apply_affine(mask_img.affine, ijk).mean(axis=0)


def test_fallback_reproduces_the_fmriprep_mask_on_an_aligned_tree(derivatives):
    """With the two spaces coincident the transform is identity, so both paths
    must agree. This is what a reviewer checks against pre-change output.
    """
    fs_dir = derivatives / "sourcedata" / "freesurfer" / "sub-01"
    fast = load_ribbon_mask(derivatives, "01")
    fallback = load_ribbon_mask(derivatives, "01", fs_dir=fs_dir)

    assert fallback.shape == fast.shape
    assert np.allclose(fallback.affine, fast.affine)
    a = np.asarray(fast.dataobj) >= MASK_THRESHOLD
    b = np.asarray(fallback.dataobj) >= MASK_THRESHOLD
    dice = 2 * (a & b).sum() / (a.sum() + b.sum())
    assert dice > 0.95, f"paths disagree on an aligned tree: dice={dice:.4f}"


def test_fallback_applies_the_transform_in_the_right_direction(
    derivatives_misaligned,
):
    """The sign trap, and the reason this change exists.

    In the misaligned tree the FreeSurfer anatomy sits ANAT_OFFSET_MM away in
    +y from the T1w anatomy. Applying the transform correctly brings it back
    onto the T1w anatomy; applying it inverted, or not at all, leaves it
    displaced by ANAT_OFFSET_MM or moves it twice as far. All three outcomes
    render without error and look plausible, which is exactly why this is
    asserted numerically rather than left to inspection.
    """
    fs_dir = (
        derivatives_misaligned / "sourcedata" / "freesurfer" / "sub-01"
    )
    fast = load_ribbon_mask(derivatives_misaligned, "01")
    fallback = load_ribbon_mask(derivatives_misaligned, "01", fs_dir=fs_dir)

    expected = _centroid(fast)
    actual = _centroid(fallback)
    assert np.allclose(actual, expected, atol=RIBBON_VOXEL_SIZE), (
        f"aligned ribbon centroid {actual} does not match the fMRIPrep "
        f"mask's {expected}"
    )
    # And it must not merely be "somewhere": the uncorrected and
    # doubly-corrected positions are both far away and both must be excluded.
    assert abs(actual[1] - ANAT_OFFSET_MM) > ANAT_OFFSET_MM / 2, (
        "ribbon left at its uncorrected fsnative position: the transform was "
        "dropped"
    )
    assert abs(actual[1] - 2 * ANAT_OFFSET_MM) > ANAT_OFFSET_MM / 2, (
        "ribbon moved the wrong way: the transform was inverted"
    )


def test_fallback_mask_is_not_binary_at_a_fractional_offset(
    derivatives_misaligned_fractional,
):
    """A dtype that silently discards interpolation is invisible on a
    whole-voxel offset: map_coordinates never has to interpolate when the
    displacement lands exactly on a voxel boundary, so a staircase and a
    true sub-voxel level set produce identical output there. This fixture's
    offset does not land on a voxel boundary, so it is the one that can
    actually tell them apart.
    """
    fs_dir = (
        derivatives_misaligned_fractional
        / "sourcedata"
        / "freesurfer"
        / "sub-01"
    )
    fallback = load_ribbon_mask(
        derivatives_misaligned_fractional, "01", fs_dir=fs_dir
    )
    data = np.asarray(fallback.dataobj)
    assert np.any((data > 0) & (data < 1)), (
        "fallback mask contains no values strictly between 0 and 1: "
        "interpolation was discarded, most likely by resampling an "
        "integer-dtype moving image"
    )


def test_fallback_applies_the_transform_in_the_right_direction_at_a_fractional_offset(
    derivatives_misaligned_fractional,
):
    """The sign trap repeated where resampling actually interpolates.

    ANAT_OFFSET_MM lands on a whole voxel, so the direction check above
    could in principle pass even if interpolation itself were broken. This
    fixture's fractional offset forces map_coordinates to interpolate, so
    this is the direction check that depends on order=1 actually working.
    """
    fs_dir = (
        derivatives_misaligned_fractional
        / "sourcedata"
        / "freesurfer"
        / "sub-01"
    )
    fast = load_ribbon_mask(derivatives_misaligned_fractional, "01")
    fallback = load_ribbon_mask(
        derivatives_misaligned_fractional, "01", fs_dir=fs_dir
    )

    expected = _centroid(fast)
    actual = _centroid(fallback)
    assert np.allclose(actual, expected, atol=RIBBON_VOXEL_SIZE), (
        f"aligned ribbon centroid {actual} does not match the fMRIPrep "
        f"mask's {expected}"
    )
    assert (
        abs(actual[1] - ANAT_FRACTIONAL_OFFSET_MM)
        > ANAT_FRACTIONAL_OFFSET_MM / 2
    ), (
        "ribbon left at its uncorrected fsnative position: the transform "
        "was dropped"
    )
    assert (
        abs(actual[1] - 2 * ANAT_FRACTIONAL_OFFSET_MM)
        > ANAT_FRACTIONAL_OFFSET_MM / 2
    ), "ribbon moved the wrong way: the transform was inverted"


def test_explicit_xfm_selects_the_freesurfer_path(derivatives_misaligned):
    """Passing --fs-to-t1w-xfm must not be ignored just because a
    desc-ribbon_mask happens to exist, or the flag would be a no-op and the
    FreeSurfer path would be unreachable on any complete fMRIPrep tree.
    """
    xfm = next(
        (derivatives_misaligned / "sub-01").glob(
            "**/anat/*_from-fsnative_to-T1w_mode-image_xfm.txt"
        )
    )
    mask = load_ribbon_mask(derivatives_misaligned, "01", xfm=xfm)
    expected = _centroid(load_ribbon_mask(derivatives_misaligned, "01"))
    assert np.allclose(_centroid(mask), expected, atol=RIBBON_VOXEL_SIZE)


def test_ribbon_landing_nowhere_on_the_t1w_grid_is_reported(
    derivatives, tmp_path
):
    """A --fs-dir pointing at another subject used to produce a mask with a
    wildly wrong affine, which nilearn caught later as a BoundingBoxError.
    Now the mask is always resampled onto the T1w grid, so the same mistake
    yields an all-zero mask instead -- and without this guard the failure
    would surface as numpy's opaque "zero-size array to reduction operation
    minimum which has no identity" once cuts are derived from it.
    """
    fs_dir = tmp_path / "wrong-subject-fs"
    mri = fs_dir / "mri"
    mri.mkdir(parents=True)
    affine = np.eye(4)
    affine[:3, 3] = 5000.0  # nowhere near the T1w grid
    data = np.zeros((20, 20, 20), dtype=np.uint8)
    data[5:15, 5:15, 5:15] = 3  # a real cortical gray label, wrongly placed
    nib.MGHImage(data, affine).to_filename(mri / "ribbon.mgz")

    with pytest.raises(FreeSurferNotFoundError) as excinfo:
        load_ribbon_mask(derivatives, "01", fs_dir=fs_dir)
    message = str(excinfo.value)
    assert "ribbon.mgz" in message
    assert "T1w" in message


def test_ribbon_without_gray_matter_labels_is_reported(derivatives, tmp_path):
    """A ribbon.mgz with no cortical gray labels -- a failed recon-all, or an
    aseg.mgz mistaken for a ribbon -- must be named before any resampling.
    """
    fs_dir = tmp_path / "no-gray-fs"
    mri = fs_dir / "mri"
    mri.mkdir(parents=True)
    # Label 4 (left lateral ventricle) is a real aseg label but not cortex.
    data = np.full((10, 10, 10), 4, dtype=np.uint8)
    nib.MGHImage(data, np.eye(4)).to_filename(mri / "ribbon.mgz")

    with pytest.raises(FreeSurferNotFoundError) as excinfo:
        load_ribbon_mask(derivatives, "01", fs_dir=fs_dir)
    message = str(excinfo.value)
    assert str(mri / "ribbon.mgz") in message
    # Assert the exact tuple: "2" is a substring of "42", so per-label
    # substring checks silently pass against the wrong label set.
    assert str(RIBBON_LABELS) in message


def test_no_ribbon_at_all_names_both_sources(tmp_path, monkeypatch):
    """With neither a desc-ribbon_mask nor a FreeSurfer directory, reporting
    only the second failure would name a missing desc-preproc_T1w and never
    mention the ribbon mask -- which is the file most users are missing.
    """
    monkeypatch.delenv("SUBJECTS_DIR", raising=False)
    (tmp_path / "sub-01").mkdir()
    with pytest.raises(FreeSurferNotFoundError) as excinfo:
        load_ribbon_mask(tmp_path, "01")
    message = str(excinfo.value)
    assert "desc-ribbon_mask" in message
    assert "--ribbon" in message
