"""Synthetic fMRIPrep derivatives tree shared by the whole test suite.

The ribbon is built from concentric spheres of known radius, so tests can
assert that derived surfaces land at the correct anatomical location rather
than merely checking that an image was produced.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

RIBBON_SHAPE = (128, 128, 128)
RIBBON_VOXEL_SIZE = 1.0
RIBBON_ORIGIN = -64.0

BOLDREF_SHAPE = (48, 48, 48)
BOLDREF_VOXEL_SIZE = 3.0
BOLDREF_ORIGIN = -72.0

WHITE_RADIUS_MM = 30.0
PIAL_RADIUS_MM = 40.0

# How far the FreeSurfer anatomy sits from the T1w anatomy in the misaligned
# fixture, along +y. Deliberately large: a dropped or inverted transform must
# fail loudly rather than hide inside a tolerance.
ANAT_OFFSET_MM = 20.0

# Like ANAT_OFFSET_MM, but not a whole number of voxels at RIBBON_VOXEL_SIZE.
# ANAT_OFFSET_MM alone can't catch a resampled mask that silently collapses
# to an integer dtype: at a whole-voxel offset map_coordinates never has to
# interpolate, so a staircase and a true level set look identical.
ANAT_FRACTIONAL_OFFSET_MM = 20.5

# (session, task, run) for each space-T1w boldref in the synthetic tree.
# run-10 is present so ordering tests can catch naive string sorting.
BOLDREF_ENTITIES = [
    ("01", "rest", "1"),
    ("01", "rest", "2"),
    ("01", "rest", "10"),
    ("02", "rest", "1"),
    ("02", "nback", "1"),
]

MNI_BOLDREF_NAME = (
    "sub-01_ses-01_task-rest_run-1"
    "_space-MNI152NLin2009cAsym_res-2_boldref.nii.gz"
)


def _affine(voxel_size: float, origin: float) -> np.ndarray:
    affine = np.eye(4)
    affine[:3, :3] = np.diag([voxel_size] * 3)
    affine[:3, 3] = origin
    return affine


def _world_grids(shape, voxel_size, origin):
    axes = [np.arange(n) * voxel_size + origin for n in shape]
    return np.meshgrid(*axes, indexing="ij")


def _world_radius(shape, voxel_size, origin, centre=(0.0, 0.0, 0.0)):
    grids = _world_grids(shape, voxel_size, origin)
    radius = np.sqrt(sum((g - c) ** 2 for g, c in zip(grids, centre)))
    return radius, grids


def _ribbon_data(centre=(0.0, 0.0, 0.0)) -> np.ndarray:
    """Concentric spheres labelled with real FreeSurfer ribbon values.

    `centre` displaces the anatomy within an unchanged voxel grid, which is
    how the misaligned fixture puts the FreeSurfer anatomy somewhere the T1w
    anatomy is not.
    """
    radius, grids = _world_radius(
        RIBBON_SHAPE, RIBBON_VOXEL_SIZE, RIBBON_ORIGIN, centre
    )
    left = grids[0] < 0
    white = radius < WHITE_RADIUS_MM
    gray = (radius >= WHITE_RADIUS_MM) & (radius < PIAL_RADIUS_MM)

    data = np.zeros(RIBBON_SHAPE, dtype=np.uint8)
    data[white & left] = 2
    data[white & ~left] = 41
    data[gray & left] = 3
    data[gray & ~left] = 42
    return data


def _boldref_data(seed: int) -> np.ndarray:
    radius, _ = _world_radius(BOLDREF_SHAPE, BOLDREF_VOXEL_SIZE, BOLDREF_ORIGIN)
    rng = np.random.default_rng(seed)
    data = np.where(radius < PIAL_RADIUS_MM, 1000.0, 0.0)
    return (data + rng.normal(0.0, 20.0, BOLDREF_SHAPE)).astype(np.float32)


def _write_boldrefs(root: Path) -> None:
    for index, (ses, task, run) in enumerate(BOLDREF_ENTITIES):
        func = root / "sub-01" / f"ses-{ses}" / "func"
        func.mkdir(parents=True, exist_ok=True)
        name = (
            f"sub-01_ses-{ses}_task-{task}_run-{run}"
            "_space-T1w_boldref.nii.gz"
        )
        image = nib.Nifti1Image(
            _boldref_data(index), _affine(BOLDREF_VOXEL_SIZE, BOLDREF_ORIGIN)
        )
        image.to_filename(func / name)


def _write_mni_boldref(root: Path) -> None:
    """A non-T1w boldref, so error messages have another space to report."""
    func = root / "sub-01" / "ses-01" / "func"
    func.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(
        _boldref_data(99), _affine(BOLDREF_VOXEL_SIZE, BOLDREF_ORIGIN)
    )
    image.to_filename(func / MNI_BOLDREF_NAME)


def _write_ribbon(root: Path, offset_mm: float = 0.0) -> None:
    mri = root / "sourcedata" / "freesurfer" / "sub-01" / "mri"
    mri.mkdir(parents=True, exist_ok=True)
    ribbon = nib.MGHImage(
        _ribbon_data(centre=(0.0, offset_mm, 0.0)),
        _affine(RIBBON_VOXEL_SIZE, RIBBON_ORIGIN),
    )
    ribbon.to_filename(mri / "ribbon.mgz")


def _write_anat(
    root: Path,
    subject: str = "01",
    session: str | None = None,
    offset_mm: float = 0.0,
) -> Path:
    """Write the anat derivatives fMRIPrep produces: the T1w reference, the
    T1w-space cortical ribbon mask, and the fsnative-to-T1w transform.

    `offset_mm` is how far the FreeSurfer anatomy sits from the T1w anatomy
    along +y. The transform written here undoes exactly that displacement, so
    a correct FreeSurfer path must reproduce the T1w-space ribbon mask and an
    incorrect one must not.
    """
    anat = root / f"sub-{subject}"
    if session is not None:
        anat = anat / f"ses-{session}"
    anat = anat / "anat"
    anat.mkdir(parents=True, exist_ok=True)
    stem = f"sub-{subject}" + (f"_ses-{session}" if session else "")
    t1w_affine = _affine(RIBBON_VOXEL_SIZE, RIBBON_ORIGIN)

    # Only the grid matters: this image is used as a resampling reference and
    # its voxel values are never displayed.
    nib.Nifti1Image(
        np.zeros(RIBBON_SHAPE, dtype=np.float32), t1w_affine
    ).to_filename(anat / f"{stem}_desc-preproc_T1w.nii.gz")

    gray = np.isin(_ribbon_data(), (3, 42)).astype(np.uint8)
    nib.Nifti1Image(gray, t1w_affine).to_filename(
        anat / f"{stem}_desc-ribbon_mask.nii.gz"
    )

    # ITK stores the point map from the fixed image to the moving one, i.e.
    # T1w -> fsnative, in LPS. A T1w point at y=0 must sample the fsnative
    # point at y=+offset, so the RAS translation is (0, +offset, 0); LPS
    # negates x and y, hence the sign on the parameter below.
    (anat / f"{stem}_from-fsnative_to-T1w_mode-image_xfm.txt").write_text(
        "#Insight Transform File V1.0\n"
        "#Transform 0\n"
        "Transform: AffineTransform_float_3_3\n"
        f"Parameters: 1 0 0 0 1 0 0 0 1 0 {-offset_mm} 0\n"
        "FixedParameters: 0 0 0\n"
    )
    return anat


@pytest.fixture(scope="session")
def derivatives(tmp_path_factory) -> Path:
    """A complete synthetic tree whose FreeSurfer and T1w spaces coincide."""
    root = tmp_path_factory.mktemp("derivatives")
    _write_boldrefs(root)
    _write_mni_boldref(root)
    _write_ribbon(root)
    _write_anat(root)
    return root


@pytest.fixture(scope="session")
def derivatives_without_t1w(tmp_path_factory) -> Path:
    """A tree fMRIPrep produced without T1w among its output spaces."""
    root = tmp_path_factory.mktemp("mni_only")
    _write_mni_boldref(root)
    _write_ribbon(root)
    return root


@pytest.fixture(scope="session")
def derivatives_misaligned(tmp_path_factory) -> Path:
    """A tree where recon-all ran outside fMRIPrep: the FreeSurfer anatomy is
    ANAT_OFFSET_MM from the T1w anatomy, and the transform is the only thing
    that reconciles them.
    """
    root = tmp_path_factory.mktemp("misaligned")
    _write_boldrefs(root)
    _write_ribbon(root, offset_mm=ANAT_OFFSET_MM)
    _write_anat(root, offset_mm=ANAT_OFFSET_MM)
    return root


@pytest.fixture(scope="session")
def derivatives_misaligned_fractional(tmp_path_factory) -> Path:
    """Like `derivatives_misaligned`, but offset by a fractional number of
    voxels so resampling must actually interpolate.

    `derivatives_misaligned` alone cannot catch a mask that silently loses
    its sub-voxel precision to an integer dtype: at a whole-voxel offset,
    map_coordinates never interpolates, so a staircase and a true level set
    produce the same output.
    """
    root = tmp_path_factory.mktemp("misaligned_fractional")
    _write_boldrefs(root)
    _write_ribbon(root, offset_mm=ANAT_FRACTIONAL_OFFSET_MM)
    _write_anat(root, offset_mm=ANAT_FRACTIONAL_OFFSET_MM)
    return root


@pytest.fixture(scope="session")
def derivatives_session_anat(tmp_path_factory) -> Path:
    """fMRIPrep writes anat outputs under ses-<label>/anat when sessions carry
    their own anatomicals, so discovery must glob rather than assume sub/anat.
    """
    root = tmp_path_factory.mktemp("session_anat")
    _write_boldrefs(root)
    _write_ribbon(root)
    _write_anat(root, session="nsdanat")
    return root


@pytest.fixture(scope="session")
def surface_masks(derivatives):
    """White and pial masks derived from the synthetic ribbon."""
    from fmriprepviz.ribbon import load_surface_masks, resolve_fs_dir

    return load_surface_masks(resolve_fs_dir(derivatives, "01"))


@pytest.fixture(scope="session")
def ribbon_mask(derivatives):
    """The T1w-space cortical ribbon mask from the synthetic tree."""
    from fmriprepviz.ribbon import load_ribbon_mask

    return load_ribbon_mask(derivatives, "01")
