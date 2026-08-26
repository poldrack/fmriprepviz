"""The synthetic dataset is the foundation of every other test, so its own
geometry is verified first."""

import nibabel as nib
import numpy as np

from conftest import (
    ANAT_OFFSET_MM,
    BOLDREF_ENTITIES,
    PIAL_RADIUS_MM,
    RIBBON_ORIGIN,
    RIBBON_VOXEL_SIZE,
    WHITE_RADIUS_MM,
)


def _voxel_radii(mask, affine):
    ijk = np.argwhere(mask)
    homogeneous = np.column_stack([ijk, np.ones(len(ijk))])
    world = (affine @ homogeneous.T).T[:, :3]
    return np.linalg.norm(world, axis=1)


def test_tree_contains_one_t1w_boldref_per_entity_set(derivatives):
    found = sorted((derivatives / "sub-01").glob("**/func/*_space-T1w_boldref.nii.gz"))
    assert len(found) == len(BOLDREF_ENTITIES)


def test_tree_contains_a_non_t1w_boldref(derivatives):
    found = list((derivatives / "sub-01").glob("**/func/*space-MNI*_boldref.nii.gz"))
    assert len(found) == 1


def test_ribbon_uses_real_freesurfer_labels(derivatives):
    ribbon = nib.load(
        derivatives / "sourcedata" / "freesurfer" / "sub-01" / "mri" / "ribbon.mgz"
    )
    labels = set(np.unique(np.asarray(ribbon.dataobj)).tolist())
    assert labels == {0, 2, 3, 41, 42}


def test_white_labels_fill_a_sphere_of_the_expected_radius(derivatives):
    ribbon = nib.load(
        derivatives / "sourcedata" / "freesurfer" / "sub-01" / "mri" / "ribbon.mgz"
    )
    data = np.asarray(ribbon.dataobj)
    radii = _voxel_radii(np.isin(data, (2, 41)), ribbon.affine)
    assert radii.max() < WHITE_RADIUS_MM
    assert radii.max() > WHITE_RADIUS_MM - 2 * RIBBON_VOXEL_SIZE


def test_gray_labels_extend_to_the_pial_radius(derivatives):
    ribbon = nib.load(
        derivatives / "sourcedata" / "freesurfer" / "sub-01" / "mri" / "ribbon.mgz"
    )
    data = np.asarray(ribbon.dataobj)
    radii = _voxel_radii(np.isin(data, (3, 42)), ribbon.affine)
    assert radii.max() < PIAL_RADIUS_MM
    assert radii.max() > PIAL_RADIUS_MM - 2 * RIBBON_VOXEL_SIZE
    assert radii.min() < WHITE_RADIUS_MM + 2 * RIBBON_VOXEL_SIZE


def test_ribbon_origin_places_the_sphere_centre_at_the_world_origin(derivatives):
    ribbon = nib.load(
        derivatives / "sourcedata" / "freesurfer" / "sub-01" / "mri" / "ribbon.mgz"
    )
    assert np.allclose(ribbon.affine[:3, 3], RIBBON_ORIGIN)


def test_anat_dir_has_a_t1w_space_ribbon_mask(derivatives):
    mask_path = (
        derivatives / "sub-01" / "anat" / "sub-01_desc-ribbon_mask.nii.gz"
    )
    mask = nib.load(mask_path)
    data = np.asarray(mask.dataobj)
    assert set(np.unique(data).tolist()) == {0, 1}
    radii = _voxel_radii(data > 0, mask.affine)
    assert radii.max() < PIAL_RADIUS_MM
    assert radii.max() > PIAL_RADIUS_MM - 2 * RIBBON_VOXEL_SIZE
    assert radii.min() > WHITE_RADIUS_MM - 2 * RIBBON_VOXEL_SIZE


def test_ribbon_mask_shares_the_t1w_grid(derivatives):
    anat = derivatives / "sub-01" / "anat"
    mask = nib.load(anat / "sub-01_desc-ribbon_mask.nii.gz")
    t1w = nib.load(anat / "sub-01_desc-preproc_T1w.nii.gz")
    assert mask.shape == t1w.shape
    assert np.allclose(mask.affine, t1w.affine)


def test_misaligned_fixture_really_is_misaligned(derivatives_misaligned):
    """If the offset were ever reduced to zero the sign-trap test in
    test_ribbon_mask.py would pass whether or not the transform is applied, so
    the displacement itself is asserted here.
    """
    ribbon = nib.load(
        derivatives_misaligned
        / "sourcedata" / "freesurfer" / "sub-01" / "mri" / "ribbon.mgz"
    )
    data = np.asarray(ribbon.dataobj)
    ijk = np.argwhere(np.isin(data, (3, 42)))
    centre = nib.affines.apply_affine(ribbon.affine, ijk).mean(axis=0)
    assert abs(centre[1] - ANAT_OFFSET_MM) < RIBBON_VOXEL_SIZE
    assert abs(centre[0]) < RIBBON_VOXEL_SIZE
    assert abs(centre[2]) < RIBBON_VOXEL_SIZE


def test_session_level_anat_tree_puts_anat_under_a_session(
    derivatives_session_anat,
):
    assert not (derivatives_session_anat / "sub-01" / "anat").exists()
    found = list(
        (derivatives_session_anat / "sub-01").glob(
            "**/anat/*_desc-ribbon_mask.nii.gz"
        )
    )
    assert len(found) == 1
    assert found[0].parent.parent.name == "ses-nsdanat"
