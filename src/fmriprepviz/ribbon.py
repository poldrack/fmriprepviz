"""Derive white and pial surface masks from FreeSurfer's ribbon.mgz.

The masks are filled binary volumes. Their boundaries — the surfaces we
actually want to draw — are produced by contouring them at level 0.5 at
render time, so no boundary extraction happens here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np

from fmriprepviz.discovery import AnatNotFoundError, find_unique_anat

RIBBON_RELATIVE_PATH = Path("mri") / "ribbon.mgz"
WHITE_LABELS = (2, 41)
GRAY_LABELS = (3, 42)
CORTICAL_LABELS = WHITE_LABELS + GRAY_LABELS

# The mask's boundary -- the surface we actually draw -- is its 0.5 level set.
# Stated once here because render.py contours at it and cuts.py and render.py
# both threshold at it.
MASK_THRESHOLD = 0.5

RIBBON_LABELS = (3, 42)  # left/right cortical gray matter
RIBBON_MASK_PATTERN = "*_desc-ribbon_mask.nii.gz"
T1W_PATTERN = "*_desc-preproc_T1w.nii.gz"
FS_TO_T1W_PATTERN = "*_from-fsnative_to-T1w_mode-image_xfm.txt"


class FreeSurferNotFoundError(RuntimeError):
    """No FreeSurfer subject directory containing a ribbon could be used."""


@dataclass(frozen=True)
class SurfaceMasks:
    """Filled binary masks whose contours are the white and pial surfaces."""

    white: nib.Nifti1Image
    pial: nib.Nifti1Image


def resolve_fs_dir(derivatives_root, subject: str, fs_dir=None) -> Path:
    """Find the FreeSurfer subject directory holding ribbon.mgz."""
    candidates = _fs_dir_candidates(derivatives_root, subject, fs_dir)
    for candidate in candidates:
        if (candidate / RIBBON_RELATIVE_PATH).is_file():
            return candidate
    tried = "\n  ".join(str(c / RIBBON_RELATIVE_PATH) for c in candidates)
    raise FreeSurferNotFoundError(
        "Could not find a FreeSurfer ribbon.mgz for "
        f"sub-{subject}. Tried:\n  {tried}\n"
        "Pass --fs-dir to point at the subject's FreeSurfer directory."
    )


def _fs_dir_candidates(derivatives_root, subject: str, fs_dir) -> list[Path]:
    if fs_dir is not None:
        return [Path(fs_dir)]
    candidates = [
        Path(derivatives_root) / "sourcedata" / "freesurfer" / f"sub-{subject}"
    ]
    subjects_dir = os.environ.get("SUBJECTS_DIR")
    if subjects_dir:
        candidates.append(Path(subjects_dir) / f"sub-{subject}")
    return candidates


def load_surface_masks(fs_dir) -> SurfaceMasks:
    """Build filled white and pial masks from a FreeSurfer ribbon."""
    ribbon_path = Path(fs_dir) / RIBBON_RELATIVE_PATH
    if not ribbon_path.is_file():
        raise FreeSurferNotFoundError(f"Expected a ribbon at {ribbon_path}")
    ribbon = nib.load(ribbon_path)
    data = np.asarray(ribbon.dataobj)
    _require_cortical_labels(data, ribbon_path)
    return SurfaceMasks(
        white=_mask_image(np.isin(data, WHITE_LABELS), ribbon.affine),
        pial=_mask_image(np.isin(data, WHITE_LABELS + GRAY_LABELS), ribbon.affine),
    )


def load_ribbon_mask(
    derivatives_root, subject: str, ribbon=None, fs_dir=None, xfm=None
) -> nib.Nifti1Image:
    """Return the cortical ribbon mask in the fMRIPrep T1w space.

    Resolution order:

    1. an explicitly given `ribbon` path
    2. an explicitly given `fs_dir` or `xfm`, which selects the FreeSurfer
       path even when fMRIPrep's own mask is present -- silently ignoring a
       flag the user went out of their way to pass is its own bug
    3. fMRIPrep's `desc-ribbon_mask`, already in T1w space
    4. FreeSurfer's `ribbon.mgz`, resampled through the fsnative-to-T1w
       transform

    An absent `desc-ribbon_mask` falls through to FreeSurfer; an ambiguous one
    does not, because that ambiguity is the user's to resolve.
    """
    if ribbon is not None:
        return nib.load(ribbon)
    absent = None
    if fs_dir is None and xfm is None:
        try:
            return nib.load(
                find_unique_anat(derivatives_root, subject, RIBBON_MASK_PATTERN)
            )
        except AnatNotFoundError as error:
            # Only "absent" falls through. An AmbiguousAnatError is the user's
            # to resolve and must not be worked around by quietly taking a
            # different path that happens to succeed.
            absent = error
    try:
        return _ribbon_from_freesurfer(derivatives_root, subject, fs_dir, xfm)
    except (AnatNotFoundError, FreeSurferNotFoundError) as fallback_error:
        if absent is None:
            raise
        # Neither source worked. Reporting only the second failure would name
        # a missing desc-preproc_T1w and never mention the ribbon mask, which
        # is the file most users are actually missing.
        raise FreeSurferNotFoundError(
            f"Could not find a cortical ribbon for sub-{subject}.\n"
            f"  fMRIPrep ribbon mask: {absent}\n"
            f"  FreeSurfer fallback: {fallback_error}\n"
            "Pass --ribbon to name a T1w-space ribbon mask directly."
        ) from fallback_error


def _ribbon_from_freesurfer(derivatives_root, subject: str, fs_dir, xfm):
    raise FreeSurferNotFoundError(
        "Building the ribbon from FreeSurfer is not implemented yet."
    )


def _mask_image(mask: np.ndarray, affine: np.ndarray) -> nib.Nifti1Image:
    return nib.Nifti1Image(mask.astype(np.uint8), affine)


def _require_cortical_labels(data: np.ndarray, ribbon_path: Path) -> None:
    """Fail clearly if none of the expected cortical labels are present.

    Without this check, a ribbon with no matching voxels (a failed
    recon-all, or an aseg.mgz mistaken for the ribbon) silently produces
    all-zero masks, and the eventual failure is numpy's opaque "zero-size
    array to reduction operation minimum which has no identity" once cuts
    are derived from the empty pial mask.
    """
    if not np.isin(data, CORTICAL_LABELS).any():
        raise FreeSurferNotFoundError(
            f"{ribbon_path} contains none of the expected cortical labels "
            f"{CORTICAL_LABELS}. This usually means recon-all failed, or "
            "--fs-dir points at a whole-brain segmentation (aseg.mgz) "
            "rather than the cortical ribbon (ribbon.mgz)."
        )
