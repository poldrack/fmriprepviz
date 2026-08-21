"""Derive the cortical gray-matter ribbon mask, always in the T1w space.

The mask is a filled binary (or, after resampling, continuous 0..1) volume:
gray matter is a sheet, so contouring it at level 0.5 at render time traces
the white surface along its inner boundary and the pial along its outer, and
no separate boundary extraction happens here.
"""

from __future__ import annotations

import os
from pathlib import Path

import nibabel as nib
import nitransforms as nt
import numpy as np
from nitransforms.resampling import apply as nt_apply

from fmriprepviz.discovery import AnatNotFoundError, find_unique_anat

RIBBON_RELATIVE_PATH = Path("mri") / "ribbon.mgz"

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
    """Build the ribbon mask from FreeSurfer, moved into the T1w space.

    Resampling rather than composing the transform into the affine is
    deliberate: it puts this mask on the same grid as fMRIPrep's own
    desc-ribbon_mask, so nothing downstream can behave differently depending
    on which path produced it. The cost is sub-voxel -- around 0.3 mm against
    a BOLD voxel several times that.
    """
    t1w_path = find_unique_anat(derivatives_root, subject, T1W_PATTERN)
    xfm_path = (
        Path(xfm)
        if xfm is not None
        else find_unique_anat(derivatives_root, subject, FS_TO_T1W_PATTERN)
    )
    ribbon_path = (
        resolve_fs_dir(derivatives_root, subject, fs_dir) / RIBBON_RELATIVE_PATH
    )
    ribbon_img = nib.load(ribbon_path)
    data = np.asarray(ribbon_img.dataobj)
    _require_gray_matter_labels(data, ribbon_path)

    # float32, not uint8: nitransforms.resampling.apply propagates the
    # moving image's dtype through map_coordinates, so an integer mask here
    # would round every interpolated value back to a staircase, silently
    # discarding the sub-voxel accuracy order=1 is meant to provide.
    gray = nib.Nifti1Image(
        np.isin(data, RIBBON_LABELS).astype(np.float32), ribbon_img.affine
    )
    # The ITK file stores the point map from the T1w (fixed) image to fsnative
    # (moving), which is the direction resampling.apply pulls, so it is used
    # as loaded. Inverting it here would displace the ribbon by twice the
    # true offset, silently.
    transform = nt.linear.load(xfm_path, fmt="itk")
    aligned = nt_apply(
        transform, gray, reference=nib.load(t1w_path), order=1
    )
    _require_overlap_on_the_t1w_grid(aligned, ribbon_path, xfm_path)
    return aligned


def _require_gray_matter_labels(data: np.ndarray, ribbon_path: Path) -> None:
    """Fail clearly if none of the cortical gray labels are present."""
    if not np.isin(data, RIBBON_LABELS).any():
        raise FreeSurferNotFoundError(
            f"{ribbon_path} contains none of the expected cortical gray "
            f"matter labels {RIBBON_LABELS}. This usually means recon-all "
            "failed, or --fs-dir points at a whole-brain segmentation "
            "(aseg.mgz) rather than the cortical ribbon (ribbon.mgz)."
        )


def _require_overlap_on_the_t1w_grid(
    mask_img, ribbon_path: Path, xfm_path: Path
) -> None:
    """Fail clearly when alignment leaves nothing on the T1w grid."""
    if not (np.asarray(mask_img.dataobj) >= MASK_THRESHOLD).any():
        raise FreeSurferNotFoundError(
            f"No cortical gray matter from {ribbon_path} lands on the T1w "
            f"grid after applying {xfm_path}. This usually means --fs-dir "
            "points at a different subject, or that transform does not "
            "relate these two images."
        )
