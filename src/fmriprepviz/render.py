"""Render one BOLD reference image as a labelled QC frame."""

from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from nilearn import image, plotting  # noqa: E402

from fmriprepviz.ribbon import MASK_THRESHOLD

FIGSIZE = (12.0, 6.0)
# Three slices side by side are limited by width, not height, so the tall
# frame the stacked layout needs would leave over half of this one black.
SINGLE_CUT_FIGSIZE = (12.0, 3.6)
DPI = 100
FRAME_SIZE = (int(FIGSIZE[0] * DPI), int(FIGSIZE[1] * DPI))
SINGLE_CUT_FRAME_SIZE = (
    int(SINGLE_CUT_FIGSIZE[0] * DPI),
    int(SINGLE_CUT_FIGSIZE[1] * DPI),
)
CONTOUR_COLOR = "red"
CONTOUR_LEVELS = [MASK_THRESHOLD]
CONTOUR_LINEWIDTH = 0.6
WINDOW_PERCENTILES = (2.0, 98.0)


NO_OVERLAP_MESSAGE = (
    "The cortical ribbon mask and the BOLD reference do not overlap on the "
    "display grid. This usually means the ribbon and the BOLD reference have "
    "mismatched affines -- a wrong or missing fsnative-to-T1w transform, or "
    "--fs-dir or --ribbon pointing at the wrong subject."
)


class NoMaskOverlapError(RuntimeError):
    """The ribbon mask shares no voxels with the BOLD reference's display grid."""


def render_frame(
    boldref_img, ribbon_mask, cuts, label: str, upsample: bool = True
) -> bytes:
    """Draw one frame and return it as PNG bytes."""
    display_img = _prepare(boldref_img, ribbon_mask, upsample)
    vmin, vmax = _window(display_img, ribbon_mask)
    nrows, ncols = _grid_shape(cuts)
    figsize = SINGLE_CUT_FIGSIZE if _is_single_cut(cuts) else FIGSIZE
    figure, axes = plt.subplots(
        nrows, ncols, figsize=figsize, dpi=DPI, facecolor="black"
    )
    try:
        for axis, (mode, coordinates) in zip(np.atleast_1d(axes), cuts.items()):
            _draw_row(
                figure, axis, display_img, ribbon_mask, mode, coordinates,
                vmin, vmax,
            )
        figure.subplots_adjust(
            left=0, right=1, top=0.94, bottom=0.0, hspace=0.05, wspace=0.02
        )
        figure.text(0.01, 0.985, label, color="white", va="top", fontsize=11)
        return _to_png(figure)
    finally:
        plt.close(figure)


def _is_single_cut(cuts) -> bool:
    return all(len(coordinates) == 1 for coordinates in cuts.values())


def _grid_shape(cuts) -> tuple[int, int]:
    """Lay the orientations out in a row when each shows a single slice.

    A row of cuts is wide and short, so stacking several of them fills a
    landscape frame well. A single slice is not: stacked, each panel is
    limited by its row's height and comes out barely larger than one cut of
    the multi-cut view. Side by side, each panel gets a full share of the
    width instead.
    """
    if _is_single_cut(cuts):
        return 1, len(cuts)
    return len(cuts), 1


def frame_size(cuts) -> tuple[int, int]:
    """Pixel dimensions of a frame drawn with these cuts.

    Fixed for any one movie, since the cuts do not change between frames.
    """
    return SINGLE_CUT_FRAME_SIZE if _is_single_cut(cuts) else FRAME_SIZE


def _prepare(boldref_img, reference_img, upsample: bool):
    """Resample the BOLD onto the ribbon mask's grid.

    The display grid must be pinned to the same grid for every frame: with
    interpolation switched off, nilearn instead sets each axis's limits from
    the bounds of what is actually drawn on it, so two boldrefs that differ
    only in field of view would render the same anatomy at different zoom
    levels -- moving the red contours on screen, the exact artefact this
    tool exists to rule out as misregistration. So `upsample` no longer
    decides *whether* to resample, only *how*: `linear` gives a smoothly
    upsampled image, `nearest` keeps voxels visibly blocky (no
    interpolation) while still pinning the display grid. The ribbon mask
    now always shares the T1w grid, so the pinned display grid is the T1w
    grid.
    """
    interpolation = "linear" if upsample else "nearest"
    return image.resample_to_img(
        boldref_img,
        reference_img,
        interpolation=interpolation,
        force_resample=True,
        copy_header=True,
    )


def _window(display_img, mask_img) -> tuple[float, float]:
    """Robust intensity limits from inside the brain only.

    Sessions carry arbitrary scanner intensity units, so a shared global
    window would make the movie flicker. Masked robust scaling keeps
    brightness comparable without concealing anything.
    """
    mask = image.resample_to_img(
        mask_img,
        display_img,
        interpolation="nearest",
        force_resample=True,
        copy_header=True,
    )
    data = np.asarray(display_img.dataobj, dtype=float)
    inside = data[np.asarray(mask.dataobj) >= MASK_THRESHOLD]
    if inside.size == 0:
        raise NoMaskOverlapError(NO_OVERLAP_MESSAGE)
    low, high = np.percentile(inside, WINDOW_PERCENTILES)
    return float(low), float(high)


def _draw_row(
    figure, axis, display_img, ribbon_mask, mode, coordinates, vmin, vmax
):
    display = plotting.plot_anat(
        display_img,
        figure=figure,
        axes=axis,
        display_mode=mode,
        cut_coords=coordinates,
        vmin=vmin,
        vmax=vmax,
        black_bg=True,
        annotate=True,
        draw_cross=False,
        colorbar=False,
    )
    # The mask is a gray-matter sheet, so this single level set traces the
    # white surface along its inner boundary and the pial along its outer.
    display.add_contours(
        ribbon_mask,
        levels=CONTOUR_LEVELS,
        colors=CONTOUR_COLOR,
        linewidths=CONTOUR_LINEWIDTH,
    )


def _to_png(figure) -> bytes:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=DPI, facecolor="black")
    return buffer.getvalue()
