import io

import nibabel as nib
import numpy as np
import pytest
from nilearn.image.resampling import BoundingBoxError
from PIL import Image

from conftest import BOLDREF_VOXEL_SIZE, PIAL_RADIUS_MM, WHITE_RADIUS_MM
from fmriprepviz.cuts import select_cuts
from fmriprepviz.discovery import find_boldrefs
from fmriprepviz.render import (
    FRAME_SIZE,
    NoMaskOverlapError,
    frame_size,
    render_frame,
)

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@pytest.fixture(scope="module")
def cuts(ribbon_mask):
    return select_cuts(ribbon_mask, n_cuts=5)


@pytest.fixture(scope="module")
def boldrefs(derivatives):
    return find_boldrefs(derivatives, "01")


def _as_array(png_bytes):
    return np.asarray(Image.open(io.BytesIO(png_bytes)).convert("RGB"))


def _reddish_mask(png_bytes):
    """Boolean pixel mask of the drawn red contours.

    Same criterion as test_frame_contains_red_contour_pixels: the
    background is grayscale, so any strongly red pixel is a contour.
    """
    pixels = _as_array(png_bytes).astype(int)
    red, green, blue = pixels[..., 0], pixels[..., 1], pixels[..., 2]
    return (red > 120) & (red - green > 60) & (red - blue > 60)


def _reddish_bbox(png_bytes):
    """(x_min, x_max, y_min, y_max) of the drawn red contours.

    A handful of anti-aliased border pixels legitimately flicker in or out
    of the `_reddish_mask` threshold between two otherwise-identical renders
    whenever the underlying grayscale windowing differs even slightly (here,
    because the two boldrefs' independent noise draws give the masked
    intensity percentiles slightly different values) -- this is unrelated to
    contour placement, so a full pixel-mask equality check would be too
    strict. The bounding box is robust to that noise while still catching
    the actual defect: a display grid that is not pinned across frames
    shifts and rescales the whole contour, moving this box by many pixels,
    not flickering a handful of edge pixels.
    """
    ys, xs = np.where(_reddish_mask(png_bytes))
    return (int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max()))


def _sphere_boldref(shape, voxel_size=3.0, radius_mm=PIAL_RADIUS_MM, seed=0):
    """A synthetic boldref: a bright sphere of `radius_mm` centred at the
    world origin, on a grid of the given shape/voxel size. Two boldrefs
    built with the same radius_mm but different shape/voxel_size share the
    same anatomy but differ in field of view -- exactly the case that
    exposed the --no-upsample bug.
    """
    origin = -(shape[0] * voxel_size) / 2.0
    axes = [np.arange(n) * voxel_size + origin for n in shape]
    grids = np.meshgrid(*axes, indexing="ij")
    radius = np.sqrt(sum(g**2 for g in grids))
    signal = np.where(radius < radius_mm, 1000.0, 0.0)
    rng = np.random.default_rng(seed)
    data = (signal + rng.normal(0.0, 20.0, shape)).astype(np.float32)
    affine = np.eye(4)
    affine[:3, :3] = np.diag([voxel_size] * 3)
    affine[:3, 3] = origin
    return nib.Nifti1Image(data, affine)


def test_returns_png_bytes(derivatives, ribbon_mask, cuts, boldrefs):
    png = render_frame(
        nib.load(boldrefs[0].path), ribbon_mask, cuts, boldrefs[0].label
    )
    assert png.startswith(PNG_MAGIC)


def test_frame_has_the_fixed_size(ribbon_mask, cuts, boldrefs):
    png = render_frame(
        nib.load(boldrefs[0].path), ribbon_mask, cuts, boldrefs[0].label
    )
    assert Image.open(io.BytesIO(png)).size == FRAME_SIZE


def test_every_frame_has_identical_dimensions(ribbon_mask, cuts, boldrefs):
    """GIF and MP4 assembly require it, so it is checked rather than assumed."""
    sizes = {
        Image.open(
            io.BytesIO(render_frame(nib.load(ref.path), ribbon_mask, cuts, ref.label))
        ).size
        for ref in boldrefs[:3]
    }
    assert sizes == {FRAME_SIZE}


def test_rendering_is_deterministic(ribbon_mask, cuts, boldrefs):
    first = render_frame(
        nib.load(boldrefs[0].path), ribbon_mask, cuts, boldrefs[0].label
    )
    second = render_frame(
        nib.load(boldrefs[0].path), ribbon_mask, cuts, boldrefs[0].label
    )
    assert np.array_equal(_as_array(first), _as_array(second))


def test_frame_contains_red_contour_pixels(ribbon_mask, cuts, boldrefs):
    """The background is grayscale, so any strongly red pixel is a contour."""
    pixels = _as_array(
        render_frame(nib.load(boldrefs[0].path), ribbon_mask, cuts, boldrefs[0].label)
    ).astype(int)
    red, green, blue = pixels[..., 0], pixels[..., 1], pixels[..., 2]
    reddish = (red > 120) & (red - green > 60) & (red - blue > 60)
    assert reddish.sum() > 200


def test_one_contour_draws_both_the_white_and_pial_boundaries(
    ribbon_mask, boldrefs
):
    """The mask is cortical gray matter, a sheet, so its 0.5 level set has an
    inner boundary at the white surface and an outer one at the pial. Drawing
    a single contour must therefore produce two concentric red rings, not one.

    On an axial cut through the sphere's centre the central row crosses four
    red bands: outer-left, inner-left, inner-right, outer-right. Their
    distances from the row's centre must match the two known radii. A change
    that drew only the outer boundary, or filled the sheet, would pass every
    other test in this file.
    """
    png = render_frame(nib.load(boldrefs[0].path), ribbon_mask, {"z": [0.0]}, "label")
    row = _reddish_mask(png)[_as_array(png).shape[0] // 2]
    xs = np.where(row)[0]
    assert xs.size > 0, "no red contour pixels on the central row"

    # Split the run of red pixels into contiguous bands.
    breaks = np.where(np.diff(xs) > 1)[0]
    bands = np.split(xs, breaks + 1)
    assert len(bands) == 4, (
        f"expected 4 red bands (outer/inner on each side), got {len(bands)}: "
        f"{[(int(b.min()), int(b.max())) for b in bands]}"
    )

    centre = (xs.min() + xs.max()) / 2.0
    outer_px = max(abs(bands[0].mean() - centre), abs(bands[3].mean() - centre))
    inner_px = min(abs(bands[1].mean() - centre), abs(bands[2].mean() - centre))
    px_per_mm = outer_px / PIAL_RADIUS_MM
    measured_inner_mm = inner_px / px_per_mm
    assert abs(measured_inner_mm - WHITE_RADIUS_MM) <= 2 * BOLDREF_VOXEL_SIZE, (
        f"inner contour sits at {measured_inner_mm:.1f}mm, expected "
        f"{WHITE_RADIUS_MM}mm"
    )


def _bright_disc_mask(png_bytes):
    """Boolean pixel mask of the bright (grayscale, non-contour) disc."""
    pixels = _as_array(png_bytes).astype(int)
    red, green, blue = pixels[..., 0], pixels[..., 1], pixels[..., 2]
    grayscale = (np.abs(red - green) < 10) & (np.abs(green - blue) < 10)
    return grayscale & (red > 100)


def test_contours_land_on_the_anatomy_not_merely_somewhere(ribbon_mask, boldrefs):
    """Guards the join between mask geometry (verified in test_ribbon.py)
    and red-pixel presence (verified by test_frame_contains_red_contour_pixels
    above): neither of those checks would catch contours drawn at the wrong
    scale, offset, orientation, or on the wrong row, since both pass as long
    as red pixels exist *somewhere* with the right mask *shape*. A left-right
    flip or a half-voxel offset in the join between mask and boldref would
    ship looking correct to both and only be visible here.

    The synthetic fixture makes this tractable: the boldref's bright disc
    and the ribbon mask's outer (pial) boundary share a 40 mm radius
    (PIAL_RADIUS_MM), so on a single axial cut through the sphere's centre
    (z=0, its widest cross-section), the outer red ring must sit right on
    the bright disc's edge. This compares their positions along the frame's
    central row.

    Tolerance: one BOLD voxel's width, converted to pixels using the
    disc's own measured width (rather than a hard-coded pixel count) --
    the coarsest unit at which the underlying BOLD data can resolve the
    sphere's true edge in the first place, so agreement any tighter than
    that would be over-claiming precision the input data does not have.
    """
    single_axial_cut = {"z": [0.0]}
    png = render_frame(
        nib.load(boldrefs[0].path), ribbon_mask, single_axial_cut, "label"
    )
    pixels = _as_array(png)
    central_row = pixels.shape[0] // 2

    red_xs = np.where(_reddish_mask(png)[central_row])[0]
    bright_xs = np.where(_bright_disc_mask(png)[central_row])[0]
    assert red_xs.size > 0, "no red contour pixels on the central row"
    assert bright_xs.size > 0, "no bright disc pixels on the central row"

    disc_width_px = bright_xs.max() - bright_xs.min()
    px_per_mm = disc_width_px / (2.0 * PIAL_RADIUS_MM)
    tolerance_px = BOLDREF_VOXEL_SIZE * px_per_mm

    assert abs(red_xs.min() - bright_xs.min()) <= tolerance_px, (
        f"left contour edge {red_xs.min()} vs disc edge {bright_xs.min()}, "
        f"tolerance {tolerance_px:.1f}px"
    )
    assert abs(red_xs.max() - bright_xs.max()) <= tolerance_px, (
        f"right contour edge {red_xs.max()} vs disc edge {bright_xs.max()}, "
        f"tolerance {tolerance_px:.1f}px"
    )


def test_upsampling_disabled_still_renders(ribbon_mask, cuts, boldrefs):
    png = render_frame(
        nib.load(boldrefs[0].path),
        ribbon_mask,
        cuts,
        boldrefs[0].label,
        upsample=False,
    )
    assert Image.open(io.BytesIO(png)).size == FRAME_SIZE


def test_one_row_per_display_mode(ribbon_mask, boldrefs):
    """Fewer orientations must still produce a correctly sized frame."""
    single = select_cuts(ribbon_mask, n_cuts=5, display_modes=("z",))
    png = render_frame(
        nib.load(boldrefs[0].path), ribbon_mask, single, boldrefs[0].label
    )
    assert Image.open(io.BytesIO(png)).size == FRAME_SIZE


def test_non_overlapping_mask_is_reported(ribbon_mask, cuts, boldrefs):
    """A ribbon mask sharing the display grid exactly but containing no
    foreground must raise clearly rather than fall back to whole-volume
    intensity percentiles, which would reintroduce the cross-session flicker
    the masked windowing exists to prevent.
    """
    empty = nib.Nifti1Image(
        np.zeros_like(np.asarray(ribbon_mask.dataobj)), ribbon_mask.affine
    )
    with pytest.raises(NoMaskOverlapError):
        render_frame(
            nib.load(boldrefs[0].path), empty, cuts, boldrefs[0].label,
            upsample=False,
        )


def test_shifted_mask_raises_bounding_box_error_either_way(
    ribbon_mask, cuts, boldrefs
):
    """A wildly mismatched mask affine trips nilearn's own BoundingBoxError
    while placing the BOLD reference onto the mask grid, under both upsample
    values. Reaching render_frame with such a mask is now unusual -- ribbon.py
    catches a mismatched --fs-dir first -- but render_frame is a public entry
    point and must still fail cleanly rather than draw nonsense.
    """
    shifted_affine = ribbon_mask.affine.copy()
    shifted_affine[:3, 3] += 1000.0
    shifted = nib.Nifti1Image(
        np.asarray(ribbon_mask.dataobj), shifted_affine
    )
    for upsample in (True, False):
        with pytest.raises(BoundingBoxError):
            render_frame(
                nib.load(boldrefs[0].path), shifted, cuts,
                boldrefs[0].label, upsample=upsample,
            )


def test_display_pinned_across_differing_boldref_fov(ribbon_mask, cuts):
    """This is the direct expression of the tool's central invariant: masks
    and cuts are computed once and reused for every frame, so anything that
    appears to move between frames must be misregistration, never a display
    artefact. Two boldrefs here are identical except for field of view (a
    48^3 grid at 3 mm vs. a 100^3 grid at 3 mm), both containing a bright
    sphere of the same radius as the pial mask. If the display grid were not
    pinned, nilearn would set each axis's limits from the bounds of what is
    actually drawn on it, so the same anatomy would appear at a different
    zoom level -- moving the red contours on screen -- purely because of a
    FOV difference between sessions that has nothing to do with
    registration. This must hold under both upsample=True (already correct
    before this fix) and upsample=False (the bug Fix 1 corrects: previously
    it left the BOLD on its own native grid, letting the display zoom to
    that grid's own bounds).
    """
    small_fov = _sphere_boldref((48, 48, 48), voxel_size=3.0)
    large_fov = _sphere_boldref((100, 100, 100), voxel_size=3.0)

    for upsample in (True, False):
        small_png = render_frame(
            small_fov, ribbon_mask, cuts, "small-fov", upsample=upsample
        )
        large_png = render_frame(
            large_fov, ribbon_mask, cuts, "large-fov", upsample=upsample
        )
        small_bbox = _reddish_bbox(small_png)
        large_bbox = _reddish_bbox(large_png)
        assert small_bbox == large_bbox, (
            f"contour bounding box moved between differing FOVs with "
            f"upsample={upsample}: {small_bbox} vs {large_bbox}"
        )


def test_label_text_is_drawn_in_the_frame(ribbon_mask, cuts, boldrefs):
    """A regression that dropped, mis-coloured or mis-positioned the label
    would pass every other test, yet the label is the only thing telling a
    viewer which session a frame shows.
    """
    boldref = nib.load(boldrefs[0].path)
    first = _as_array(render_frame(boldref, ribbon_mask, cuts, "label-one"))
    second = _as_array(render_frame(boldref, ribbon_mask, cuts, "label-two"))
    top_rows = int(round(FRAME_SIZE[1] * 0.05))
    first_strip = first[:top_rows]
    second_strip = second[:top_rows]
    assert not np.array_equal(first_strip, second_strip)
    assert first_strip.std() > 0
    assert second_strip.std() > 0


def test_no_colorbar_is_drawn(ribbon_mask, cuts, boldrefs):
    """nilearn's plot_anat defaults to colorbar=True, adding a per-row
    colorbar whose labelled range is recomputed from each frame's own
    masked intensity window. That per-frame windowing is correct and
    deliberate, but this tool's premise is a display that never changes on
    its own: a number that ticks between frames for reasons unrelated to
    registration trains the eye on a meaningless signal, so no colorbar may
    be drawn at all.

    A right-edge strip is the natural place to look, but the true right
    edge of the frame is not where nilearn puts things: inspecting an
    actual rendered frame shows the colorbar's tick text and gradient bar
    sit around columns 997-1077 of 1200, with a ~120px fully-black margin
    beyond that to the frame's true edge -- so the rightmost 2% (or even
    10%) of columns is black whether or not the bug is present, and would
    not detect it. It is also necessary to clear the coronal/axial "R"
    orientation annotation, which sits just to the left of the colorbar
    today and shifts right (to end around column 1040) once the colorbar
    is removed, as the freed width is given back to the anatomical slices.
    The rightmost 12.5% of columns (the last 150 of 1200) was chosen by
    rendering both the current and fixed output and checking column
    contents directly: it reliably contains the colorbar in the current,
    buggy output, and reliably stays outside both the colorbar and the
    (shifted) "R" annotation once the colorbar is gone.
    """
    pixels = _as_array(
        render_frame(nib.load(boldrefs[0].path), ribbon_mask, cuts, boldrefs[0].label)
    )
    strip_width = int(round(FRAME_SIZE[0] * 0.125))
    right_strip = pixels[:, -strip_width:]
    assert np.array_equal(right_strip, np.zeros_like(right_strip))


def _content_by_horizontal_third(png_bytes):
    """Non-background pixel counts in the left, middle and right thirds,
    ignoring the top strip where the frame label is drawn."""
    pixels = _as_array(png_bytes)
    body = pixels[int(pixels.shape[0] * 0.08):]
    lit = body.max(axis=2) > 20
    third = lit.shape[1] // 3
    return [
        int(lit[:, :third].sum()),
        int(lit[:, third : 2 * third].sum()),
        int(lit[:, 2 * third :].sum()),
    ]


def test_single_cut_per_orientation_lays_the_panels_side_by_side(
    ribbon_mask, boldrefs
):
    """With one cut each, stacking the orientations as rows leaves every panel
    height-limited in a 1200x600 frame -- the slices come out barely larger
    than the seven-cut view they were meant to replace. Laying them out in a
    single row instead gives each panel a full third of the width.

    Stacked rows put all the anatomy in the horizontal middle, so requiring
    substantial content in the left and right thirds too is what actually
    distinguishes the two layouts.
    """
    single = select_cuts(ribbon_mask, n_cuts=1)
    png = render_frame(
        nib.load(boldrefs[0].path), ribbon_mask, single, boldrefs[0].label
    )
    left, middle, right = _content_by_horizontal_third(png)
    assert Image.open(io.BytesIO(png)).size == frame_size(single)
    assert left > 0.25 * middle, f"left third nearly empty: {left} vs {middle}"
    assert right > 0.25 * middle, f"right third nearly empty: {right} vs {middle}"


def test_multi_cut_layout_is_unchanged(ribbon_mask, cuts, boldrefs):
    """The seven-cut view keeps its row-per-orientation layout; only the
    single-cut case is rearranged."""
    png = render_frame(
        nib.load(boldrefs[0].path), ribbon_mask, cuts, boldrefs[0].label
    )
    assert Image.open(io.BytesIO(png)).size == FRAME_SIZE


def _content_rows(png_bytes):
    """Rows containing anything but background, ignoring the label strip."""
    pixels = _as_array(png_bytes)
    label_rows = int(round(pixels.shape[0] * 0.08))
    body = pixels[label_rows:]
    rows = np.nonzero((body.max(axis=2) > 20).any(axis=1))[0]
    return rows, body.shape[0]


def test_single_cut_frame_is_not_mostly_empty(ribbon_mask, boldrefs):
    """Three slices laid side by side are width-limited, so a 600px-tall frame
    leaves more than half its height black. That dead space shrinks the
    anatomy on screen for any given display width, which is the whole
    complaint the side-by-side layout was meant to answer.

    The frame height should track the content. Crucially this must trim empty
    space rather than shrink the picture, so the anatomy's own pixel height is
    asserted not to regress: making the frame shorter until the panels become
    height-limited would satisfy a fill-fraction check while making the slices
    smaller, which is the opposite of the point.
    """
    single = select_cuts(ribbon_mask, n_cuts=1)
    png = render_frame(
        nib.load(boldrefs[0].path), ribbon_mask, single, boldrefs[0].label
    )
    rows, body_height = _content_rows(png)
    anatomy_height = rows.max() - rows.min()

    assert anatomy_height >= 250, (
        f"anatomy shrank to {anatomy_height}px; trimming must remove empty "
        "space, not the picture"
    )
    fill = anatomy_height / body_height
    assert fill >= 0.75, f"frame is {(1 - fill) * 100:.0f}% empty vertically"


def test_multi_cut_frame_size_is_unchanged(ribbon_mask, cuts, boldrefs):
    """Only the single-slice layout is retrimmed; rows of cuts still need the
    taller frame."""
    png = render_frame(
        nib.load(boldrefs[0].path), ribbon_mask, cuts, boldrefs[0].label
    )
    assert Image.open(io.BytesIO(png)).size == FRAME_SIZE
