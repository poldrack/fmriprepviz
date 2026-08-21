# fmriprepviz: cross-session BOLD registration movie

**Date:** 2026-08-16
**Status:** Approved design, ready for implementation planning

## Purpose

Assess the quality and stability of BOLD-to-anatomical registration across
sessions in an fMRIPrep output directory. The tool builds a flipbook movie in
which each frame shows one BOLD reference image at a fixed set of anatomical
locations, with the subject's FreeSurfer white and pial surfaces drawn over it
in red.

Because the cut locations and the surface contours are identical in every
frame, anything that appears to move as the movie plays is misregistration.
A well-registered subject produces a movie in which the red contours stay
locked to the same tissue boundaries throughout.

## Scope

**In scope**

- `space-T1w` BOLD reference images from an fMRIPrep derivatives tree
- Red white-surface and pial-surface contours derived from FreeSurfer's
  `ribbon.mgz`
- One frame per boldref file, ordered by session then run
- Animated GIF output; MP4 when `ffmpeg` is available
- A self-contained HTML viewer for frame-by-frame stepping

**Out of scope (v1)**

- Template spaces such as MNI152NLin2009cAsym. FreeSurfer surfaces exist only
  in the subject's native anatomical space; producing them in a template space
  would require warping the surfaces ourselves, which is deferred.
- Applying the `from-boldref_to-T1w` transform ourselves to synthesize a
  T1w-space boldref when fMRIPrep did not write one.
- Contours computed from surface meshes (`lh.white`, `rh.pial`) rather than
  from the ribbon volume.
- Multi-subject batch mode. One subject per invocation.

## Key decisions

| Decision          | Choice                                          | Rationale                                                                                                                                                                                                                    |
| ----------------- | ----------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Movie axis        | Frame = boldref, fixed cuts                     | Misregistration appears as jitter against a static background; this is the standard registration flipbook.                                                                                                                   |
| Space             | `space-T1w` only                                | FreeSurfer surfaces are generated in native anatomical space. Any other space needs a warp we are not doing in v1.                                                                                                           |
| Contour source    | `ribbon.mgz`                                    | Matches what fMRIPrep's own reports show, needs only nibabel/nilearn, and avoids mesh-plane intersection geometry.                                                                                                           |
| Frame granularity | One frame per boldref                           | Averaging runs within a session would hide real within-session misregistration. Filters narrow the set when the movie is too long.                                                                                           |
| Rendering         | nilearn `plot_anat` + `add_contours`            | nilearn handles cut placement, orientation, and aspect ratio correctly. Hand-rolled slice display is the usual source of silently left-right-flipped QC figures.                                                             |
| Resampling        | BOLD always resampled onto the 1 mm ribbon grid | Contours stay crisp, the background is smooth, and — the load-bearing reason — the display bounds stay pinned across frames. `--no-upsample` switches the interpolation to nearest-neighbour; it does not skip the resample. |
| File discovery    | glob + regex, not pybids                        | One filename pattern; pybids indexing of a derivatives tree is slow and buys nothing here.                                                                                                                                   |

## Preconditions

1. fMRIPrep was run with `T1w` among its `--output-spaces`, so that
   `*_space-T1w_boldref.nii.gz` files exist. If the glob finds no such files,
   the tool lists the `space-*` values it did find and states that `T1w` output
   is required, rather than failing obscurely.
2. A FreeSurfer subject directory containing `mri/ribbon.mgz` is reachable.
   Resolution order: `--fs-dir` if given, else
   `<derivatives>/sourcedata/freesurfer/sub-<label>`, else
   `$SUBJECTS_DIR/sub-<label>`. If none resolve, the tool reports every path it
   tried.

Note that fMRIPrep runs using precomputed anatomical derivatives
(`--derivatives`) may have no `sourcedata/freesurfer` directory at all, which
is why `--fs-dir` exists.

## Architecture

```
derivatives root ──> discovery ──> [BoldRef(path, entities, label), ...] ordered
                                          │
freesurfer/mri/ribbon.mgz ──> ribbon ──> white_mask, pial_mask (1 mm, scanner RAS)
                                          │
                                       cuts ──> cut_coords (mm), computed ONCE
                                          │
                    ┌─────────────────────┴──────────────────┐
                    ▼                                        ▼
              render(frame) ──> PNG bytes ──┬──> movie ──> qc.gif / qc.mp4
                                            └──> viewer ──> qc.html
```

The masks and cut coordinates are computed once, before the frame loop, and
reused unchanged for every frame. This is the invariant that makes the movie a
valid registration test.

**Three things must be held fixed, not two.** The masks and the cut coordinates
are the obvious pair, but the _display bounds_ are a third, and they are easy to
overlook because nothing in the code names them. nilearn derives each axis's
limits from the bounds of the image drawn on it, so two BOLD references with
different fields of view render at different zoom — and since the contours are
drawn in world coordinates onto those axes, the contours themselves move on
screen. A viewer reads that as misregistration, or real misregistration is
masked by the rescaling.

Resampling every BOLD reference onto the ribbon's grid is what pins the display
bounds. That step therefore serves two purposes, and only one of them is
cosmetic: it smooths the background, and it holds the axes fixed. It must
happen on every path, including when interpolation is not wanted — which is why
`--no-upsample` selects nearest-neighbour interpolation rather than skipping the
resample.

This was found by the final whole-branch review rather than during design, and
the omission was in this document as much as in the code.

## Modules

All live in `src/fmriprepviz/`. `__init__.py` stays empty.

### `discovery.py`

Locate and order the BOLD reference images.

- `find_boldrefs(derivatives_root, subject, ses=None, task=None, run=None) -> list[BoldRef]`
- `BoldRef` is a dataclass holding `path`, an `entities` dict, and a
  human-readable `label` (e.g. `"ses-02 task-stopSignal run-1"`).
- Globs `sub-<label>/[ses-*/]func/*_space-T1w_boldref.nii.gz`.
- Parses BIDS key-value entities from the filename with a regex.
- Sorts by `(ses, task, acq, run)` using numeric-aware comparison, so `run-2`
  precedes `run-10`.
- Optional filters narrow the result set.
- Raises a descriptive error when nothing matches, listing the `space-*` values
  present in the tree.

### `ribbon.py`

Turn FreeSurfer's ribbon into two boundary masks.

- `resolve_fs_dir(derivatives_root, subject, fs_dir=None) -> Path`
- `load_surface_masks(fs_dir) -> SurfaceMasks` with `.white` and `.pial`
  nibabel images.

FreeSurfer `ribbon.mgz` label values: 2 = left white matter, 3 = left cortical
gray, 41 = right white matter, 42 = right cortical gray. Therefore:

- white surface = boundary of `{2, 41}`
- pial surface = boundary of `{2, 3, 41, 42}`

`ribbon.mgz` carries a scanner-RAS affine matching fMRIPrep's
`desc-preproc_T1w`, so no manual `c_ras` correction is required; nilearn
resamples the masks onto each display plane itself.

### `cuts.py`

Choose the anatomical locations shown in every frame.

- `select_cuts(mask_img, n_cuts=7, display_modes=("x", "y", "z")) -> dict[str, list[float]]`
- Computes the bounding box of the pial mask and places `n_cuts` evenly spaced
  cuts across the central 80% of each extent, avoiding empty superior and
  inferior slices.
- Purely a function of the ribbon, hence identical for every frame.

### `render.py`

Render one frame.

- `render_frame(boldref_img, masks, cuts, label, upsample=True) -> bytes` (PNG)
- Three stacked axes (sagittal, coronal, axial rows). Each is drawn with
  `plot_anat(..., axes=..., display_mode=..., cut_coords=...)` followed by two
  `add_contours` calls in red for the white and pial masks.
- Fixed `figsize` and `dpi` so every frame has identical pixel dimensions,
  which GIF and MP4 assembly require. This is asserted by a test rather than
  assumed.
- Intensity windowing is per frame, from the 2nd and 98th percentiles computed
  **inside the pial mask**. Sessions have arbitrary scanner intensity units, so
  a single global window would make the movie flicker; masked robust scaling
  makes brightness comparable across frames without concealing anything.
- The frame label and index are drawn into the figure itself, so they survive
  into the GIF and MP4 where no surrounding UI exists.
- When `upsample` is true the boldref is resampled onto the ribbon's 1 mm grid
  with linear interpolation before display.

### `movie.py`

Assemble frames into a movie.

- `write_gif(frames, path, fps)` using Pillow, always available.
- `write_mp4(frames, path, fps)` by piping the PNG frames to `ffmpeg` through a
  subprocess. No extra dependency; raises a clear error when `ffmpeg` is not on
  `PATH`.
- Dispatch is by output suffix.

### `viewer.py`

- `write_html(frames, labels, path)` produces one self-contained file with the
  PNG frames base64-embedded, a slider, prev/next buttons, play/pause, and
  left/right arrow key bindings. No external assets, no build step.

### `cli.py`

```
fmriprepviz DERIVATIVES_DIR --subject 01 [-o qc.gif]
    [--ses 01 02] [--task rest] [--run 1]
    [--fps 2] [--n-cuts 7] [--display-mode zyx] [--cut-coords ...]
    [--fs-dir PATH] [--no-html] [--no-upsample]
```

- `main()` is the console entry point.
- Default output is `qc.gif` in the current directory; the HTML viewer is
  written alongside it with a matching stem unless `--no-html` is given.
- `-o` ending in `.mp4` selects MP4 instead of GIF.

## Outputs

| File          | Condition                                   |
| ------------- | ------------------------------------------- |
| `<stem>.gif`  | default                                     |
| `<stem>.mp4`  | `-o` ends in `.mp4` and `ffmpeg` is present |
| `<stem>.html` | unless `--no-html`                          |

## Error handling

Every failure mode reports what was searched and what to do about it:

- No boldrefs matched: list the `space-*` values found, state that
  `--output-spaces T1w` is required.
- FreeSurfer directory unresolvable: list all paths tried.
- `ribbon.mgz` missing from an otherwise valid FreeSurfer directory: name the
  expected file.
- `.mp4` requested without `ffmpeg`: say so and suggest `.gif`.
- Exactly one frame found: warn that a one-frame movie cannot show
  cross-session drift, but still produce it.

## Testing

Test-driven throughout; tests are committed before the implementation they
cover. pytest, functions rather than classes, fixtures for persistent objects.

The whole unit suite runs against a synthetic derivatives tree built by a
fixture in `conftest.py`:

- two sessions x two runs of small 3 mm `space-T1w_boldref.nii.gz` images
- a `ribbon.mgz` built from concentric spheres of known radius, using real
  FreeSurfer label values

Because that geometry is known analytically, tests assert that the extracted
white and pial boundaries lie at the correct radius. This verifies contour
_placement_, not merely that an image was produced, which is the property that
actually matters for a QC tool.

| Module      | What is verified                                                                                                                                  |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `discovery` | all boldrefs found; correct ordering including `run-2` before `run-10`; each filter narrows correctly; empty-match error names the spaces present |
| `ribbon`    | label-to-mask mapping; boundary voxels at the known sphere radii; fs-dir resolution order                                                         |
| `cuts`      | cuts inside the mask bounding box; deterministic across calls; `n_cuts` respected                                                                 |
| `render`    | returns valid PNG bytes; every frame has identical dimensions; deterministic for identical input                                                  |
| `movie`     | GIF reopens with the expected frame count; MP4 test skipped when `ffmpeg` is absent                                                               |
| `viewer`    | HTML contains the expected number of embedded images and every frame label                                                                        |
| `cli`       | end-to-end on the fixture tree writes both movie and HTML                                                                                         |

A manual smoke run against real fMRIPrep derivatives provides the visual
confirmation that the overlay lands correctly. As of this writing no dataset on
the development machine has both `space-T1w` boldrefs and a FreeSurfer
directory, so that run is pending.

## Dependencies

Runtime: `nibabel`, `nilearn`, `matplotlib`, `numpy`, `pillow`.
Development: `pytest`.

Managed with `uv`; all local commands run through `uv run`. Packaged with
hatchling in a `src/` layout, exposing the console script
`fmriprepviz = fmriprepviz.cli:main`.

## Possible later extensions

Recorded so they are not mistaken for oversights:

- Mesh-based contours behind the same interface `ribbon.py` presents.
- Synthesizing a T1w-space boldref from `from-boldref_to-T1w` transforms when
  fMRIPrep wrote no `space-T1w` output.
- Template-space support by warping surfaces through the anatomical-to-template
  transform chain.
- A static filmstrip PNG for pasting into reports.
