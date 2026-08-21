# Aligning the cortical ribbon with the fMRIPrep T1w space

**Date:** 2026-08-21
**Status:** Approved design, ready for implementation planning
**Supersedes:** the `ribbon.py` section of
[the v1 design](2026-08-16-fmriprepviz-session-registration-movie-design.md),
which asserted that no alignment step was needed

## Purpose

Correct a silent misalignment. `fmriprepviz` builds contours from FreeSurfer's
`ribbon.mgz` and uses that volume's own affine unchanged. The v1 design
justified this explicitly:

> `ribbon.mgz` carries a scanner-RAS affine matching fMRIPrep's
> `desc-preproc_T1w`, so no manual `c_ras` correction is required.

That is false whenever `recon-all` ran outside fMRIPrep — a common setup, and
what `--fs-subjects-dir` exists for. FreeSurfer's conformed space is then a
different physical space from fMRIPrep's T1w reference, related by a rigid
registration that fMRIPrep computes and writes out but that this tool ignores.

Measured on `/data/derivatives/fmriprep-25.2.5` (sub-07, an externally-run NSD
FreeSurfer), `from-fsnative_to-T1w` is a ~6° rotation with an 18.6 mm
translation. Uncorrected, the cortical ribbon lands ~19 mm off in _y_.

This is the worst available failure mode. A tool whose entire purpose is
detecting a few millimetres of misregistration instead produces a confident,
plausible-looking movie that is grossly wrong, with nothing on screen
announcing the error. Nothing crashes and no message is printed.

The v1 assumption was never verified because Task 9 of the implementation plan
— "smoke run against real data" — has been open since the design was written,
for want of a dataset with both `space-T1w` boldrefs and a FreeSurfer
directory. Such a dataset now exists, and it is what exposed this.

## Measured evidence

fMRIPrep already writes `desc-ribbon_mask.nii.gz` into the anat directory. It
is the cortical **gray matter** ribbon, already in T1w space:

| compared against `desc-ribbon_mask` |  dice |
| ----------------------------------- | ----: |
| `ribbon.mgz` labels `{3,42}` (GM)   | 0.887 |
| `ribbon.mgz` labels `{2,41}` (WM)   | 0.054 |
| `ribbon.mgz` labels `{2,3,41,42}`   | 0.633 |

It occupies 886,235 voxels against 883,765 for the FreeSurfer GM labels, and
sits on the _exact_ `desc-preproc_T1w` grid — identical shape and identical
affine. It is written whenever `run_reconall = true`: the reference run had
both `cifti_output = false` and `project_goodvoxels = false` and still produced
it, so it is not gated behind CIFTI or goodvoxels output.

Dice of 0.887 rather than ~0.99 reflects what a 2–3-voxel-thick sheet scores
after a nearest-neighbour resample through a 6° rotation, plus the fact that
fMRIPrep derives its mask from the surfaces rather than from `ribbon.mgz`. It
is agreement, not disagreement.

Two consequences drive the whole design:

1. A binary GM ribbon contoured at 0.5 yields **both** boundaries at once — the
   inner level set is the white surface, the outer is the pial. One mask
   replaces the white/pial pair and one `add_contours` call replaces two.
2. Being already in T1w space on the T1w grid, the common path needs **no
   transform at all**.

### Transform direction

Settled by measurement rather than by reading conventions, because both
directions run without error and only one is correct:

| route                                  | ribbon centroid (mm) | vs ground truth |
| -------------------------------------- | -------------------- | --------------- |
| `desc-ribbon_mask` (ground truth)      | 2.48, 14.57, 9.24    | —               |
| `nt_apply(xfm, mask, reference=T1w)`   | 2.83, 14.37, 9.59    | sub-voxel       |
| `(~xfm).matrix @ ribbon.affine`        | 2.50, 14.60, 9.49    | 0.25 mm         |
| **either route with the sign flipped** | -1.01, -23.30, 2.84  | **38 mm off**   |

`resampling.apply` consumes the transform **as loaded** — it pulls
target→source, the direction ITK already stores — while affine composition
needs the inverse. Both land on the anatomy. A flipped sign is 38 mm of silent
error, so it gets a dedicated test rather than a comment.

### Why resampling rather than affine composition

Affine composition leaves the data untouched and is exact, which is
superficially attractive. Resampling wins anyway, because it puts the fallback
mask on a grid byte-identical to the fast path's, so `cuts.py` and `render.py`
cannot behave differently depending on which path produced the mask. Affine
composition would hand them a 320³ frame, LIA-permuted and rotated ~6°
relative to T1w, in one case and an axis-aligned 208×300×320 grid in the other.

The cost of resampling is negligible and the accounting is favourable:

- nearest versus linear moves the 0.5 level set by mean 0.311 mm, p95
  0.800 mm — against a 1.8 mm BOLD voxel
- the per-frame resample target is 1.6× smaller (20.0M versus 32.8M voxels)
- "no interpolation" was illusory in any case: nilearn's `reorder_img` would
  interpolate an oblique volume at display time, merely outside our control

## Scope

**In scope**

- Contours that land correctly on `space-T1w` BOLD references regardless of
  where `recon-all` ran
- `desc-ribbon_mask` as the primary source; `ribbon.mgz` plus the
  fsnative→T1w transform as a fallback
- Anat-file discovery that tolerates session-level `anat/` directories

**Out of scope**

- Mesh contours from `hemi-*_{white,pial}.surf.gii`. These are already
  T1w-aligned as stored — verified while investigating — and would be the most
  accurate option available. `add_contours` takes volumes rather than meshes,
  and hand-rolled slice geometry is the documented source of silently
  left-right-flipped QC figures, so this stays deferred.
- Displaying FreeSurfer volumes such as `T1.mgz` as an underlay. This change is
  alignment correctness only.
- Template spaces, synthesizing a T1w boldref from `from-boldref_to-T1w`, and
  multi-subject batch mode all remain out of scope as in v1.

## Key decisions

| Decision              | Choice                                     | Rationale                                                                                                                                                                                           |
| --------------------- | ------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Primary ribbon source | `desc-ribbon_mask`                         | Already in T1w space on the T1w grid. Needs no transform, no FreeSurfer directory, and no new failure modes.                                                                                        |
| Contour count         | One mask, one contour at 0.5               | A GM ribbon's level set already carries both the white and pial boundaries, so the white/pial pair was redundant once the source is GM-only.                                                        |
| Fallback              | `ribbon.mgz` `{3,42}` + `resampling.apply` | Preserves existing capability for derivatives predating `desc-ribbon_mask`, and keeps a path a reviewer can run against an already-aligned FreeSurfer dataset to compare against pre-change output. |
| Fallback grid         | Resample onto `desc-preproc_T1w`           | Makes both paths return the same grid, so downstream code cannot tell them apart.                                                                                                                   |
| Interpolation         | `order=1`, mask left float                 | A float level set contours sub-voxel accurately; nearest-neighbour would compound staircasing.                                                                                                      |
| `resolve_fs_dir`      | Unchanged                                  | This tool targets fMRIPrep-generated derivatives, and `--fs-dir` is the existing escape hatch for anything else. Teaching the resolver one dataset's layout would be fitting the tool to a fixture. |
| Multiple anat matches | Hard error                                 | Per-session anatomicals mean several T1w spaces, which breaks the one-cut-set-per-movie invariant. Guessing is worse than stopping.                                                                 |
| Override flag         | `--ribbon` only                            | Covers explicit selection, already-aligned volumes, and multi-session disambiguation with one flag and no new semantics. No `--assume-aligned`.                                                     |

## Architecture

One concept replaces two: **the cortical ribbon mask, in T1w world space** — a
single image however it was obtained. `SurfaceMasks(white, pial)` goes away.

```
derivatives ──> discovery ──> [BoldRef(...)] ordered            (unchanged)
                                    │
desc-ribbon_mask ──┐                │
                   ├──> ribbon ──> mask (T1w world)  computed ONCE
ribbon.mgz + xfm ──┘                │
                                  cuts ──> cut_coords (mm)      computed ONCE
                                    │
              render(frame) ──> PNG ─┬──> movie ──> qc.gif / qc.mp4
                                     └──> viewer ──> qc.html
```

The three-fixed-quantities invariant from v1 is untouched: mask, cut
coordinates, and display bounds. Because the ribbon mask is now always on the
T1w grid, `render._prepare` keeping it as the resample reference means the
pinned display grid _becomes_ the T1w grid at no cost.

No new modules. Anat lookup joins `discovery.py`, already the module
responsible for locating files in a derivatives tree.

## Modules

### `discovery.py`

Add `find_unique_anat(derivatives_root, subject, pattern) -> Path`.

- Globs `sub-<label>/**/anat/<pattern>`. Session-level `anat/` directories
  (`sub-07/ses-nsdanat/anat/`) are normal fMRIPrep output, so globbing is
  required rather than a workaround for one dataset.
- Raises on zero matches, naming the pattern and the directory searched.
- Raises on multiple matches, listing them.
- Reuses the existing `parse_entities` when describing candidates.

`find_boldrefs`, `BoldRef`, and frame ordering are untouched.

### `ribbon.py`

Rewritten around a single loader:

```python
load_ribbon_mask(derivatives, subject, ribbon=None, fs_dir=None, xfm=None)
    -> nib.Nifti1Image   # T1w world space
```

Resolution order:

1. explicit `ribbon` path → `nib.load`
2. **explicit `fs_dir` or `xfm`** → the FreeSurfer path below
3. `find_unique_anat("*_desc-ribbon_mask.nii.gz")` → `nib.load`
4. the FreeSurfer path, by discovery:
   - `find_unique_anat("*_desc-preproc_T1w.nii.gz")` for the reference grid
   - `find_unique_anat("*_from-fsnative_to-T1w_mode-image_xfm.txt")`
   - `resolve_fs_dir(...)/mri/ribbon.mgz`, reduced to `isin(data, RIBBON_LABELS)`
   - `nitransforms.resampling.apply(xfm, gm, reference=t1w, order=1)`

Step 2 matters: an explicitly passed `--fs-dir` or `--fs-to-t1w-xfm` must take
precedence over auto-discovery, or those flags would be silently ignored on any
tree that happens to have a `desc-ribbon_mask`. Silently ignoring a flag the
user went out of their way to pass is its own bug, and it would also make the
FreeSurfer path untestable against real data, since the smoke run reaches it
precisely by passing those flags. Anything not given explicitly is still
discovered.

Retained: `resolve_fs_dir` exactly as it stands, and
`_require_cortical_labels`, rechecked against the gray-matter labels alone,
which still catches a failed `recon-all` or an `aseg.mgz` passed as a ribbon.

Label constants collapse to a single `RIBBON_LABELS = (3, 42)`.

Removed: `WHITE_LABELS`, `GRAY_LABELS`, `CORTICAL_LABELS`, `SurfaceMasks`,
`_mask_image`.

### `render.py`

- One `add_contours` at level 0.5, replacing the loop over white and pial.
- `_window` thresholds the mask at `>= 0.5` rather than `> 0`, since the mask
  may now be float.
- `_prepare` keeps the ribbon mask as its resample reference.
- `NO_OVERLAP_MESSAGE` gains a wrong or missing fsnative→T1w transform as a
  likely cause.

### `cuts.py`

Unchanged. It already takes a mask image and derives world coordinates from
that image's affine.

### `cli.py`

Adds `--ribbon PATH` and `--fs-to-t1w-xfm PATH`, threaded into
`load_ribbon_mask`. The `masks.white` / `masks.pial` call sites collapse to the
single mask.

## Error handling

Every new failure names what was searched and what to do, and exits 1 through
the existing `RuntimeError` chain in `main`:

- No ribbon mask and no usable fallback: list the anat glob and the FreeSurfer
  candidates.
- Multiple matches for an anat pattern: list them and point at `--ribbon`.
- Fallback missing its transform or `desc-preproc_T1w`: name the path and
  `--fs-to-t1w-xfm`.

## Testing

Test-driven, per repo convention: a failing test is committed before the change
that makes it pass, and no existing test is weakened to accommodate this work.
The suite stays hermetic and synthetic — no committed binary fixtures.

`tests/conftest.py` gains `desc-ribbon_mask` (the existing 30–40 mm GM shell),
`desc-preproc_T1w`, and a known non-identity ITK transform, with `ribbon.mgz`
written in a deliberately offset fsnative frame that the transform exactly
undoes. The offset is ~20 mm, large enough that a sign error cannot hide inside
a tolerance.

| Test                             | What it pins down                                                                                                                                                                                                             |
| -------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Sign trap                        | The fallback mask's centroid matches the fast path's to sub-voxel **and** does not match the uncorrected ribbon's. This is the test that would have caught the original bug.                                                  |
| Path equivalence                 | Fast and fallback masks agree to high dice on an identical grid — same shape, same affine.                                                                                                                                    |
| Both boundaries from one contour | Extends `test_contours_land_on_the_anatomy_not_merely_somewhere`: on the central row of a `z=0` cut, red appears at **both** r≈30 and r≈40. Automates the one claim in this design argued topologically rather than rendered. |
| Anat discovery                   | Session-level `anat/` found; multiple matches error; `--ribbon` overrides both paths.                                                                                                                                         |
| Flag precedence                  | An explicit `--fs-dir` or `--fs-to-t1w-xfm` selects the FreeSurfer path even when a `desc-ribbon_mask` is present. Without this the flags would be silently ignored, and the smoke run would not reach the code it claims to. |
| Regression                       | `resolve_fs_dir` behaviour and `--fs-dir` resolution unchanged.                                                                                                                                                               |

Existing invariants stay as they are: masks and cuts computed once regardless of
frame count, display bounds pinned across differing boldref fields of view,
fixed and deterministic frame size, total frame ordering, HTML escaping.

## Verification

- `uv run pytest` green.
- A run against the stated `nitransforms` minimum, since stated minimums must
  be tested.
- A standalone smoke script, **kept out of version control** so the committed
  suite stays synthetic, exercising both paths against real data:
  - fast path — `fmriprepviz /data/derivatives/fmriprep-25.2.5 --subject 07`,
    which uses `desc-ribbon_mask` and needs no FreeSurfer directory
  - fallback — the same plus `--fs-dir /data/sourcedata/freesurfer/sub-07` and
    `--fs-to-t1w-xfm .../from-fsnative_to-T1w_mode-image_xfm.txt`
  - assert the two agree, then inspect a frame by eye for contour placement and
    left-right orientation, the one property no assertion covers

This closes Task 9 of the v1 implementation plan.

### Comparing against pre-change output

The fallback exists partly so a reviewer can point this at an **already
aligned** FreeSurfer dataset — one where the transform is effectively identity
— and confirm that an existing invocation still produces the expected picture.

That comparison is visual equivalence, not pixel identity, and the difference
is expected rather than a regression. Before this change the two contours were
the boundary of `{2,41}` and of `{2,3,41,42}`; after it they are the inner and
outer boundaries of `{3,42}`. Those coincide wherever cortical gray matter
separates white matter from CSF, which is everywhere the contours carry
registration information. They diverge where white matter abuts something that
is not cortical gray — the medial wall, and the borders of subcortical
structures — because the old white contour continued along the white-matter
boundary there, while the new inner contour has no gray matter to follow and
the level set closes around the sheet instead.

A reviewer expecting identical output would read that as a bug, so it is
recorded here in advance. What must hold is that no contour moves relative to
the anatomy.

## Dependencies

Adds `nitransforms>=25.1.0` to runtime dependencies. The floor is the version
that provides `nitransforms.resampling.apply`; `Affine.apply` is deprecated as
of that release.
