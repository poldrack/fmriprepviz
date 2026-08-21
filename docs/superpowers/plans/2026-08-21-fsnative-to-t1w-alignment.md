# fsnative-to-T1w Ribbon Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the red surface contours land on the anatomy for fMRIPrep derivatives whose FreeSurfer volumes are not aligned with the preprocessed T1w image.

**Architecture:** Replace the white/pial mask pair with a single cortical gray-matter ribbon mask that always lives in the fMRIPrep T1w space. Prefer fMRIPrep's own `desc-ribbon_mask` (already in that space); fall back to `ribbon.mgz` resampled through the `from-fsnative_to-T1w` transform. Because gray matter is a sheet, contouring one mask at 0.5 yields both the white (inner) and pial (outer) surfaces, so one `add_contours` call replaces two.

**Tech Stack:** Python 3.11+, nibabel, nilearn, nitransforms, numpy, matplotlib, Pillow, pytest, uv.

**Spec:** `docs/superpowers/specs/2026-08-21-fsnative-to-t1w-alignment-design.md`

**Branch:** `fix/fsnative-to-t1w-alignment` (already exists, spec committed)

## Global Constraints

- `nitransforms>=25.1.0` — the floor providing `nitransforms.resampling.apply`; `Affine.apply` is deprecated. Already present in `pyproject.toml`; commit it in Task 4.
- `src/fmriprepviz/__init__.py` stays empty. Non-negotiable.
- Tests are functions, never classes. Shared constants are imported `from conftest import ...`.
- No committed binary fixtures. Everything synthetic, built in `tests/conftest.py`.
- **Never weaken an existing test to make it pass.** Where behaviour genuinely changes, rewrite the test and its docstring to state the new behaviour and why.
- Run `uv run pytest` after every implementation step. Run all commands through `uv run`.
- Conventional commits. This repo spells the docs type `docs:` (not `doc:`). Types in use: `feat`, `fix`, `rf`, `test`, `docs`, `style`, `build`, `chore`.
- TDD: the failing test is committed before the change that makes it pass, in a separate commit.
- `npx --yes prettier -w <file>` before committing any Markdown, YAML, or JSON.
- Never let a task end with the suite red. New functions are added alongside old ones and the switchover happens in Task 5.

## File Structure

| File                           | Responsibility                                                      | Task |
| ------------------------------ | ------------------------------------------------------------------- | ---- |
| `tests/conftest.py`            | Synthetic trees: aligned, misaligned, session-level anat            | 1    |
| `tests/test_fixtures.py`       | Verifies the fixtures' own geometry                                 | 1    |
| `src/fmriprepviz/discovery.py` | **+** `find_unique_anat`, `AnatNotFoundError`, `AmbiguousAnatError` | 2    |
| `tests/test_discovery.py`      | **+** anat discovery tests                                          | 2    |
| `src/fmriprepviz/ribbon.py`    | `load_ribbon_mask`; owns `MASK_THRESHOLD`                           | 3, 4 |
| `tests/test_ribbon.py`         | Fast path, fallback, sign trap, alignment guard                     | 3, 4 |
| `src/fmriprepviz/render.py`    | One contour from one mask                                           | 5    |
| `tests/test_render.py`         | Single-mask signature; both boundaries from one contour             | 5    |
| `src/fmriprepviz/cuts.py`      | Threshold at `MASK_THRESHOLD` instead of `> 0`                      | 5    |
| `src/fmriprepviz/cli.py`       | `--ribbon`, `--fs-to-t1w-xfm`; retire `SurfaceMasks`                | 5    |
| `tests/test_cli.py`            | Flag precedence; retarget mismatch tests                            | 5    |
| `README.md`                    | Document the new flags and the ribbon source order                  | 6    |

**One deliberate divergence from the spec.** The spec says `cuts.py` is
unchanged. It gets a one-line change here — `> 0` becomes
`>= MASK_THRESHOLD` — because the fallback mask is float after resampling, so
`> 0` would include partial-volume voxels and inflate the mask extent the cuts
are derived from. The effect is sub-voxel, but having two different notions of
where the mask boundary lies is the kind of inconsistency that later reads as
a bug.

---

### Task 1: Extend the synthetic fixtures

The whole suite rests on these fixtures, and `test_fixtures.py` exists precisely
because their geometry is verified before anything depends on it. The
misaligned tree is the new centrepiece: the FreeSurfer anatomy sits 20 mm from
the T1w anatomy and the transform is the only thing reconciling them.

**Files:**

- Modify: `tests/conftest.py`
- Test: `tests/test_fixtures.py`

**Interfaces:**

- Consumes: nothing.
- Produces: `ANAT_OFFSET_MM = 20.0`; fixtures `derivatives` (now with an anat
  directory), `derivatives_misaligned`, `derivatives_session_anat`; helpers
  `_write_anat(root, subject="01", session=None, offset_mm=0.0)` and
  `_write_ribbon(root, offset_mm=0.0)`.

- [ ] **Step 1: Write the failing fixture-geometry tests**

Append to `tests/test_fixtures.py`, extending its imports with `ANAT_OFFSET_MM`
and `PIAL_RADIUS_MM`:

```python
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
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_fixtures.py -v`
Expected: FAIL — `ImportError: cannot import name 'ANAT_OFFSET_MM' from 'conftest'`.

- [ ] **Step 3: Add the offset-aware geometry helpers to `tests/conftest.py`**

Add `ANAT_OFFSET_MM` beside the existing radius constants:

```python
# How far the FreeSurfer anatomy sits from the T1w anatomy in the misaligned
# fixture, along +y. Deliberately large: a dropped or inverted transform must
# fail loudly rather than hide inside a tolerance.
ANAT_OFFSET_MM = 20.0
```

Give `_world_radius` and `_ribbon_data` an optional centre. The defaults keep
every existing caller unchanged:

```python
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
```

- [ ] **Step 4: Add the anat writer to `tests/conftest.py`**

```python
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
```

- [ ] **Step 5: Add the fixtures to `tests/conftest.py`**

Add `_write_anat(root)` to the existing `derivatives` fixture, then add the two
new trees. Leave `derivatives_without_t1w` alone.

```python
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
def derivatives_session_anat(tmp_path_factory) -> Path:
    """fMRIPrep writes anat outputs under ses-<label>/anat when sessions carry
    their own anatomicals, so discovery must glob rather than assume sub/anat.
    """
    root = tmp_path_factory.mktemp("session_anat")
    _write_boldrefs(root)
    _write_ribbon(root)
    _write_anat(root, session="nsdanat")
    return root
```

Leave the existing `surface_masks` fixture in place for now; Task 5 removes it.

- [ ] **Step 6: Run the whole suite to confirm the new tests pass and nothing regressed**

Run: `uv run pytest`
Expected: PASS. Adding an anat directory must not disturb any existing test;
the boldref globs are scoped to `**/func/`.

- [ ] **Step 7: Commit**

```bash
git add tests/conftest.py tests/test_fixtures.py
git commit -m "test: add T1w-space anat fixtures and a misaligned tree

The misaligned tree puts the FreeSurfer anatomy 20 mm from the T1w
anatomy, so a dropped or inverted transform cannot pass unnoticed."
```

---

### Task 2: Locate anat derivatives

**Files:**

- Modify: `src/fmriprepviz/discovery.py`
- Test: `tests/test_discovery.py`

**Interfaces:**

- Consumes: `derivatives`, `derivatives_session_anat` fixtures from Task 1.
- Produces: `find_unique_anat(derivatives_root, subject: str, pattern: str) -> Path`;
  exceptions `AnatNotFoundError(RuntimeError)` and `AmbiguousAnatError(RuntimeError)`.
  Task 3 relies on these two being **distinct** types: "absent" falls through to
  the FreeSurfer path, "ambiguous" must not.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_discovery.py`, adding to its imports:

```python
from fmriprepviz.discovery import (
    AmbiguousAnatError,
    AnatNotFoundError,
    find_unique_anat,
)

RIBBON_MASK_PATTERN = "*_desc-ribbon_mask.nii.gz"


def test_finds_anat_directly_under_the_subject(derivatives):
    found = find_unique_anat(derivatives, "01", RIBBON_MASK_PATTERN)
    assert found.name == "sub-01_desc-ribbon_mask.nii.gz"
    assert found.parent == derivatives / "sub-01" / "anat"


def test_finds_anat_under_a_session_directory(derivatives_session_anat):
    """fMRIPrep puts anat outputs under ses-<label>/anat when sessions carry
    their own anatomicals, so this must not require sub-<label>/anat.
    """
    found = find_unique_anat(derivatives_session_anat, "01", RIBBON_MASK_PATTERN)
    assert found.parent.parent.name == "ses-nsdanat"


def test_missing_anat_names_the_pattern_and_the_directory(tmp_path):
    (tmp_path / "sub-01").mkdir()
    with pytest.raises(AnatNotFoundError) as excinfo:
        find_unique_anat(tmp_path, "01", RIBBON_MASK_PATTERN)
    message = str(excinfo.value)
    assert RIBBON_MASK_PATTERN in message
    assert "sub-01" in message


def test_multiple_anat_matches_are_ambiguous_not_a_guess(tmp_path):
    """Per-session anatomicals mean several T1w spaces. The tool draws one set
    of cuts and one mask for every frame, so it cannot pick a space on the
    user's behalf; it must stop and say so.
    """
    for session in ("a", "b"):
        anat = tmp_path / "sub-01" / f"ses-{session}" / "anat"
        anat.mkdir(parents=True)
        (anat / f"sub-01_ses-{session}_desc-ribbon_mask.nii.gz").touch()
    with pytest.raises(AmbiguousAnatError) as excinfo:
        find_unique_anat(tmp_path, "01", RIBBON_MASK_PATTERN)
    message = str(excinfo.value)
    assert "ses-a" in message and "ses-b" in message
    assert "--ribbon" in message


def test_ambiguous_and_missing_are_distinct_types():
    """load_ribbon_mask falls through to FreeSurfer when the mask is absent,
    but must not swallow an ambiguous match, so these cannot share a type.
    """
    assert not issubclass(AmbiguousAnatError, AnatNotFoundError)
    assert not issubclass(AnatNotFoundError, AmbiguousAnatError)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_discovery.py -k anat -v`
Expected: FAIL with `ImportError: cannot import name 'AmbiguousAnatError'`.

- [ ] **Step 3: Implement `find_unique_anat`**

Add to `src/fmriprepviz/discovery.py`, beside `NoBoldRefsError`:

```python
class AnatNotFoundError(RuntimeError):
    """No anatomical derivative matched the requested pattern."""


class AmbiguousAnatError(RuntimeError):
    """Several anatomical derivatives matched, so the T1w space is ambiguous."""


def find_unique_anat(derivatives_root, subject: str, pattern: str) -> Path:
    """Return the one anat derivative matching `pattern` for this subject.

    fMRIPrep writes anatomical outputs to `sub-<label>/anat/`, or to
    `sub-<label>/ses-<label>/anat/` when sessions carry their own
    anatomicals, so the glob covers both rather than assuming either.

    Several matches mean several T1w spaces. Since one mask and one set of
    cuts are reused for every frame, choosing between them silently would
    invalidate the comparison the movie exists to make, so it is an error.
    """
    subject_dir = Path(derivatives_root) / f"sub-{subject}"
    matches = sorted(subject_dir.glob(f"**/anat/{pattern}"))
    if not matches:
        raise AnatNotFoundError(
            f"No file matching {pattern} found under {subject_dir}/**/anat/."
        )
    if len(matches) > 1:
        listed = "\n  ".join(str(match) for match in matches)
        raise AmbiguousAnatError(
            f"Found {len(matches)} files matching {pattern} under "
            f"{subject_dir}, so which T1w space to use is ambiguous:\n"
            f"  {listed}\n"
            "Pass --ribbon to name the ribbon mask to use."
        )
    return matches[0]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_discovery.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_discovery.py
git commit -m "test: require unique anat discovery across session layouts"
git add src/fmriprepviz/discovery.py
git commit -m "feat(discovery): locate a unique anat derivative

fMRIPrep may write anat outputs under sub-<label>/anat or under
sub-<label>/ses-<label>/anat, so glob both. Report several matches as
ambiguous rather than picking one: multiple T1w spaces would break the
single-mask, single-cut-set invariant the flipbook rests on."
```

---

### Task 3: Load the ribbon mask — fast path

`load_ribbon_mask` is added **alongside** the existing `load_surface_masks`, so
the suite stays green. Task 5 removes the old function.

**Files:**

- Modify: `src/fmriprepviz/ribbon.py`
- Modify: `tests/conftest.py` (add the `ribbon_mask` fixture)
- Test: `tests/test_ribbon.py`

**Interfaces:**

- Consumes: `find_unique_anat`, `AnatNotFoundError`, `AmbiguousAnatError` (Task 2).
- Produces: `load_ribbon_mask(derivatives_root, subject, ribbon=None, fs_dir=None, xfm=None) -> nib.Nifti1Image`;
  `MASK_THRESHOLD = 0.5`; `RIBBON_LABELS = (3, 42)`;
  `RIBBON_MASK_PATTERN`, `T1W_PATTERN`, `FS_TO_T1W_PATTERN`.
  `MASK_THRESHOLD` is imported by `render.py` and `cuts.py` in Task 5.
  Fixture `ribbon_mask`, used by Task 5.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ribbon_mask.py` (a new file, so Task 5 can delete the old
`tests/test_ribbon.py` tests without untangling two concerns in one file):

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_ribbon_mask.py -v`
Expected: FAIL with `ImportError: cannot import name 'MASK_THRESHOLD'`.

- [ ] **Step 3: Implement the fast path in `src/fmriprepviz/ribbon.py`**

Add the new constants and function. Leave `load_surface_masks`, `SurfaceMasks`,
`WHITE_LABELS`, `GRAY_LABELS`, `CORTICAL_LABELS` and `_mask_image` untouched
for now.

```python
from fmriprepviz.discovery import AnatNotFoundError, find_unique_anat

# The mask's boundary -- the surface we actually draw -- is its 0.5 level set.
# Stated once here because render.py contours at it and cuts.py and render.py
# both threshold at it.
MASK_THRESHOLD = 0.5

RIBBON_LABELS = (3, 42)  # left/right cortical gray matter
RIBBON_MASK_PATTERN = "*_desc-ribbon_mask.nii.gz"
T1W_PATTERN = "*_desc-preproc_T1w.nii.gz"
FS_TO_T1W_PATTERN = "*_from-fsnative_to-T1w_mode-image_xfm.txt"


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
```

Add a placeholder that Task 4 fills in, so this task's tests can run:

```python
def _ribbon_from_freesurfer(derivatives_root, subject: str, fs_dir, xfm):
    raise FreeSurferNotFoundError(
        "Building the ribbon from FreeSurfer is not implemented yet."
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_ribbon_mask.py -v`
Expected: PASS.

- [ ] **Step 5: Add the `ribbon_mask` fixture to `tests/conftest.py`**

```python
@pytest.fixture(scope="session")
def ribbon_mask(derivatives):
    """The T1w-space cortical ribbon mask from the synthetic tree."""
    from fmriprepviz.ribbon import load_ribbon_mask

    return load_ribbon_mask(derivatives, "01")
```

- [ ] **Step 6: Run the whole suite**

Run: `uv run pytest`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add tests/test_ribbon_mask.py tests/conftest.py
git commit -m "test: require the T1w-space ribbon mask to be preferred"
git add src/fmriprepviz/ribbon.py
git commit -m "feat(ribbon): load fMRIPrep's T1w-space ribbon mask

desc-ribbon_mask is cortical gray matter already in the T1w space and on
the T1w grid, so it needs no transform. Being gray-matter only, its 0.5
level set carries the white surface inside and the pial outside."
```

---

### Task 4: Load the ribbon mask — FreeSurfer fallback

The sign trap lives here. It is the test that would have caught the original
bug, so it is worth more than the rest of this task combined.

**Files:**

- Modify: `src/fmriprepviz/ribbon.py`
- Modify: `pyproject.toml` (commit the already-present `nitransforms` pin)
- Test: `tests/test_ribbon_mask.py`

**Interfaces:**

- Consumes: `load_ribbon_mask` (Task 3), `find_unique_anat` (Task 2),
  `derivatives_misaligned` (Task 1).
- Produces: a working `_ribbon_from_freesurfer`; no new public names.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_ribbon_mask.py`, extending its imports with
`ANAT_OFFSET_MM` from conftest and `FreeSurferNotFoundError` from
`fmriprepviz.ribbon`:

```python
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
    assert abs(actual[1] + ANAT_OFFSET_MM) > ANAT_OFFSET_MM / 2, (
        "ribbon moved the wrong way: the transform was inverted"
    )


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
```

Add `RIBBON_LABELS` to the imports from `fmriprepviz.ribbon`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_ribbon_mask.py -v`
Expected: the five new tests FAIL with "Building the ribbon from FreeSurfer is
not implemented yet."

- [ ] **Step 3: Implement the fallback in `src/fmriprepviz/ribbon.py`**

Add the imports at the top of the module:

```python
import nitransforms as nt
from nitransforms.resampling import apply as nt_apply
```

Replace the placeholder `_ribbon_from_freesurfer` with:

```python
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

    gray = nib.Nifti1Image(
        np.isin(data, RIBBON_LABELS).astype(np.uint8), ribbon_img.affine
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
```

Note `order=1` leaves the mask float, so its boundary is a sub-voxel-accurate
level set rather than a staircase.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_ribbon_mask.py -v`
Expected: PASS.

- [ ] **Step 5: Run the whole suite**

Run: `uv run pytest`
Expected: PASS. `load_surface_masks` is still present and untouched, so the old
`tests/test_ribbon.py` still passes.

- [ ] **Step 6: Commit**

```bash
git add tests/test_ribbon_mask.py
git commit -m "test: pin the fsnative-to-T1w transform direction

The misaligned fixture distinguishes a correctly applied transform from a
dropped or inverted one. All three render without error, so the direction
is asserted numerically."
git add src/fmriprepviz/ribbon.py pyproject.toml uv.lock
git commit -m "fix(ribbon): move FreeSurfer's ribbon into the T1w space

ribbon.mgz was used with its own affine, on the assumption that
FreeSurfer's conformed space matches fMRIPrep's desc-preproc_T1w. That
holds only when recon-all ran inside fMRIPrep; otherwise the two are
different physical spaces and the contours land on the wrong tissue with
no error raised -- measured at ~19 mm on a real dataset.

Resample through from-fsnative_to-T1w onto the T1w grid, matching the
grid fMRIPrep's own desc-ribbon_mask uses. Guard against a ribbon that
lands nowhere on that grid, which is now how a wrong --fs-dir presents."
```

---

### Task 5: Switch to a single mask end-to-end

One task, not two: `render_frame`'s signature and its only caller must change
together. Splitting them would leave the suite red at the intermediate commit,
and no reviewer could accept one half without the other. It is correspondingly
the largest task here — expect the full cycle rather than a quick edit.

**Files:**

- Modify: `src/fmriprepviz/render.py`
- Modify: `src/fmriprepviz/cuts.py`
- Modify: `src/fmriprepviz/cli.py`
- Modify: `src/fmriprepviz/ribbon.py` (delete the retired API)
- Modify: `tests/conftest.py` (drop `surface_masks`)
- Modify: `tests/test_ribbon.py` (drop `load_surface_masks` coverage)
- Test: `tests/test_render.py`, `tests/test_cli.py`

**Interfaces:**

- Consumes: `MASK_THRESHOLD`, `load_ribbon_mask` (Tasks 3–4), `ribbon_mask`
  fixture (Task 3).
- Produces: `render_frame(boldref_img, ribbon_mask, cuts, label, upsample=True) -> bytes`
  — the second parameter is a single image, not a `SurfaceMasks`. CLI flags
  `--ribbon` and `--fs-to-t1w-xfm` (argparse dest `fs_to_t1w_xfm`).

- [ ] **Step 1: Write the failing test for both boundaries**

This is the automated form of the claim that one contour yields two surfaces —
the one assertion in this whole change that replaces an argument with evidence.
Add to `tests/test_render.py`:

```python
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
```

Extend the imports of `tests/test_render.py` with `WHITE_RADIUS_MM` from
conftest.

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_render.py::test_one_contour_draws_both_the_white_and_pial_boundaries -v`
Expected: FAIL — `render_frame` still expects a `SurfaceMasks` and will raise
`AttributeError: 'Nifti1Image' object has no attribute 'pial'`.

- [ ] **Step 3: Update `src/fmriprepviz/render.py`**

Import the threshold and use it for both the contour level and the window:

```python
from fmriprepviz.ribbon import MASK_THRESHOLD

CONTOUR_LEVELS = [MASK_THRESHOLD]
```

Change the message to name the new likely cause:

```python
NO_OVERLAP_MESSAGE = (
    "The cortical ribbon mask and the BOLD reference do not overlap on the "
    "display grid. This usually means the ribbon and the BOLD reference have "
    "mismatched affines -- a wrong or missing fsnative-to-T1w transform, or "
    "--fs-dir or --ribbon pointing at the wrong subject."
)
```

Take a single mask throughout:

```python
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
```

In `_window`, threshold rather than test for non-zero, since the mask is float
after resampling:

```python
    inside = data[np.asarray(mask.dataobj) >= MASK_THRESHOLD]
```

In `_draw_row`, one contour replaces the loop:

```python
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
```

Update the `_prepare` docstring's references to "the mask grid" to say the
ribbon mask, and note that the mask now always shares the T1w grid, so the
pinned display grid is the T1w grid.

- [ ] **Step 4: Update `src/fmriprepviz/cuts.py` to use the same threshold**

```python
from fmriprepviz.ribbon import MASK_THRESHOLD


def _world_coordinates(mask_img) -> np.ndarray:
    ijk = np.argwhere(np.asarray(mask_img.dataobj) >= MASK_THRESHOLD)
    homogeneous = np.column_stack([ijk, np.ones(len(ijk))])
    return (mask_img.affine @ homogeneous.T).T[:, :3]
```

- [ ] **Step 5: Update the existing tests in `tests/test_render.py`**

Mechanical: every `surface_masks` argument becomes `ribbon_mask`, and every
`surface_masks.pial` becomes `ribbon_mask`. The module fixture becomes:

```python
@pytest.fixture(scope="module")
def cuts(ribbon_mask):
    return select_cuts(ribbon_mask, n_cuts=5)
```

Drop `from fmriprepviz.ribbon import SurfaceMasks`. Two tests construct
`SurfaceMasks` directly and must be rewritten to pass a single image:

```python
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
```

- [ ] **Step 6: Run the render tests**

Run: `uv run pytest tests/test_render.py -v`
Expected: PASS, including the new both-boundaries test.

If `test_one_contour_draws_both_the_white_and_pial_boundaries` reports fewer
than 4 bands, do not relax it — inspect the rendered frame first. Write it out
with `Path("/tmp/frame.png").write_bytes(png)` and look. Anti-aliasing merging
the two rings would mean the contour linewidth is too large relative to the
5 mm gap between the radii at this zoom, which is a real finding about
legibility, not a test problem.

The full suite is expected to be **red** at this point: `cli.py` still hands a
`SurfaceMasks` to a `render_frame` that now wants one image. Steps 7–13 close
that gap; do not commit until Step 14.

- [ ] **Step 7: Write the failing CLI tests**

Add to `tests/test_cli.py`:

```python
def test_parser_accepts_the_new_ribbon_flags():
    args = build_parser().parse_args(
        [
            "/derivs", "--subject", "01",
            "--ribbon", "/tmp/r.nii.gz",
            "--fs-to-t1w-xfm", "/tmp/x.txt",
        ]
    )
    assert args.ribbon.name == "r.nii.gz"
    assert args.fs_to_t1w_xfm.name == "x.txt"


def test_end_to_end_on_a_misaligned_tree(derivatives_misaligned, tmp_path):
    """The whole point: a tree whose FreeSurfer volumes are not aligned with
    the T1w image must still produce a movie, via fMRIPrep's own T1w-space
    ribbon mask.
    """
    output = tmp_path / "qc.gif"
    exit_code = main(
        [
            str(derivatives_misaligned), "--subject", "01",
            "-o", str(output), "--n-cuts", "3",
        ]
    )
    assert exit_code == 0
    assert output.is_file()


def test_explicit_fs_dir_uses_the_freesurfer_path(
    derivatives_misaligned, tmp_path
):
    """--fs-dir must not be silently ignored because a desc-ribbon_mask
    exists, or the FreeSurfer path would be unreachable on a complete tree
    and could not be compared against the fast path at all.
    """
    output = tmp_path / "qc.gif"
    fs_dir = (
        derivatives_misaligned / "sourcedata" / "freesurfer" / "sub-01"
    )
    exit_code = main(
        [
            str(derivatives_misaligned), "--subject", "01",
            "-o", str(output), "--n-cuts", "3", "--fs-dir", str(fs_dir),
        ]
    )
    assert exit_code == 0
    assert output.is_file()


def test_explicit_ribbon_flag_is_used(derivatives, tmp_path):
    output = tmp_path / "qc.gif"
    ribbon = (
        derivatives / "sub-01" / "anat" / "sub-01_desc-ribbon_mask.nii.gz"
    )
    exit_code = main(
        [
            str(derivatives), "--subject", "01", "-o", str(output),
            "--n-cuts", "3", "--ribbon", str(ribbon),
        ]
    )
    assert exit_code == 0
    assert output.is_file()
```

Then retarget the two mismatched-`--fs-dir` tests. The failure mode genuinely
moved: `ribbon.py` now catches this before rendering, with a message that names
the transform. Replace both bodies' assertion and rewrite their docstrings:

```python
def test_mismatched_fs_dir_is_reported_before_rendering(
    derivatives, tmp_path, capsys
):
    """A --fs-dir pointing at another subject used to reach nilearn, which
    raised BoundingBoxError while resampling onto the badly shifted mask grid.
    Now the ribbon is resampled onto the T1w grid first, so the mistake
    presents as a ribbon that lands nowhere on that grid and is reported by
    ribbon.py with a message naming the transform -- earlier, and more
    specifically, than before.
    """
    fs_dir = _write_mismatched_fs_dir(tmp_path)
    for extra in ([], ["--no-upsample"]):
        exit_code = main(
            [
                str(derivatives), "--subject", "01",
                "-o", str(tmp_path / "qc.gif"), "--fs-dir", str(fs_dir),
            ]
            + extra
        )
        assert exit_code == 1
        err = capsys.readouterr().err
        assert "T1w" in err and "ribbon.mgz" in err
```

Delete `test_default_upsample_reports_a_mismatched_fs_dir_as_no_overlap` and
`test_no_upsample_reports_a_mismatched_fs_dir_as_no_overlap`, which the above
replaces, and drop the now-unused `NO_OVERLAP_MESSAGE` import if nothing else
uses it.

Update the compute-once test to monkeypatch the new name:

```python
    real_load_ribbon_mask = cli_module.load_ribbon_mask
    ...
    def counting_load_ribbon_mask(*args, **kwargs):
        masks_calls.append(1)
        return real_load_ribbon_mask(*args, **kwargs)

    monkeypatch.setattr(cli_module, "load_ribbon_mask", counting_load_ribbon_mask)
```

- [ ] **Step 8: Run them to verify they fail**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL — unrecognised arguments `--ribbon` / `--fs-to-t1w-xfm`, and
`AttributeError` on `cli_module.load_ribbon_mask`.

- [ ] **Step 9: Update `src/fmriprepviz/cli.py`**

Swap the import:

```python
from fmriprepviz.ribbon import load_ribbon_mask
```

`resolve_fs_dir` is no longer called here — `ribbon.py` calls it internally.

Add the flags next to `--fs-dir`:

```python
    parser.add_argument(
        "--ribbon", type=Path,
        help=(
            "explicit cortical ribbon mask, already in the T1w space; "
            "overrides all automatic discovery"
        ),
    )
    parser.add_argument(
        "--fs-to-t1w-xfm", type=Path,
        help=(
            "fsnative-to-T1w transform (ITK .txt) to place FreeSurfer's "
            "ribbon.mgz in the T1w space; implies the FreeSurfer path"
        ),
    )
```

In `_run`, load one mask and thread it through:

```python
    mask = load_ribbon_mask(
        args.derivatives,
        args.subject,
        ribbon=args.ribbon,
        fs_dir=args.fs_dir,
        xfm=args.fs_to_t1w_xfm,
    )
    cuts = _resolve_cuts(args, mask)
    frames = _render_frames(refs, mask, cuts, upsample=not args.no_upsample)
```

Rename the parameters of `_resolve_cuts` and `_render_frames` from `masks` to
`mask`, and inside `_resolve_cuts` call `select_cuts(mask, ...)` rather than
`select_cuts(masks.pial, ...)`.

- [ ] **Step 10: Delete the retired API from `src/fmriprepviz/ribbon.py`**

Remove `SurfaceMasks`, `load_surface_masks`, `_mask_image`, `WHITE_LABELS`,
`GRAY_LABELS`, `CORTICAL_LABELS`, and the old `_require_cortical_labels`
(superseded by `_require_gray_matter_labels`). Keep `resolve_fs_dir`,
`FreeSurferNotFoundError`, and `RIBBON_RELATIVE_PATH` exactly as they are.
Update the module docstring to describe one mask rather than two.

- [ ] **Step 11: Remove the retired tests and fixture**

Delete the `surface_masks` fixture from `tests/conftest.py`. From
`tests/test_ribbon.py`, delete every test of `load_surface_masks` — the mask
geometry, affine, and label-guard tests, all of which are covered by
`tests/test_ribbon_mask.py`. Keep the four `resolve_fs_dir` tests, which are
deliberately unchanged behaviour, and trim the imports to match.

- [ ] **Step 12: Run the whole suite**

Run: `uv run pytest`
Expected: PASS — green again, for the first time since Step 5.

- [ ] **Step 13: Confirm no reference to the retired API survives**

Run: `grep -rn "SurfaceMasks\|load_surface_masks\|surface_masks\|WHITE_LABELS\|GRAY_LABELS\|CORTICAL_LABELS" src tests`
Expected: no output.

- [ ] **Step 14: Commit**

Two commits: the tests, then the change they demand. Both are staged only now,
since the suite could not be green in between.

```bash
git add tests/test_render.py tests/test_cli.py tests/test_ribbon.py tests/conftest.py
git commit -m "test: require one contour and the new ribbon flags

Covers both cortical boundaries from a single contour, --ribbon and
--fs-to-t1w-xfm, and that an explicit FreeSurfer flag is not ignored
when fMRIPrep's own ribbon mask is present."
git add src/fmriprepviz/render.py src/fmriprepviz/cuts.py \
        src/fmriprepviz/cli.py src/fmriprepviz/ribbon.py
git commit -m "rf: draw the ribbon as one mask with one contour

The cortical gray mask is a sheet, so its 0.5 level set already traces
the white surface along its inner boundary and the pial along its outer;
two masks and two add_contours calls were redundant. Threshold at that
same level when windowing and when choosing cuts, since the mask is
float after resampling.

Add --ribbon and --fs-to-t1w-xfm, and let an explicit --fs-dir or
--fs-to-t1w-xfm select the FreeSurfer path even when fMRIPrep's own
ribbon mask is present, so neither flag can be silently ignored. Retire
SurfaceMasks and load_surface_masks."
```

---

### Task 6: Document the change

Documentation last, per the repo's commit-modularity convention.

**Files:**

- Modify: `README.md`

- [ ] **Step 1: Update the option table and the FreeSurfer prose**

Add `--ribbon` and `--fs-to-t1w-xfm` to the options table. Replace any text
saying contours come from `ribbon.mgz` with the resolution order, and state
plainly why it matters:

```markdown
### Where the contours come from

The red contours trace the cortical ribbon. The ribbon is resolved in this
order:

1. `--ribbon PATH`, if given — any binary or float mask already in the T1w
   space
2. `--fs-dir` or `--fs-to-t1w-xfm`, if either is given — FreeSurfer's
   `mri/ribbon.mgz`, moved into the T1w space
3. fMRIPrep's own `*_desc-ribbon_mask.nii.gz`, which is already in the T1w
   space
4. FreeSurfer's `mri/ribbon.mgz`, moved into the T1w space using the
   `*_from-fsnative_to-T1w_mode-image_xfm.txt` transform fMRIPrep writes

Step 4 matters when `recon-all` ran outside fMRIPrep, as with
`--fs-subjects-dir`. FreeSurfer's conformed space is then a different physical
space from fMRIPrep's T1w reference — on one real dataset, a 6° rotation and an
18.6 mm translation apart. Without the transform the contours land on the wrong
tissue and nothing reports an error, which for a registration QC tool is worse
than crashing.

A single contour is drawn, not two: the ribbon is cortical gray matter, so the
inner edge of its 0.5 level set is the white surface and the outer edge is the
pial.
```

- [ ] **Step 2: Format, verify, and commit**

```bash
npx --yes prettier -w README.md
uv run pytest
git add README.md
git commit -m "docs: describe the ribbon source order and why it matters"
```

---

## Verification

Beyond `uv run pytest`:

- [ ] **Minimum dependency check.** The stated floor must be tested:

```bash
uv run --isolated --with "nitransforms==25.1.0" --with pytest pytest
```

- [ ] **Real-data smoke run.** A standalone script, **not committed** — keep it
      in the scratchpad so the suite stays synthetic. Both paths, one subject:

```bash
# fast path: desc-ribbon_mask, no FreeSurfer directory needed
uv run fmriprepviz /data/derivatives/fmriprep-25.2.5 --subject 07 \
    -o /tmp/qc-fast.gif

# FreeSurfer path: forced by the explicit flags
XFM=/data/derivatives/fmriprep-25.2.5/sub-07/ses-nsdanat/anat/sub-07_ses-nsdanat_from-fsnative_to-T1w_mode-image_xfm.txt
uv run fmriprepviz /data/derivatives/fmriprep-25.2.5 --subject 07 \
    --fs-dir /data/sourcedata/freesurfer/sub-07 --fs-to-t1w-xfm "$XFM" \
    -o /tmp/qc-fallback.gif
```

Then confirm by eye, on a frame from each:

- the contours follow the gray/white and gray/CSF boundaries of the BOLD, not
  some offset copy of them
- left and right are not flipped
- the two outputs agree with each other

This closes Task 9 of the v1 implementation plan, open since 2026-08-16.

- [ ] **If comparing against pre-change output**, read the spec's "Comparing
      against pre-change output" section first. The old and new contours
      coincide wherever gray matter separates white matter from CSF, and
      diverge at the medial wall and around subcortical structures, where the
      old white contour followed the white-matter boundary and the new inner
      contour has no gray matter to follow. That difference is expected. What
      must hold is that no contour moves relative to the anatomy.

## Notes for the implementer

**The one trap.** `nitransforms.resampling.apply` takes the transform **as
loaded** from the ITK file. Do not invert it. An inverted transform raises no
error and produces a plausible-looking movie displaced by twice the true
offset. `test_fallback_applies_the_transform_in_the_right_direction` is the
only thing standing between that and a silent regression, so if it ever fails,
the transform handling is wrong — never adjust its tolerance to accommodate a
change.

**Behaviour that legitimately changed.** A mismatched `--fs-dir` used to
surface as nilearn's `BoundingBoxError` mapped to `NO_OVERLAP_MESSAGE`. It now
fails earlier, in `ribbon.py`, with a message naming the transform. The two
CLI tests asserting the old wording are replaced in Task 5, not weakened.

**What must not change.** The invariants the flipbook rests on: mask and cuts
computed once outside the frame loop, display bounds pinned across differing
boldref fields of view, fixed and deterministic frame size, total frame
ordering, HTML escaping. Every one has a test; none should need touching.
