"""Locate and order the space-T1w BOLD reference images of one subject."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

BOLDREF_SUFFIX = "_space-T1w_boldref.nii.gz"
ANY_BOLDREF_GLOB = "**/func/*_boldref.nii.gz"

# BIDS canonical entity order for everything a boldref filename may carry,
# excluding `sub` (implicit -- every ref is one subject's), `space` (always
# T1w by construction) and `desc` (not a distinguishing dimension here).
# Entities outside this list (unknown to BIDS-Derivatives at the time this
# was written) still appear in the label, appended in sorted order, rather
# than being silently dropped.
LABEL_ENTITY_ORDER = ("ses", "task", "acq", "ce", "dir", "rec", "run", "echo", "part")
LABEL_EXCLUDED_ENTITIES = frozenset({"sub", "space", "desc"})

# Deliberately narrower than LABEL_ENTITY_ORDER: sorting only needs to
# reflect the dimensions playback should step through in a stable, numeric
# order. Entities like `dir` or `echo` matter for telling frames apart (so
# they belong in the label) but not for ordering them, so they are left out
# here and instead fall through to the final str(path) tie-break below.
SORT_ENTITIES = ("ses", "task", "acq", "run")

_ENTITY_RE = re.compile(r"(?:^|_)(?P<key>[a-zA-Z0-9]+)-(?P<value>[a-zA-Z0-9]+)")


class NoBoldRefsError(RuntimeError):
    """No usable BOLD reference images were found."""


class AnatNotFoundError(RuntimeError):
    """No anatomical derivative matched the requested pattern."""


class AmbiguousAnatError(RuntimeError):
    """Several anatomical derivatives matched, so the T1w space is ambiguous."""


@dataclass(frozen=True)
class BoldRef:
    """One BOLD reference image and the BIDS entities of its filename."""

    path: Path
    entities: dict[str, str]

    @property
    def label(self) -> str:
        shown = {
            k: v for k, v in self.entities.items() if k not in LABEL_EXCLUDED_ENTITIES
        }
        known = [k for k in LABEL_ENTITY_ORDER if k in shown]
        unknown = sorted(k for k in shown if k not in LABEL_ENTITY_ORDER)
        return " ".join(f"{k}-{shown[k]}" for k in known + unknown)


def parse_entities(name: str) -> dict[str, str]:
    """Extract BIDS key-value entities from a filename."""
    return {m["key"]: m["value"] for m in _ENTITY_RE.finditer(name)}


def find_boldrefs(
    derivatives_root,
    subject: str,
    ses: Sequence[str] | None = None,
    task: Sequence[str] | None = None,
    run: Sequence[str] | None = None,
) -> list[BoldRef]:
    """Return this subject's space-T1w boldrefs, ordered for playback."""
    subject_dir = Path(derivatives_root) / f"sub-{subject}"
    found = [
        BoldRef(path, parse_entities(path.name))
        for path in subject_dir.glob(f"**/func/*{BOLDREF_SUFFIX}")
    ]
    filters = {"ses": ses, "task": task, "run": run}
    kept = [ref for ref in found if _matches(ref, filters)]
    if not kept:
        raise NoBoldRefsError(_explain_empty(subject_dir, found, filters))
    return sorted(kept, key=_sort_key)


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


def _matches(ref: BoldRef, filters: dict[str, Sequence[str] | None]) -> bool:
    return all(
        wanted is None or ref.entities.get(key) in set(wanted)
        for key, wanted in filters.items()
    )


def _sort_key(ref: BoldRef) -> tuple:
    # str(ref.path) is a final, total tie-break: without it, refs identical
    # on every SORT_ENTITIES field (e.g. an AP/PA fieldmap-corrected pair)
    # would sort however Path.glob happened to yield them, which depends on
    # filesystem/inode order and is not guaranteed stable across machines
    # or after the dataset is re-copied.
    return tuple(
        _natural(ref.entities.get(key, "")) for key in SORT_ENTITIES
    ) + (str(ref.path),)


def _natural(value: str) -> tuple[int, int, str]:
    """Sort numeric entity values numerically, so run-2 precedes run-10."""
    if value.isdigit():
        return (0, int(value), "")
    return (1, 0, value)


def _explain_empty(subject_dir: Path, found, filters) -> str:
    if found:
        active = {k: list(v) for k, v in filters.items() if v is not None}
        return (
            f"No space-T1w boldref under {subject_dir} matched the requested "
            f"filters {active}. Relax or remove them."
        )
    spaces = _spaces_present(subject_dir)
    if spaces:
        return (
            f"No space-T1w boldref found under {subject_dir}. The spaces "
            f"present are: {', '.join(spaces)}. Re-run fMRIPrep with T1w "
            "among its --output-spaces to produce the images this tool needs."
        )
    return (
        f"No boldref images found under {subject_dir}. Check the derivatives "
        "path and the subject label."
    )


def _spaces_present(subject_dir: Path) -> list[str]:
    spaces = {
        entities["space"]
        for path in subject_dir.glob(ANY_BOLDREF_GLOB)
        if "space" in (entities := parse_entities(path.name))
    }
    return sorted(spaces)
