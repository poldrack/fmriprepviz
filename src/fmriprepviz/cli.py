"""Command-line entry point for fmriprepviz."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib
from nibabel.filebasedimages import ImageFileError
from nilearn.image.resampling import BoundingBoxError

from fmriprepviz.cuts import DEFAULT_N_CUTS, select_cuts
from fmriprepviz.discovery import (
    DEFAULT_ORDER,
    SORT_ENTITIES_BY_ORDER,
    NoBoldRefsError,
    find_boldrefs,
)
from fmriprepviz.movie import DEFAULT_FPS, FFmpegNotFoundError, write_movie
from fmriprepviz.render import NO_OVERLAP_MESSAGE, NoMaskOverlapError, render_frame
from fmriprepviz.ribbon import (
    FreeSurferNotFoundError,
    load_surface_masks,
    resolve_fs_dir,
)
from fmriprepviz.viewer import write_html

DEFAULT_OUTPUT = Path("qc.gif")
DEFAULT_DISPLAY_MODE = "xyz"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fmriprepviz",
        description=(
            "Build a flipbook movie of a subject's space-T1w BOLD reference "
            "images with the FreeSurfer white and pial surfaces overlaid in "
            "red, for assessing registration across sessions."
        ),
    )
    parser.add_argument("derivatives", type=Path, help="fMRIPrep derivatives root")
    parser.add_argument("--subject", required=True, help="subject label, without 'sub-'")
    parser.add_argument("--ses", nargs="+", help="restrict to these session labels")
    parser.add_argument("--task", nargs="+", help="restrict to these task labels")
    parser.add_argument("--run", nargs="+", help="restrict to these run labels")
    parser.add_argument(
        "--order", choices=tuple(SORT_ENTITIES_BY_ORDER), default=DEFAULT_ORDER,
        help=(
            "frame order: session first, or task/acquisition/run then session "
            "for comparing matching runs across sessions (default: session)"
        ),
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=DEFAULT_OUTPUT,
        help="output movie; .gif or .mp4 (default: qc.gif)",
    )
    parser.add_argument(
        "--fps", type=int, default=DEFAULT_FPS, help="frames per second"
    )
    parser.add_argument(
        "--n-cuts", type=int, default=DEFAULT_N_CUTS,
        help="number of cuts per orientation",
    )
    parser.add_argument(
        "--display-mode", default=DEFAULT_DISPLAY_MODE,
        help="orientations to show, as letters from x, y, z (default: xyz)",
    )
    parser.add_argument(
        "--cut-coords", nargs="+", type=float,
        help="explicit cut coordinates in mm; requires a single --display-mode",
    )
    parser.add_argument(
        "--fs-dir", type=Path,
        help="FreeSurfer subject directory containing mri/ribbon.mgz",
    )
    parser.add_argument(
        "--no-html", action="store_true", help="skip the HTML viewer"
    )
    parser.add_argument(
        "--no-upsample", action="store_true",
        help=(
            "show BOLD with nearest-neighbour (blocky, no interpolation) "
            "resampling instead of smooth linear resampling; the display "
            "grid is still pinned to the anatomy either way"
        ),
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _run(args)
    except BoundingBoxError:
        # Must precede ValueError: BoundingBoxError subclasses it, and Python
        # matches except clauses in order, so the general clause below would
        # otherwise catch it first and print nilearn's raw wording instead.
        print(f"error: {NO_OVERLAP_MESSAGE}", file=sys.stderr)
        return 1
    except (RuntimeError, OSError, ValueError, ImageFileError) as error:
        # Every project-specific exception (NoBoldRefsError,
        # FreeSurferNotFoundError, FFmpegNotFoundError, NoMaskOverlapError,
        # ...) already subclasses RuntimeError, so this single clause covers
        # them along with: a plain RuntimeError such as movie.py's own
        # ffmpeg-failure message; OSError, for an unwritable -o location;
        # and nibabel's ImageFileError, for a truncated or corrupt boldref.
        print(f"error: {error}", file=sys.stderr)
        return 1


def _run(args) -> int:
    _validate_fps(args.fps)
    refs = find_boldrefs(
        args.derivatives, args.subject, ses=args.ses, task=args.task, run=args.run,
        order=args.order,
    )
    if len(refs) == 1:
        print(
            "warning: only one frame was found, so the movie cannot show "
            "cross-session drift.",
            file=sys.stderr,
        )
    masks = load_surface_masks(
        resolve_fs_dir(args.derivatives, args.subject, args.fs_dir)
    )
    cuts = _resolve_cuts(args, masks)
    frames = _render_frames(refs, masks, cuts, upsample=not args.no_upsample)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_movie(frames, args.output, fps=args.fps)
    print(f"wrote {args.output}", file=sys.stderr)
    if not args.no_html:
        html_path = args.output.with_suffix(".html")
        write_html(
            frames,
            [ref.label for ref in refs],
            html_path,
            fps=args.fps,
            title=f"fmriprepviz sub-{args.subject}",
        )
        print(f"wrote {html_path}", file=sys.stderr)
    return 0


def _validate_fps(fps: int) -> None:
    if fps <= 0:
        raise ValueError(f"--fps must be positive, got {fps}.")


def _resolve_cuts(args, masks) -> dict[str, list[float]]:
    display_modes = tuple(args.display_mode)
    if args.cut_coords is None:
        return select_cuts(
            masks.pial, n_cuts=args.n_cuts, display_modes=display_modes
        )
    if len(display_modes) != 1:
        raise ValueError(
            "--cut-coords applies to one orientation only; pass a single "
            "letter to --display-mode, for example --display-mode z."
        )
    return {display_modes[0]: list(args.cut_coords)}


def _render_frames(refs, masks, cuts, upsample: bool) -> list[bytes]:
    frames = []
    for index, ref in enumerate(refs, start=1):
        label = f"{index}/{len(refs)}  {ref.label}"
        print(f"rendering {label}", file=sys.stderr)
        frames.append(
            render_frame(
                nib.load(ref.path), masks, cuts, label, upsample=upsample
            )
        )
    return frames
