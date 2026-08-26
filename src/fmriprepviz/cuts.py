"""Choose the anatomical locations shown identically in every frame.

The cuts derive from the ribbon mask alone, so they are the same for every
session. That is what makes the flipbook a registration test: anything that
moves between frames is the BOLD, not the display.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from fmriprepviz.ribbon import MASK_THRESHOLD

AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
DEFAULT_N_CUTS = 1
CENTRAL_FRACTION = 0.8


def select_cuts(
    mask_img,
    n_cuts: int = DEFAULT_N_CUTS,
    display_modes: Sequence[str] = ("x", "y", "z"),
) -> dict[str, list[float]]:
    """Evenly spaced cut coordinates across the central part of the mask."""
    unknown = [mode for mode in display_modes if mode not in AXIS_INDEX]
    if unknown:
        raise ValueError(
            f"Unknown display mode(s) {unknown}; expected any of "
            f"{sorted(AXIS_INDEX)}"
        )
    world = _world_coordinates(mask_img)
    return {
        mode: _even_cuts(world[:, AXIS_INDEX[mode]], n_cuts)
        for mode in display_modes
    }


def _world_coordinates(mask_img) -> np.ndarray:
    ijk = np.argwhere(np.asarray(mask_img.dataobj) >= MASK_THRESHOLD)
    homogeneous = np.column_stack([ijk, np.ones(len(ijk))])
    return (mask_img.affine @ homogeneous.T).T[:, :3]


def _even_cuts(values: np.ndarray, n_cuts: int) -> list[float]:
    low, high = float(values.min()), float(values.max())
    if n_cuts == 1:
        # np.linspace(a, b, 1) returns [a], which would put the sole cut at the
        # edge of the brain -- the least informative slice there is. One cut
        # means the middle.
        return [(low + high) / 2.0]
    margin = (1.0 - CENTRAL_FRACTION) / 2.0 * (high - low)
    return [float(v) for v in np.linspace(low + margin, high - margin, n_cuts)]
