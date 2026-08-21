import numpy as np
import pytest

from conftest import PIAL_RADIUS_MM
from fmriprepviz.cuts import CENTRAL_FRACTION, DEFAULT_N_CUTS, select_cuts


def test_returns_one_entry_per_display_mode(ribbon_mask):
    cuts = select_cuts(ribbon_mask)
    assert set(cuts) == {"x", "y", "z"}


def test_honours_the_requested_number_of_cuts(ribbon_mask):
    cuts = select_cuts(ribbon_mask, n_cuts=5)
    assert all(len(values) == 5 for values in cuts.values())


def test_honours_the_requested_display_modes(ribbon_mask):
    cuts = select_cuts(ribbon_mask, display_modes=("z",))
    assert set(cuts) == {"z"}


def test_cuts_stay_inside_the_mask_extent(ribbon_mask):
    cuts = select_cuts(ribbon_mask)
    for values in cuts.values():
        assert min(values) > -PIAL_RADIUS_MM
        assert max(values) < PIAL_RADIUS_MM


def test_cuts_avoid_the_outer_edges_of_the_mask(ribbon_mask):
    """The outermost slices of a brain are nearly empty and waste frames."""
    cuts = select_cuts(ribbon_mask)
    margin = (1 - CENTRAL_FRACTION) / 2 * (2 * PIAL_RADIUS_MM)
    for values in cuts.values():
        assert min(values) > -PIAL_RADIUS_MM + margin - 2.0


def test_cuts_of_a_centred_sphere_are_symmetric(ribbon_mask):
    cuts = select_cuts(ribbon_mask)
    for values in cuts.values():
        assert pytest.approx(values[0], abs=2.0) == -values[-1]


def test_cuts_are_evenly_spaced(ribbon_mask):
    cuts = select_cuts(ribbon_mask, n_cuts=7)
    for values in cuts.values():
        spacing = np.diff(values)
        assert np.allclose(spacing, spacing[0])


def test_selection_is_deterministic(ribbon_mask):
    assert select_cuts(ribbon_mask) == select_cuts(ribbon_mask)


def test_unknown_display_mode_is_rejected(ribbon_mask):
    with pytest.raises(ValueError, match="w"):
        select_cuts(ribbon_mask, display_modes=("w",))


def test_single_cut_lands_at_the_centre_not_the_edge(ribbon_mask):
    """np.linspace(low, high, 1) returns [low], so asking for one cut used to
    give the outermost slice of the central band rather than the middle of the
    brain -- the least informative slice available, and the opposite of what
    "show me one slice" means. The synthetic ribbon mask is a gray-matter
    shell centred on the world origin, so every axis's single cut must sit
    at 0.
    """
    cuts = select_cuts(ribbon_mask, n_cuts=1)
    for mode, values in cuts.items():
        assert len(values) == 1
        assert abs(values[0]) < 2.0, f"{mode} cut at {values[0]}, expected ~0"


def test_default_is_a_single_cut(ribbon_mask):
    """One large legible slice per orientation is the default view; multi-cut
    coverage is available by asking for it."""
    assert DEFAULT_N_CUTS == 1
    assert all(len(v) == 1 for v in select_cuts(ribbon_mask).values())
