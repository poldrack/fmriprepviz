import pytest

from fmriprepviz.discovery import (
    AmbiguousAnatError,
    AnatNotFoundError,
    NoBoldRefsError,
    find_boldrefs,
    find_unique_anat,
    parse_entities,
)


def test_parses_bids_entities_from_a_filename():
    entities = parse_entities(
        "sub-01_ses-02_task-rest_run-10_space-T1w_boldref.nii.gz"
    )
    assert entities["sub"] == "01"
    assert entities["ses"] == "02"
    assert entities["task"] == "rest"
    assert entities["run"] == "10"
    assert entities["space"] == "T1w"


def test_finds_every_t1w_boldref(derivatives):
    refs = find_boldrefs(derivatives, "01")
    assert len(refs) == 5


def test_ignores_boldrefs_in_other_spaces(derivatives):
    refs = find_boldrefs(derivatives, "01")
    assert all(ref.entities["space"] == "T1w" for ref in refs)


def test_orders_by_session_then_task_then_run_numerically(derivatives):
    refs = find_boldrefs(derivatives, "01")
    ordered = [(r.entities["ses"], r.entities["task"], r.entities["run"]) for r in refs]
    assert ordered == [
        ("01", "rest", "1"),
        ("01", "rest", "2"),
        ("01", "rest", "10"),
        ("02", "nback", "1"),
        ("02", "rest", "1"),
    ]


def test_label_reads_as_human_readable_entities(derivatives):
    refs = find_boldrefs(derivatives, "01")
    assert refs[0].label == "ses-01 task-rest run-1"


def test_filters_by_session(derivatives):
    refs = find_boldrefs(derivatives, "01", ses=["02"])
    assert len(refs) == 2
    assert all(ref.entities["ses"] == "02" for ref in refs)


def test_filters_by_task(derivatives):
    refs = find_boldrefs(derivatives, "01", task=["nback"])
    assert len(refs) == 1


def test_filters_by_run(derivatives):
    refs = find_boldrefs(derivatives, "01", run=["1"])
    assert len(refs) == 3


def test_filters_combine(derivatives):
    refs = find_boldrefs(derivatives, "01", ses=["01"], run=["2"])
    assert len(refs) == 1
    assert refs[0].entities["run"] == "2"


def test_missing_t1w_output_names_the_spaces_that_are_present(
    derivatives_without_t1w,
):
    with pytest.raises(NoBoldRefsError) as excinfo:
        find_boldrefs(derivatives_without_t1w, "01")
    message = str(excinfo.value)
    assert "MNI152NLin2009cAsym" in message
    assert "T1w" in message


def test_unknown_subject_reports_the_directory_searched(derivatives):
    with pytest.raises(NoBoldRefsError) as excinfo:
        find_boldrefs(derivatives, "99")
    assert "sub-99" in str(excinfo.value)


def test_over_restrictive_filters_report_the_filters(derivatives):
    with pytest.raises(NoBoldRefsError) as excinfo:
        find_boldrefs(derivatives, "01", task=["nosuchtask"])
    assert "nosuchtask" in str(excinfo.value)


def _write_boldref(func_dir, name):
    import nibabel as nib
    import numpy as np

    func_dir.mkdir(parents=True, exist_ok=True)
    affine = np.eye(4)
    image = nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), affine)
    image.to_filename(func_dir / name)


def _write_dir_ap_pa_tree(tmp_path):
    """Two runs identical in every LABEL_ENTITIES-style field except the
    fieldmap-correction `dir` entity -- the AP/PA pattern fMRIPrep produces
    when it has reversed phase-encoding runs to correct for distortion.
    """
    func = tmp_path / "sub-01" / "ses-01" / "func"
    _write_boldref(
        func, "sub-01_ses-01_task-rest_run-1_dir-AP_space-T1w_boldref.nii.gz"
    )
    _write_boldref(
        func, "sub-01_ses-01_task-rest_run-1_dir-PA_space-T1w_boldref.nii.gz"
    )
    return tmp_path


def test_dir_entity_runs_get_distinct_labels(tmp_path):
    """ses/task/acq/run alone collide for AP/PA fieldmap-corrected pairs, a
    common fMRIPrep output; the label must still tell them apart.
    """
    root = _write_dir_ap_pa_tree(tmp_path)
    refs = find_boldrefs(root, "01")
    labels = {ref.label for ref in refs}
    assert len(labels) == 2
    assert any("dir-AP" in label for label in labels)
    assert any("dir-PA" in label for label in labels)


def test_ordering_is_stable_regardless_of_filesystem_order(tmp_path, monkeypatch):
    """`Path.glob` order reflects filesystem/inode order, not anything
    meaningful, so two AP/PA runs tied on every SORT_ENTITIES field must not
    let that order leak into frame order -- otherwise the same dataset could
    play back in a different sequence on another machine, or after being
    re-copied. This is verified by reversing what glob() yields and checking
    that find_boldrefs' output is unaffected: it only stays unaffected if the
    sort key is total (never tied), per the fix's `str(ref.path)` tie-break.
    """
    from pathlib import Path

    root = _write_dir_ap_pa_tree(tmp_path)
    real_glob = Path.glob

    def reversed_glob(self, pattern):
        return reversed(list(real_glob(self, pattern)))

    first = [ref.path for ref in find_boldrefs(root, "01")]
    monkeypatch.setattr(Path, "glob", reversed_glob)
    second = [ref.path for ref in find_boldrefs(root, "01")]
    assert first == second


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
