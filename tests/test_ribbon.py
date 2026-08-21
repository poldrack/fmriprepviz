import pytest

from fmriprepviz.ribbon import FreeSurferNotFoundError, resolve_fs_dir


def test_resolves_the_freesurfer_dir_inside_sourcedata(derivatives):
    resolved = resolve_fs_dir(derivatives, "01")
    assert resolved == derivatives / "sourcedata" / "freesurfer" / "sub-01"


def test_explicit_fs_dir_takes_precedence(derivatives):
    explicit = derivatives / "sourcedata" / "freesurfer" / "sub-01"
    assert resolve_fs_dir(derivatives, "99", fs_dir=explicit) == explicit


def test_falls_back_to_subjects_dir(derivatives, monkeypatch, tmp_path):
    monkeypatch.setenv(
        "SUBJECTS_DIR", str(derivatives / "sourcedata" / "freesurfer")
    )
    empty_root = tmp_path / "no_sourcedata"
    empty_root.mkdir()
    resolved = resolve_fs_dir(empty_root, "01")
    assert resolved == derivatives / "sourcedata" / "freesurfer" / "sub-01"


def test_unresolvable_fs_dir_lists_every_path_tried(tmp_path, monkeypatch):
    monkeypatch.delenv("SUBJECTS_DIR", raising=False)
    with pytest.raises(FreeSurferNotFoundError) as excinfo:
        resolve_fs_dir(tmp_path, "01")
    message = str(excinfo.value)
    assert "sourcedata" in message
    assert "ribbon.mgz" in message
