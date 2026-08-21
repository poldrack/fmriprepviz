import os
import shutil

import nibabel as nib
import numpy as np
import pytest
from PIL import Image

from fmriprepviz.cli import build_parser, main
from fmriprepviz.render import NO_OVERLAP_MESSAGE


def test_parser_defaults():
    args = build_parser().parse_args(["/derivs", "--subject", "01"])
    assert args.subject == "01"
    assert args.output.name == "qc.gif"
    assert args.fps == 2
    assert args.n_cuts == 1
    assert args.display_mode == "xyz"
    assert args.no_html is False
    assert args.no_upsample is False


def test_parser_accepts_multiple_sessions():
    args = build_parser().parse_args(
        ["/derivs", "--subject", "01", "--ses", "01", "02"]
    )
    assert args.ses == ["01", "02"]


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


def test_explicit_fs_dir_uses_the_freesurfer_path(derivatives, tmp_path, capsys):
    """--fs-dir must not be silently ignored just because a desc-ribbon_mask
    is present. `derivatives` has one, so a passing --fs-dir case here could
    not distinguish "the flag was honoured" from "the flag was dropped and
    the fast path ran anyway" -- both would exit 0. Pointing --fs-dir at a
    directory with no ribbon.mgz is the discriminating case: only actually
    routing through the FreeSurfer path can fail this way, and a dropped
    flag would silently exit 0 via fMRIPrep's own ribbon mask instead.
    """
    fs_dir = tmp_path / "no_ribbon_here"
    fs_dir.mkdir()
    exit_code = main(
        [
            str(derivatives), "--subject", "01",
            "-o", str(tmp_path / "qc.gif"), "--fs-dir", str(fs_dir),
        ]
    )
    assert exit_code == 1
    assert "ribbon.mgz" in capsys.readouterr().err


def test_explicit_ribbon_flag_is_used(derivatives, tmp_path, capsys):
    """--ribbon must not be silently ignored in favour of automatic
    discovery. `derivatives` has a discoverable desc-ribbon_mask, so a
    passing case here could not tell "the flag was honoured" from "the flag
    was dropped and discovery ran instead" -- both would exit 0. Pointing
    --ribbon at a path that does not exist is the discriminating case: it
    can only fail if that exact path is loaded directly.
    """
    missing = tmp_path / "does-not-exist.nii.gz"
    exit_code = main(
        [
            str(derivatives), "--subject", "01", "-o", str(tmp_path / "qc.gif"),
            "--n-cuts", "3", "--ribbon", str(missing),
        ]
    )
    assert exit_code == 1
    assert str(missing) in capsys.readouterr().err


def test_end_to_end_writes_a_gif_and_a_viewer(derivatives, tmp_path):
    output = tmp_path / "qc.gif"
    exit_code = main(
        [str(derivatives), "--subject", "01", "-o", str(output), "--n-cuts", "3"]
    )
    assert exit_code == 0
    assert output.is_file()
    assert output.with_suffix(".html").is_file()


def test_gif_has_one_frame_per_boldref(derivatives, tmp_path):
    output = tmp_path / "qc.gif"
    main([str(derivatives), "--subject", "01", "-o", str(output), "--n-cuts", "3"])
    with Image.open(output) as gif:
        assert gif.n_frames == 5


def test_filters_reduce_the_frame_count(derivatives, tmp_path):
    output = tmp_path / "qc.gif"
    main(
        [
            str(derivatives), "--subject", "01", "-o", str(output),
            "--n-cuts", "3", "--ses", "02",
        ]
    )
    with Image.open(output) as gif:
        assert gif.n_frames == 2


def test_no_html_skips_the_viewer(derivatives, tmp_path):
    output = tmp_path / "qc.gif"
    main(
        [
            str(derivatives), "--subject", "01", "-o", str(output),
            "--n-cuts", "3", "--no-html",
        ]
    )
    assert not output.with_suffix(".html").exists()


def test_single_frame_warns_but_still_writes(derivatives, tmp_path, capsys):
    output = tmp_path / "qc.gif"
    main(
        [
            str(derivatives), "--subject", "01", "-o", str(output),
            "--n-cuts", "3", "--ses", "02", "--task", "nback",
        ]
    )
    assert "one frame" in capsys.readouterr().err
    assert output.is_file()


def test_missing_boldrefs_exits_non_zero_with_a_message(
    derivatives_without_t1w, tmp_path, capsys
):
    exit_code = main(
        [str(derivatives_without_t1w), "--subject", "01", "-o", str(tmp_path / "qc.gif")]
    )
    assert exit_code == 1
    assert "MNI152NLin2009cAsym" in capsys.readouterr().err


def test_missing_freesurfer_exits_non_zero_with_a_message(
    derivatives, tmp_path, capsys, monkeypatch
):
    monkeypatch.delenv("SUBJECTS_DIR", raising=False)
    exit_code = main(
        [
            str(derivatives), "--subject", "01", "-o", str(tmp_path / "qc.gif"),
            "--fs-dir", str(tmp_path / "nowhere"),
        ]
    )
    assert exit_code == 1
    assert "ribbon.mgz" in capsys.readouterr().err


def test_cut_coords_requires_a_single_orientation(derivatives, tmp_path, capsys):
    exit_code = main(
        [
            str(derivatives), "--subject", "01", "-o", str(tmp_path / "qc.gif"),
            "--cut-coords", "-20", "0", "20",
        ]
    )
    assert exit_code == 1
    assert "--display-mode" in capsys.readouterr().err


def test_cut_coords_are_used_when_one_orientation_is_given(derivatives, tmp_path):
    output = tmp_path / "qc.gif"
    exit_code = main(
        [
            str(derivatives), "--subject", "01", "-o", str(output),
            "--display-mode", "z", "--cut-coords", "-20", "0", "20",
        ]
    )
    assert exit_code == 0
    assert output.is_file()


def test_non_positive_fps_exits_non_zero_with_a_message(derivatives, tmp_path, capsys):
    exit_code = main(
        [
            str(derivatives), "--subject", "01", "-o", str(tmp_path / "qc.gif"),
            "--fps", "0",
        ]
    )
    assert exit_code == 1
    assert "--fps" in capsys.readouterr().err


def _write_mismatched_fs_dir(tmp_path):
    """A FreeSurfer subject directory whose ribbon shares no world-space
    voxels with any boldref in the `derivatives` fixture, simulating
    --fs-dir pointing at the wrong subject.
    """
    fs_dir = tmp_path / "wrong-subject-fs"
    mri = fs_dir / "mri"
    mri.mkdir(parents=True)
    shape = (20, 20, 20)
    affine = np.eye(4)
    affine[:3, 3] = 5000.0  # far from any boldref in the `derivatives` fixture
    data = np.zeros(shape, dtype=np.uint8)
    data[5:10, 5:10, 5:10] = 2  # white matter label
    data[10:15, 10:15, 10:15] = 3  # gray matter label
    nib.MGHImage(data, affine).to_filename(mri / "ribbon.mgz")
    return fs_dir


def test_mismatched_fs_dir_is_reported_before_rendering(
    derivatives, tmp_path, capsys
):
    """A --fs-dir pointing at another subject used to reach nilearn, which
    raised BoundingBoxError while resampling onto the badly shifted mask grid.
    Now the ribbon is resampled onto the T1w grid first, so the mistake
    presents as a ribbon that lands nowhere on that grid and is reported by
    ribbon.py with a message naming the transform -- earlier, and more
    specifically, than before.

    "from-fsnative_to-T1w" (from the transform's filename) is what actually
    pins this down: resolve_fs_dir's own FreeSurferNotFoundError also
    contains "T1w" and "ribbon.mgz" in its tried-paths listing, so those two
    substrings alone would not distinguish this message from that unrelated
    one.
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
        assert "from-fsnative_to-T1w" in err


def test_ribbon_flag_with_mismatched_affine_reports_no_overlap(
    ribbon_mask, derivatives, tmp_path, capsys
):
    """--ribbon bypasses ribbon.py's own guards entirely: the given path is
    loaded and used exactly as given, with no T1w-grid overlap check. So a
    wildly mismatched affine there is not caught until render_frame calls
    nilearn's resample_to_img, which raises BoundingBoxError. main's `except
    BoundingBoxError` clause must precede its general `except ValueError`
    clause (BoundingBoxError subclasses ValueError) to report
    NO_OVERLAP_MESSAGE instead of letting nilearn's raw wording through --
    this is the only test in the suite that still reaches that branch, since
    Task 5 retargeted the --fs-dir tests that used to.
    """
    shifted_affine = ribbon_mask.affine.copy()
    shifted_affine[:3, 3] += 1000.0
    shifted = nib.Nifti1Image(np.asarray(ribbon_mask.dataobj), shifted_affine)
    ribbon_path = tmp_path / "shifted_ribbon.nii.gz"
    shifted.to_filename(ribbon_path)

    exit_code = main(
        [
            str(derivatives), "--subject", "01",
            "-o", str(tmp_path / "qc.gif"), "--ribbon", str(ribbon_path),
        ]
    )
    assert exit_code == 1
    assert NO_OVERLAP_MESSAGE in capsys.readouterr().err


def test_cuts_and_masks_are_computed_once_regardless_of_frame_count(
    derivatives, tmp_path, monkeypatch
):
    """The registration-check guarantee this tool makes rests on every frame
    being drawn with the *same* cuts and the *same* ribbon mask. If a
    future change derived cuts or the mask per boldref instead of once from
    the ribbon, every frame would silently show slightly different anatomy
    and the movie would stop being a valid registration test -- while every
    other test in this suite, which only checks frame counts and file
    existence, would keep passing. This test guards that invariant directly
    by counting calls through a real (not stubbed) end-to-end run.
    """
    import fmriprepviz.cli as cli_module

    real_select_cuts = cli_module.select_cuts
    real_load_ribbon_mask = cli_module.load_ribbon_mask
    cuts_calls = []
    masks_calls = []

    def counting_select_cuts(*args, **kwargs):
        cuts_calls.append(1)
        return real_select_cuts(*args, **kwargs)

    def counting_load_ribbon_mask(*args, **kwargs):
        masks_calls.append(1)
        return real_load_ribbon_mask(*args, **kwargs)

    monkeypatch.setattr(cli_module, "select_cuts", counting_select_cuts)
    monkeypatch.setattr(cli_module, "load_ribbon_mask", counting_load_ribbon_mask)

    output = tmp_path / "qc.gif"
    exit_code = main(
        [str(derivatives), "--subject", "01", "-o", str(output), "--n-cuts", "3"]
    )

    assert exit_code == 0
    with Image.open(output) as gif:
        assert gif.n_frames == 5  # multiple frames, yet...
    assert len(cuts_calls) == 1  # ...cuts are computed once
    assert len(masks_calls) == 1  # ...masks are loaded once


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
def test_mp4_output(derivatives, tmp_path):
    output = tmp_path / "qc.mp4"
    main([str(derivatives), "--subject", "01", "-o", str(output), "--n-cuts", "3"])
    assert output.is_file()


def test_html_title_names_the_subject(derivatives, tmp_path):
    """Every viewer tab currently reads the generic 'fmriprepviz'; with
    several subjects' viewers open at once, the browser tab bar gives no
    way to tell them apart.
    """
    output = tmp_path / "qc.gif"
    main(
        [str(derivatives), "--subject", "01", "-o", str(output), "--n-cuts", "3"]
    )
    html_text = output.with_suffix(".html").read_text()
    assert "<title>fmriprepviz sub-01</title>" in html_text


def test_missing_output_directory_is_created(derivatives, tmp_path):
    """-o subdir/qc.gif is an ordinary thing to type; the parent directory
    not existing yet should not be treated as an error.
    """
    output = tmp_path / "does" / "not" / "exist" / "qc.gif"
    exit_code = main(
        [str(derivatives), "--subject", "01", "-o", str(output), "--n-cuts", "3"]
    )
    assert exit_code == 0
    assert output.is_file()


def test_unwritable_output_location_exits_non_zero_with_a_message(
    derivatives, tmp_path, capsys
):
    """An -o path under a read-only directory must be reported cleanly
    (exit 1, message on stderr), not as an uncaught OSError traceback.
    """
    readonly_dir = tmp_path / "readonly"
    readonly_dir.mkdir()
    readonly_dir.chmod(0o555)
    output = readonly_dir / "qc.gif"
    try:
        exit_code = main(
            [str(derivatives), "--subject", "01", "-o", str(output), "--n-cuts", "3"]
        )
        assert exit_code == 1
        assert capsys.readouterr().err.strip() != ""
    finally:
        readonly_dir.chmod(0o755)


def _write_corrupt_boldref_tree(tmp_path):
    """A minimal derivatives tree whose single space-T1w boldref is garbage
    bytes with a valid-looking name, simulating a truncated/corrupt file.
    """
    root = tmp_path / "corrupt_deriv"
    func = root / "sub-01" / "ses-01" / "func"
    func.mkdir(parents=True)
    name = "sub-01_ses-01_task-rest_run-1_space-T1w_boldref.nii.gz"
    (func / name).write_bytes(b"not a real nifti file, just garbage bytes")
    return root


def test_corrupt_boldref_exits_non_zero_with_a_message(derivatives, tmp_path, capsys):
    """A truncated or corrupt boldref .nii.gz must be reported cleanly, not
    as an uncaught nibabel ImageFileError traceback. Reuses the shared
    `derivatives` fixture's FreeSurfer directory via --fs-dir, since cuts
    and masks are derived from the ribbon alone and never touch the
    (corrupt) boldref before nib.load(ref.path) is reached.
    """
    corrupt_root = _write_corrupt_boldref_tree(tmp_path)
    fs_dir = derivatives / "sourcedata" / "freesurfer" / "sub-01"
    exit_code = main(
        [
            str(corrupt_root), "--subject", "01", "-o", str(tmp_path / "qc.gif"),
            "--fs-dir", str(fs_dir), "--n-cuts", "3",
        ]
    )
    assert exit_code == 1
    assert capsys.readouterr().err.strip() != ""


def _write_stub_ffmpeg(bin_dir):
    """A stub `ffmpeg` that is found on PATH but always fails, simulating a
    minimal ffmpeg build without libx264 -- a common homebrew/conda case.
    """
    stub = bin_dir / "ffmpeg"
    stub.write_text("#!/bin/sh\necho 'stub ffmpeg: no encoder available' >&2\nexit 1\n")
    stub.chmod(0o755)
    return stub


def test_ffmpeg_encode_failure_exits_non_zero_with_a_message(
    derivatives, tmp_path, monkeypatch, capsys
):
    """movie.write_mp4 raises a plain RuntimeError with a helpful message
    when ffmpeg itself fails, but the CLI's catch tuple used to list only
    RuntimeError *subclasses* -- so this specific, deliberately-constructed
    helpful error was exactly the one that escaped as a traceback.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_stub_ffmpeg(bin_dir)
    monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ.get("PATH", ""))

    output = tmp_path / "qc.mp4"
    exit_code = main(
        [str(derivatives), "--subject", "01", "-o", str(output), "--n-cuts", "3"]
    )
    assert exit_code == 1
    assert "ffmpeg" in capsys.readouterr().err.lower()
