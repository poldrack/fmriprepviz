# fmriprepviz

Build a flipbook movie from an fMRIPrep subject's `space-T1w` BOLD reference
images, with the FreeSurfer white and pial surfaces overlaid in red, to assess
BOLD-to-anatomical registration across sessions.

Every frame shows the same anatomical cuts and the same surface contours, so
anything that appears to move as the movie plays is misregistration.

## Requirements

- fMRIPrep run with `T1w` among its `--output-spaces`
- A FreeSurfer subject directory containing `mri/ribbon.mgz`
- `ffmpeg` on `PATH`, only if you want MP4 output

## Install

For development (clones the repo and runs commands with `uv run`):

```bash
uv sync
```

To just get the `fmriprepviz` command on your PATH:

```bash
uv tool install .
```

## Which directory do I point it at?

The positional argument is the **fMRIPrep output directory** — the one that
directly contains `sub-<label>/`, alongside `dataset_description.json`. It is
not a parent directory holding `fmriprep/` and `freesurfer/` side by side.

The tool needs two things: that directory, and the FreeSurfer subject
directory for the same subject. How you pass the second depends on your layout.

**Layout A — FreeSurfer inside the fMRIPrep output.** This is what fMRIPrep
writes by default, and `--fs-dir` is not needed:

```
derivatives/fmriprep/          <- point at this
├── dataset_description.json
├── sub-01/
│   └── ses-01/func/sub-01_ses-01_task-rest_run-1_space-T1w_boldref.nii.gz
└── sourcedata/freesurfer/
    └── sub-01/mri/ribbon.mgz
```

```bash
uv run fmriprepviz derivatives/fmriprep --subject 01 -o qc.gif
```

**Layout B — FreeSurfer as a sibling.** Common when FreeSurfer was run
separately, or when fMRIPrep was given precomputed anatomical derivatives.
Here `--fs-dir` is required, and it points at the **subject** directory, not at
the FreeSurfer root:

```
derivatives/out/
├── fmriprep/                  <- point the positional argument at this
│   ├── dataset_description.json
│   └── sub-01/...
└── freesurfer/
    └── sub-01/                <- point --fs-dir at this
        └── mri/ribbon.mgz
```

```bash
uv run fmriprepviz derivatives/out/fmriprep \
  --subject 01 \
  --fs-dir derivatives/out/freesurfer/sub-01 \
  -o qc.gif
```

Without `--fs-dir`, the tool looks in `<fmriprep>/sourcedata/freesurfer/sub-<label>`
first, then `$SUBJECTS_DIR/sub-<label>`. If neither exists it lists every path
it tried, so you can see what it expected.

## Use

```bash
uv run fmriprepviz /path/to/fmriprep --subject 01 -o qc.gif
```

This writes `qc.gif` and a self-contained `qc.html` viewer with a frame slider,
prev/next buttons, play/pause and arrow-key stepping.

Useful options:

| Option | Purpose |
|---|---|
| `--ses 01 02` | restrict to these sessions |
| `--task rest` | restrict to this task |
| `--run 1` | restrict to this run |
| `--order task` | group by task, acquisition and run, then step through sessions for each matching run (default: `session`, which groups by session, task, acquisition and run) |
| `--fs-dir PATH` | FreeSurfer **subject** directory (the one containing `mri/ribbon.mgz`), when it is not at `<fmriprep>/sourcedata/freesurfer/sub-<label>` |
| `--n-cuts 7` | cuts per orientation (default 1). One cut per orientation gives three large slices through the centre of the brain, side by side. Higher values give more coverage in smaller panels, stacked one row per orientation. |
| `--display-mode z` | orientations to show, letters from `x`, `y`, `z` |
| `--cut-coords -20 0 20` | explicit cuts in mm; needs a single `--display-mode` |
| `--fps 2` | playback speed |
| `--no-html` | skip the viewer |
| `--no-upsample` | nearest-neighbour (blocky) resampling instead of linear; the display grid stays pinned either way |

`-o qc.mp4` produces MP4 instead of GIF.

## Scope

Only `space-T1w` boldrefs are supported. FreeSurfer surfaces exist solely in
the subject's native anatomical space, so template spaces such as
`MNI152NLin2009cAsym` would require warping the surfaces, which this tool does
not do.

## Development

```bash
uv run pytest
```
