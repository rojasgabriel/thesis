"""Track the body parts of the mouse in a bottom-view video with DeepLabCut.

Input: one bottom-view video (640 x 512, 80 frames/s), given as a path or as a
subject and a session of the lab data folder. Output, in `--out`: the standard
DeepLabCut files `<video name>DLC_<model name>.h5` and `.csv`. They have one
row for each video frame, and for each body part the columns `x` and `y`
(pixels from the top left corner of the frame) and `likelihood` (0 to 1).
`--labeled-video` adds a copy of the video with the body parts drawn on it.

Body parts: `nose`, `frontpaw_l`, `frontpaw_r`, `hindpaw_l`, `hindpaw_r`,
`body_mid` (center of the trunk), `tail_base`, `tail_mid` (half of the tail
length), `tail_tip`. A paw is its palm or sole, not the digits. Left and right
are the sides of the mouse. The camera looks up through the floor, so the
image is mirrored: when the mouse faces the ports (the top of the image), its
left side is on the right of the image.

The model is a DeepLabCut project (`--config`, or the `DLC_CONFIG` environment
variable); the command uses shuffle 1 of the current iteration of the project.
DeepLabCut is not a dependency of this package: the job runs in the conda
environment `--env`. The command submits one Slurm job with one GPU and
returns; the job analyzes all frames of the video.

A body part that is hidden gets a low likelihood, not a missing value. Drop
positions below a likelihood threshold (0.6 is the DeepLabCut default) before
any analysis. The rows are video frames: use the frame times of the session to
put them on the clock of the recording.

    uv run dlc-pose GRB064 20260916_132939
    uv run dlc-pose /path/to/BottomView.avi --out results/dlc --labeled-video
"""

import argparse
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path


def conda_python(env: str) -> str:
    """Return the Python executable of a conda environment (name or path).

    The job calls it directly: a `conda activate` in the job script does not
    win over the uv environment that Slurm copies from the submitting shell.
    """
    conda = shutil.which("conda")
    if conda is None:
        raise RuntimeError("conda is not on PATH; run from a login shell")
    listing = subprocess.run(
        [conda, "env", "list", "--json"], capture_output=True, text=True, check=True
    )
    for prefix in json.loads(listing.stdout)["envs"]:
        if prefix == env or Path(prefix).name == env:
            return str(Path(prefix) / "bin" / "python")
    raise ValueError(f"no conda environment {env!r}")


def bottom_view_video(subject: str, session: str) -> Path:
    """Return the bottom-view video of a session in the lab data folder."""
    from labdata.utils import prefs

    folder = Path(prefs["local_paths"][0]) / subject / session / "chipmunk"
    videos = sorted(folder.glob("*BottomView*.avi"))
    if len(videos) != 1:
        raise FileNotFoundError(
            f"expected one bottom-view video in {folder}, found {len(videos)}"
        )
    return videos[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("video", nargs="+", help="Video path, or subject and session.")
    parser.add_argument(
        "--config",
        default=os.environ.get("DLC_CONFIG"),
        help="config.yaml of the DeepLabCut project (default: $DLC_CONFIG).",
    )
    parser.add_argument("--out", default="results/dlc", help="Output folder.")
    parser.add_argument("--env", default="DEEPLABCUT", help="Conda environment.")
    parser.add_argument(
        "--labeled-video",
        action="store_true",
        help="Also write a video with the body parts drawn on it.",
    )
    parser.add_argument("--walltime", default="04:00:00")
    parser.add_argument(
        "--partition", help="Slurm partition (default: the cluster default)."
    )
    args = parser.parse_args()
    if len(args.video) > 2:
        parser.error("give one video path, or one subject and one session")
    if not args.config:
        parser.error("give --config or set DLC_CONFIG")
    config = Path(args.config).expanduser().resolve()
    video = (
        Path(args.video[0]).expanduser().resolve()
        if len(args.video) == 1
        else bottom_view_video(*args.video)
    )
    for path in (config, video):
        if not path.is_file():
            parser.error(f"{path} does not exist")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    script = (
        "import deeplabcut; "
        f"job = ({str(config)!r}, [{str(video)!r}]); "
        f"deeplabcut.analyze_videos(*job, destfolder={str(out)!r}, save_as_csv=True)"
    )
    if args.labeled_video:
        script += (
            "; deeplabcut.create_labeled_video("
            f"*job, destfolder={str(out)!r}, draw_skeleton=True)"
        )
    from labdata.compute.schedulers import slurm_submit

    python = shlex.quote(conda_python(args.env))
    job_id = slurm_submit(
        "dlc_pose",
        f"MPLBACKEND=Agg {python} -c {shlex.quote(script)}",
        ncpuspertask=2,
        gpus=1,
        walltime=args.walltime,
        partition=args.partition,
    )
    if job_id is None:
        raise RuntimeError("Slurm did not accept the DeepLabCut job")
    print(f"submitted job {job_id}; output in {out}")
