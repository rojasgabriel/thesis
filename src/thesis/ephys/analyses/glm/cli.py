"""Command-line entry point and artifact orchestration for the V1 GLM.

`PoissonGLM` fixes the artifact layout for one session and drives the pipeline
stages in order: prepare (windows, motion-energy PCs, shared design), fit
(selection, cross-validation, refit, and the population figures), and unique
(per-regressor deviance, its figure, and the example-unit figures). `main` is
the `glm` console command.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from thesis.ephys.analyses.glm.dmat import (
    _common_design_is_current,
    prepare_common_design,
    write_stimulus_windows,
)
from thesis.ephys.analyses.glm.fit import crossvalidate, fit_models, refit_final

PIPELINE = """\
pipeline
  uv run glm prepare                 trial windows, motion-energy PCs, design
  uv run glm fit --units all         penalty, cross-validation, refit, summary
  uv run glm unique --units all      deviance per regressor, example figures

Artifacts go to figures/glm/<subject>_<session>/. Restart checkpoints stay in
the fit directory's checkpoints/ folder; figures stay in that fit directory.
Use --units sample for a 20-unit smoke run. The example unit is unit 197 if it
was fit, else the unit whose flash kernel explains the most deviance per second.
"""


@contextmanager
def _artifact_lock(root: Path):
    """Prevent two GLM commands from writing to one session at the same time."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".glm.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(
                f"Another GLM command is already running for {root}."
            ) from None
        yield


@dataclass
class PoissonGLM:
    """Default artifact layout and the commands that produce a fitted GLM."""

    subject: str = "GRB006"
    session: str = "20240821_121447"
    root: Path = Path("figures/glm/GRB006_20240821_121447")

    @property
    def windows(self) -> Path:
        return self.root / "stimulus_windows_me.npz"

    @property
    def video(self) -> Path:
        return self.root / "video_me_features.npz"

    @property
    def design(self) -> Path:
        return self.root / "common_design_me.npy"

    def fit_dir(self, units: str) -> Path:
        return self.root / f"{units}_fit_me"

    def prepare(self) -> None:
        from thesis.ephys.analyses.glm.video_svd import write_motion_energy_features

        if self.windows.exists():
            print(f"Reused prepared windows: {self.windows}", flush=True)
        else:
            write_stimulus_windows(
                self.subject, self.session, self.windows, self.root / "frame_times.npy"
            )
            print(f"Completed prepared windows: {self.windows}", flush=True)
        if self.video.exists():
            print(f"Reused motion-energy features: {self.video}", flush=True)
        else:
            write_motion_energy_features(self.windows, self.video)
            print(f"Completed motion-energy features: {self.video}", flush=True)
        if _common_design_is_current(self.design):
            print(f"Reused common design: {self.design}", flush=True)
        else:
            prepare_common_design(self.windows, self.video, self.design)
            print(f"Completed common design: {self.design}", flush=True)

    def fit(self, units: str = "sample", workers: int | None = None) -> None:
        """Select the PC count, cross-validate it, then refit on every trial.

        One command keeps the coefficients and comparisons in the same run.
        """
        directory = self.fit_dir(units)
        fit_models(self.windows, self.design, units, directory, workers)
        with (directory / "validation_summary.json").open() as handle:
            components = int(json.load(handle)["selected_video_components"])
        crossvalidate(self.windows, self.design, units, directory, components, workers)
        refit_final(self.windows, self.design, units, directory, components, workers)

    def unique(self, units: str = "sample", workers: int | None = None) -> None:
        from thesis.ephys.analyses.glm.unique_dev import run_unique

        run_unique(self.windows, self.design, self.fit_dir(units), units, workers)


def _unit_set(value: str) -> str:
    if not re.fullmatch(r"all|sample[1-9]?", value):
        raise argparse.ArgumentTypeError("use all, sample, or sample1 to sample9")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="glm",
        description=(
            "Poisson GLM for V1 spiking from flashes, task events, movement, "
            "and each unit's own spike history."
        ),
        epilog=PIPELINE,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="command")
    for name, help_text in (
        ("prepare", "Trial windows, motion-energy PCs, and the shared design"),
        ("fit", "Compare PC counts, cross-validate, refit, and plot"),
        ("unique", "Unique and maximal explained deviance per regressor"),
    ):
        command = subparsers.add_parser(
            name, help=help_text, description=help_text + "."
        )
        command.add_argument("--subject", default="GRB006", help="Subject name.")
        command.add_argument(
            "--session", default="20240821_121447", help="Session name."
        )
        command.add_argument(
            "--root",
            type=Path,
            help="Artifact directory. Defaults to figures/glm/<subject>_<session>.",
        )
        if name == "prepare":
            continue
        command.add_argument(
            "--units",
            type=_unit_set,
            default="sample",
            help=(
                "`sample` fits 20 sampled units and `all` fits every eligible "
                "one. `sample1`, `sample2`, ... fit 20 other units each."
            ),
        )
        command.add_argument(
            "--workers",
            type=int,
            help="Units to fit at once. Defaults to one less than the core count.",
        )
    args = parser.parse_args()
    model = PoissonGLM(
        subject=args.subject,
        session=args.session,
        root=args.root or Path("figures/glm") / f"{args.subject}_{args.session}",
    )
    with _artifact_lock(model.root):
        if args.command == "prepare":
            model.prepare()
            return
        getattr(model, args.command)(args.units, args.workers)


if __name__ == "__main__":
    main()
