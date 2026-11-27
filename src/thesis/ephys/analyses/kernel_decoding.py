"""Does the V1 category or choice signal follow the weight of the flashes on the choice?

Question: the category decoded from V1 is clearest after the center exit
(`decoding_time_course.py`), and in GRB064 the flashes after the exit weigh
most on the choice (`behavior/psychophysics.py --recorded`). Do the decoding time courses
line up in time with the psychophysical kernel?

Comparison: for each mouse, on one time axis from the first flash, the kernel
(top) and the held-out category and choice accuracy (bottom). No new fit: both
are read from the saved results of those two modules, which use the same
sessions (`decoding_time_course.SESSIONS`).

- Kernel: log-odds of a right choice per flash in 160-ms bins (plotted at the
  bin center), 95% bootstrap interval over trials.
- Decoding: 200-ms windows plotted at their center, so a point holds 100 ms on
  each side; ± SEM across folds; gray band: shuffle 2.5–97.5 percentiles of
  both decoders. The dashed line is the median exit of the decoder trials.
- Each mouse is shown alone; the y axes match across mice.

The script writes `figures/kernel_decoding/kernel_decoding.svg`. Run the two
source modules first.

    uv run python -m thesis.ephys.analyses.kernel_decoding
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thesis.ephys.analyses.decoding_time_course import COLORS, SESSIONS, draw_courses
from thesis.plotting import (
    CONCEPT_COLORS,
    FIGURE_STYLE,
    REFERENCE_COLOR,
    label_panels,
    mouse_lines,
    save_figure,
)

OUT = Path("results/kernel_decoding")
KERNEL = Path("results/behavior/psychophysics_recorded.json")
DECODING = Path("results/decoding_time_course/decoding_time_course.json")


def main() -> None:
    kernels = {item["subject"]: item for item in json.loads(KERNEL.read_text())}
    courses = {item["subject"]: item for item in json.loads(DECODING.read_text())}
    subjects = [session.split("_")[0] for session in SESSIONS]
    with plt.rc_context(FIGURE_STYLE):
        figure, (kernel, decoding) = plt.subplots(
            2,
            1,
            figsize=(4.8, 5.2),
            sharex=True,
            height_ratios=(1, 1.2),
            layout="constrained",
        )
        # Behavior: a gray kernel for each mouse and the black mean of the mice.
        kernel.axhline(0, color=REFERENCE_COLOR, lw=0.6)
        mouse_lines(
            kernel,
            np.asarray(kernels[subjects[0]]["kernel_bin_centers_ms"]) / 1000,
            [kernels[s]["kernel_weights"] for s in subjects],
            subjects,
        )
        # V1: a thin line for each mouse and the mean ± SEM of the mice.
        draw_courses(decoding, [courses[s]["first_flash"] for s in subjects])
        exits = [courses[s]["median_exit_after_first_flash_s"] for s in subjects]
        for axis in (kernel, decoding):
            for exit_s in exits:
                axis.axvline(exit_s, color=CONCEPT_COLORS["task"], ls="--", lw=0.8)
        kernel.text(
            min(exits) - 0.02,
            0.97,
            "median exit of each mouse",
            color=CONCEPT_COLORS["task"],
            fontsize=7,
            ha="right",
            va="top",
            transform=kernel.get_xaxis_transform(),
        )
        decoding.set_xlabel("time from the first flash (s)")
        for label, y in (("category", 0.96), ("choice", 0.86)):
            decoding.text(
                0.03,
                y,
                label,
                color=COLORS[label],
                va="top",
                transform=decoding.transAxes,
            )
        decoding.text(
            0.97, 0.02, "shuffle", color="0.5", ha="right", transform=decoding.transAxes
        )
        kernel.set_ylabel("behavior: weight on a\nright choice (log-odds/flash)")
        decoding.set_ylabel("V1: held-out accuracy")
        decoding.set_xlim(-0.1, 1.0)
        label_panels([kernel, decoding], x=-0.22, y=1.04)
        print(save_figure(figure, OUT / "kernel_decoding"))


if __name__ == "__main__":
    main()
