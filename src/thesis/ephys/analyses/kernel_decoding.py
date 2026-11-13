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

from thesis.ephys.analyses.decoding_time_course import COLORS, SESSIONS
from thesis.plotting import (
    CONCEPT_COLORS,
    FIGURE_STYLE,
    OBSERVED_COLOR,
    REFERENCE_COLOR,
    label_panels,
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
        figure, axes = plt.subplots(
            2,
            len(subjects),
            figsize=(4.5 * len(subjects), 4.8),
            sharex=True,
            sharey="row",
            height_ratios=(1, 1.2),
            layout="constrained",
        )
        for column, subject in enumerate(subjects):
            kernel, decoding = axes[:, column]
            data = kernels[subject]
            x = np.asarray(data["kernel_bin_centers_ms"]) / 1000
            w = np.asarray(data["kernel_weights"])
            ci = np.asarray(data["kernel_weights_95"]).T
            kernel.axhline(0, color=REFERENCE_COLOR, lw=0.6)
            kernel.errorbar(
                x,
                w,
                yerr=[w - ci[0], ci[1] - w],
                fmt="o-",
                color=OBSERVED_COLOR,
                mfc="white",
                ms=4,
                capsize=2,
                lw=1,
            )
            kernel.set_title(
                f"{subject} (kernel p = {data['kernel_flat_lr_p']:.2g} against flat)",
                fontsize=8,
            )
            part = courses[subject]["first_flash"]
            centers = np.asarray(part["centers_s"])
            nulls = np.stack(
                [np.asarray(part[label]["shuffle_2.5_50_97.5"]) for label in COLORS]
            )
            decoding.fill_between(
                centers,
                nulls[:, :, 0].min(0),
                nulls[:, :, 2].max(0),
                color="0.88",
                lw=0,
            )
            for label, color in COLORS.items():
                mean = np.asarray(part[label]["accuracy"])
                sem = np.asarray(part[label]["sem_across_folds"])
                decoding.fill_between(
                    centers, mean - sem, mean + sem, color=color, alpha=0.25, lw=0
                )
                decoding.plot(centers, mean, color=color, lw=1.6)
            decoding.axhline(0.5, color=REFERENCE_COLOR, ls=":", lw=0.8)
            exit_s = courses[subject]["median_exit_after_first_flash_s"]
            for axis in (kernel, decoding):
                axis.axvline(exit_s, color=CONCEPT_COLORS["task"], ls="--", lw=0.8)
            kernel.text(
                exit_s - 0.02,
                0.97,
                "median exit",
                color=CONCEPT_COLORS["task"],
                fontsize=7,
                ha="right",
                va="top",
                transform=kernel.get_xaxis_transform(),
            )
            decoding.set_xlabel("time from the first flash (s)")
        for label, y in (("category", 0.96), ("choice", 0.86)):
            axes[1, 0].text(
                0.03,
                y,
                label,
                color=COLORS[label],
                va="top",
                transform=axes[1, 0].transAxes,
            )
        axes[1, -1].text(
            0.97,
            0.02,
            "shuffle",
            color="0.5",
            ha="right",
            transform=axes[1, -1].transAxes,
        )
        axes[0, 0].set_ylabel("behavior: weight on a\nright choice (log-odds/flash)")
        axes[1, 0].set_ylabel("V1: held-out accuracy")
        axes[1, 0].set_xlim(-0.1, 1.0)
        label_panels(axes.ravel(), x=-0.22, y=1.04)
        print(save_figure(figure, OUT / "kernel_decoding"))


if __name__ == "__main__":
    main()
