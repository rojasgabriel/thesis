"""Plot the choice-history weights of each mouse.

Question: does the previous trial change the choice, and does that depend on
the reward? Read from the logistic regression of `psychophysics.py` (part
c), with no new fit.

Comparison: the weights of the flash rate (per Hz), the previous side after a
reward, and the previous side after no reward, with their 95% intervals, for
all recorded sessions (filled) and the analyzed session (open). A positive
side weight means the mouse repeats (stays on) the previous side; a negative
weight means it switches. No baseline, alignment or exclusion beyond those of
`psychophysics.py`; the sampling unit is the trial.

Writes `figures/behavior/history_weights.svg`.

    uv run python -m thesis.behavior.history_weights
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thesis.plotting import (
    FIGURE_STYLE,
    OBSERVED_COLOR,
    label_panels,
    mouse_marker,
    save_figure,
)

OUT = Path("results/behavior")
SUBJECTS = ("GRB006", "GRB064")
NAMES = (
    "flash rate\n(per Hz)",
    "previous side\nafter reward",
    "previous side\nafter no reward",
)


def main() -> None:
    every = {
        m["subject"]: m for m in json.loads((OUT / "psychophysics.json").read_text())
    }
    recorded = {
        m["subject"]: m
        for m in json.loads((OUT / "psychophysics_recorded.json").read_text())
    }
    with plt.rc_context(FIGURE_STYLE):
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
        for j, (a, subject) in enumerate(zip(ax, SUBJECTS, strict=True)):
            versions = (
                (every[subject], True, -0.12, "all recorded sessions"),
                (recorded[subject], False, 0.12, "analyzed session"),
            )
            for k, (d, filled, dx, label) in enumerate(versions):
                w = np.array(d["history_weights"])
                ci = np.array(d["history_weights_95"])
                a.errorbar(
                    np.arange(len(NAMES)) + dx,
                    w,
                    yerr=[w - ci[:, 0], ci[:, 1] - w],
                    fmt=mouse_marker(subject, j),
                    color=OBSERVED_COLOR,
                    mfc=OBSERVED_COLOR if filled else "white",
                    ms=7,
                    capsize=3,
                    ls="none",
                )
                a.text(
                    0.02,
                    0.04 + 0.08 * (1 - k),
                    f"{'●' if filled else '○'} {label}: {d['history_trials']} trials",
                    transform=a.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=7,
                )
            a.axhline(0, color="0.6", ls=":", lw=0.8)
            a.set_xticks(range(len(NAMES)), NAMES)
            a.set_xlim(-0.5, len(NAMES) - 0.5)
            a.text(0, 1.02, subject, transform=a.transAxes, va="bottom")
            for start, end, label, middle in (
                (0.15, 1.2, "stay", 0.65),
                (-0.15, -1.1, "switch", -0.6),
            ):
                a.annotate(
                    "",
                    xy=(2.62, end),
                    xytext=(2.62, start),
                    xycoords="data",
                    arrowprops=dict(arrowstyle="->", color="0.35", lw=0.8),
                    annotation_clip=False,
                )
                a.text(
                    2.68,
                    middle,
                    label,
                    rotation=90,
                    va="center",
                    fontsize=7,
                    color="0.35",
                    clip_on=False,
                )
        ax[0].set_ylabel("weight (log-odds of a right choice)")
        label_panels(ax, x=-0.15, y=1.04)
        fig.tight_layout()
        print(save_figure(fig, OUT / "history_weights"))


if __name__ == "__main__":
    main()
