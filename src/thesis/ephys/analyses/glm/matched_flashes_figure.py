"""Does V1 activity follow the center exit or the amount of body movement?

Question: V1 units fire more after the center exit, when the mouse moves. Is
that a change with the period of the trial, or does the rate go up with the
amount of movement?

Comparison: the saved rates of `matched_flashes.py`; no model and no new
computation. For each unit, the ratio of its rate between two matched groups
of flashes:

- the exit: flashes after the exit over flashes in the fixation, matched for
  the gap from the flash before;
- the amount of movement: among the flashes after the exit, the upper over
  the lower half of the motion energy of the video at the flash, matched for
  the time in the train, the gap, and the time from the exit.

- Measures: the rate before the response (`BASELINE_MS` around the flash) and
  the rate in the response window (`RESPONSE_MS` after it).
- Units: the GLM units (`fit.py`), pooled over mice; the flash-excited units
  (`flash_excited_units.json`) are the dark points. A unit with a rate of 0 in
  one of the two groups has no ratio; it is counted and left out.
- Intervals: hierarchical bootstrap (mice, then units), 95%, of the pooled
  median log ratio. The sampling unit is the unit; the units of a mouse share
  the flashes.
- Axis: ratios from 1/16 to 16 (log). A unit past that is drawn at the edge
  and counted in the report; it stays in the statistics.
- Sessions: the sessions of `SESSIONS` with a saved `matched_flashes.npz`.

The script writes `figures/glm/matched_flashes.svg` and
`results/glm/matched_flashes_figure.json`.

    uv run python -m thesis.ephys.analyses.glm.matched_flashes_figure
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thesis.ephys.analyses.glm import peaks
from thesis.ephys.analyses.glm.flash_kernels import SESSIONS
from thesis.ephys.analyses.glm.matched_flashes import (
    BASELINE_MS,
    RESPONSE_MS,
    _mean_rate,
)
from thesis.plotting import (
    CONCEPT_COLORS,
    FIGURE_STYLE,
    POINT_ALPHA,
    REFERENCE_COLOR,
    hierarchical_interval,
    label_panels,
    median_marker,
    mouse_marker,
    save_figure,
)

OUT = Path("results/glm")
# Row label: (saved comparison of `matched_flashes.py`, color of its concept).
ROWS = {
    r"$\dfrac{\mathrm{after\ the\ exit}}{\mathrm{in\ the\ port}}$": (
        "fixation against after exit, matched gap",
        CONCEPT_COLORS["task"],
    ),
    r"$\dfrac{\mathrm{more\ movement}}{\mathrm{less\ movement}}$"
    "\n(after the exit)": (
        "after exit, lower against upper half of motion energy",
        CONCEPT_COLORS["motion energy"],
    ),
}
MEASURES = {
    "rate before the flash": BASELINE_MS,
    "rate in the flash response": RESPONSE_MS,
}
LIMIT = 4.0  # log2 of the largest ratio on the axis
TICKS = {
    r"$\dfrac{1}{8}$": -3,
    r"$\dfrac{1}{4}$": -2,
    r"$\dfrac{1}{2}$": -1,
    "1": 0,
    "2": 1,
    "4": 2,
    "8": 3,
}


def log_ratios(session: str) -> tuple[dict, np.ndarray]:
    """Return the log2 ratio of each unit per (row, measure), and which units are flash-excited."""
    out = {}
    with np.load(OUT / session / "matched_flashes.npz") as saved:
        for row, (name, _) in ROWS.items():
            for measure, window in MEASURES.items():
                first, second = _mean_rate(saved[f"rate__{name}"], window)
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = np.log2(second / first)
                out[(row, measure)] = np.where(np.isfinite(ratio), ratio, np.nan)
        excited = np.isin(saved["unit_ids"], peaks.load_excited_units()[session])
    return out, excited


def main() -> None:
    sessions = [s for s in SESSIONS if (OUT / s / "matched_flashes.npz").exists()]
    data = {s: log_ratios(s) for s in sessions}
    rng = np.random.default_rng(0)
    report = {}
    with plt.rc_context(FIGURE_STYLE):
        figure, axes = plt.subplots(1, len(MEASURES), figsize=(8.6, 2.8), sharey=True)
        for axis, measure in zip(axes, MEASURES, strict=True):
            for y, (row, (name, color)) in enumerate(ROWS.items()):
                groups, excited_groups = [], []
                for index, session in enumerate(sessions):
                    values, excited = data[session]
                    ratio = values[(row, measure)]
                    used = np.isfinite(ratio)
                    groups.append(ratio[used])
                    excited_groups.append(ratio[used & excited])
                    marker = mouse_marker(session.split("_")[0], index)
                    for mask, alpha in (
                        (used & ~excited, POINT_ALPHA),
                        (used & excited, 0.9),
                    ):
                        axis.plot(
                            np.clip(ratio[mask], -LIMIT, LIMIT),
                            -y + rng.uniform(-0.25, 0.25, mask.sum()),
                            marker,
                            ms=2.8,
                            color=color,
                            alpha=alpha,
                            mew=0,
                        )
                    median_marker(
                        axis,
                        np.median(ratio[used]),
                        -y + (0.1 if index % 2 else -0.1),
                        marker,
                        color,
                    )
                stat = hierarchical_interval(groups)
                low, high = stat["interval_95"]
                axis.errorbar(
                    stat["median"],
                    -y + 0.4,
                    xerr=[[stat["median"] - low], [high - stat["median"]]],
                    fmt="|",
                    color="black",
                    capsize=2,
                    ms=7,
                )
                axis.text(
                    LIMIT,
                    -y + 0.4,
                    rf"$\times${2 ** stat['median']:.2f} [{2**low:.2f}, {2**high:.2f}]",
                    color=color,
                    ha="right",
                    va="center",
                    fontsize=7,
                )
                report[f"{measure}; {name}"] = {
                    "units": {
                        s.split("_")[0]: len(g)
                        for s, g in zip(sessions, groups, strict=True)
                    },
                    "units_without_ratio": int(
                        sum(
                            np.isnan(data[s][0][(row, measure)]).sum() for s in sessions
                        )
                    ),
                    "units_past_the_axis": int(
                        sum(np.sum(np.abs(g) > LIMIT) for g in groups)
                    ),
                    "all units, log2 ratio": stat,
                    "flash-excited units, log2 ratio": hierarchical_interval(
                        excited_groups
                    ),
                }
            axis.axvline(0, color=REFERENCE_COLOR, ls="--", lw=0.8)
            axis.set_xlim(-LIMIT, LIMIT)
            axis.set_xticks(list(TICKS.values()), list(TICKS))
            axis.set_xlabel(f"fold change of the {measure}")
        axes[0].set_yticks([0, -1], list(ROWS))
        for label, (_, color) in zip(
            axes[0].get_yticklabels(), ROWS.values(), strict=True
        ):
            label.set_color(color)
        axes[0].set_ylim(-1.5, 0.65)
        n_units = sum(len(v[1]) for v in data.values())
        axes[1].text(
            1.0,
            1.02,
            f"{n_units} units, {len(sessions)} mice; dark: flash-excited units",
            transform=axes[1].transAxes,
            ha="right",
            va="bottom",
            fontsize=7,
            color="0.4",
        )
        figure.tight_layout()
        label_panels(axes, x=-0.05, y=1.04)
        output = save_figure(figure, OUT / "matched_flashes")
    (OUT / "matched_flashes_figure.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    print(output)


if __name__ == "__main__":
    main()
