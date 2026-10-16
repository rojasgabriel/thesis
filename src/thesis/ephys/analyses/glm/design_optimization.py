"""Choose the form of each term of the V1 GLM by prediction.

Question: why does each term of the design have its form (the number of gap
classes, the length of the flash kernels, the fixation term and its end, the
basis after the center exit, the number of motion-energy PCs, the session
drift)? Whether the weights of the resulting design can be read is a separate
question (`model_trust.py`). This script makes no fit. It draws the results
of the seven design grids and writes the caption next to the figure.

Comparison: each grid scores the design in use (2026-10-15) and the same
design with one term in another form, on the same units of GRB006
(`--root`). A panel shows, for each other form, the validation D² of each
unit minus its D² with the form in use (paired). D² is relative to the
constant-rate model, on the validation trials.

- Panel a (`gap_class_grid.py`): one flash kernel and three or four gap
  classes, against the two classes in use.
- Panel b (`flash_length_grid.py`): flash kernels of 200, 250, and 300 ms
  against the 150 ms in use.
- Panel c (`hold_grid.py`): no fixation term, a step, a step with a kernel on
  the first flash, other forms of the kernel, no pre-stimulus column, and a
  weaker penalty on the kernel, against the log-spaced kernel in use.
- Panel d (`hold_window_grid.py`): the fixation kernel to 0.5 and 0.95 s,
  against 0.7 s (the 95th percentile of the fixation length).
- Panel e (`exit_grid.py`): the withdrawal kernel of the design of 2026-10-13
  in place of the response period, and the exit basis to 1.0 and 1.6 s or
  with 6 and 10 cosines, against 8 cosines to 1.3 s (the 95th percentile of
  the time from the exit to the response entry).
- Panel f (`pc_grid.py`): no motion energy and more PCs, against the 10 in
  use.
- Panel g (`drift_grid.py`): no session drift, more raised cosines, and a
  cubic spline, against the 4 raised cosines in use.

`withdrawal_grid.py` (kernels that start before the exit) ran on the design
of 2026-10-13 only, which had the withdrawal kernel. A kernel of ±300 ms
predicted better there but made the fixation level worse in parameter
recovery (`WITHDRAWAL_RECOVERY`), so the term at the exit starts at the exit;
the caption of panel e gives it.

Units: the sets of 20 units of the grids (`--samples 4` gives 80 units; no
unit is in two sets). The form of a term is chosen on these units; the
importance of a regressor is tested on all units (`inclusion.py`). Each point
is a unit, and the black line is the median. A point outside `DISPLAY_LIMIT`
is drawn at the limit; it is in the statistics, and the count is in
`design_optimization_stats.json`. Statistics: two-sided Wilcoxon signed-rank
test of the paired differences against 0.

Rule: another form replaces the one in use when it predicts reliably better
on these units and keeps the other kernels readable (parameter recovery).

    uv run python -m thesis.ephys.analyses.glm.design_optimization

Run the seven grids first; this script stops if a result file is missing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

from thesis.ephys.analyses.glm import (
    cli,
    drift_grid,
    exit_grid,
    flash_length_grid,
    gap_class_grid,
    hold_grid,
    hold_window_grid,
    pc_grid,
)
from thesis.ephys.analyses.glm.flash_by_period import SESSIONS
from thesis.ephys.analyses.glm.main_figure import DRIFT_COLOR, MOTION, TASK, VISUAL
from thesis.plotting import FIGURE_STYLE, REFERENCE_COLOR, label_panels, save_figure

# name of the grid: (its module, the parameterization in use, color, x label)
GRIDS = {
    "gap_class_grid": (
        gap_class_grid,
        gap_class_grid.IN_USE,
        VISUAL,
        "flash kernels by gap from the flash before (ms)",
    ),
    "flash_length_grid": (
        flash_length_grid,
        flash_length_grid.IN_USE,
        VISUAL,
        "length of the flash kernels (ms)",
    ),
    "hold_grid": (
        hold_grid,
        hold_grid.IN_USE,
        TASK,
        "parameterization of the fixation term",
    ),
    "hold_window_grid": (
        hold_window_grid,
        hold_window_grid.IN_USE,
        TASK,
        "end of the fixation kernel",
    ),
    "exit_grid": (
        exit_grid,
        exit_grid.IN_USE,
        TASK,
        "basis of the response period and turn direction",
    ),
    "pc_grid": (pc_grid, pc_grid.IN_USE, MOTION, "number of motion-energy PCs"),
    "drift_grid": (
        drift_grid,
        "4",
        DRIFT_COLOR,
        "session drift (number of raised cosines)",
    ),
}
# Tick labels that differ from the name of the design in its grid.
LABELS = {
    "40 | 80+": "40,\n80+",
    "40 | 80-120 | 160+": "40,\n80-120,\n160+",
    "40 | 80 | 120+": "40, 80,\n120+",
    "40 | 80 | 120 | 160+": "40, 80,\n120,\n160+",
    "no\nfixation": "no\npre-\nstimulus",
    "penalty\n÷ 100": "weaker\npenalty",
}
# Names of the other forms in the caption (keys as in the statistics).
CAPTION_NAMES = {
    "1 kernel": "one kernel for all flashes",
    "40 | 80-120 | 160+": "three classes (40, 80 to 120, 160+)",
    "40 | 80 | 120+": "three classes (40, 80, 120+)",
    "40 | 80 | 120 | 160+": "four classes",
    "200": "200 ms",
    "250": "250 ms",
    "300": "300 ms",
    "none": "no term",
    "step, entry": "a step from the entry",
    "step, flash": "a step from the first flash",
    "linear kernel": "linear spacing",
    "fine kernel": "the 12-cosine kernel of 2026-10-15",
    "no fixation": "no pre-stimulus column",
    "penalty ÷ 100": "a weaker penalty",
    "withdrawal 0–100": "the withdrawal kernel",
    "8": "8 cosines",
    "12": "12 cosines",
    "spline": "a spline",
}
DISPLAY_LIMIT = 0.006
# Fixation level, recovered over true (median unit), with the withdrawal
# kernel in use on 2026-10-13 and with the ±300 ms kernel (`parameter_recovery
# --grid withdrawal_grid --design ±300`, 2026-10-13; docs/glm.md). The design
# has no withdrawal kernel now, so the test cannot be run again on it.
WITHDRAWAL_RECOVERY = {"GRB006": (0.76, 0.55), "GRB064": (0.84, 0.73)}
CAPTION = """\
**The form of each term, chosen by prediction.** Each panel compares the
design in use (2026-10-15) with the same design where one term has another
form, on {units} units of {subject} ({sets} sets of 20 units; no unit is in
two sets). A point is one unit: its validation D² with that form minus its D²
with the one in use (open point at zero). The black line is the median. Marks
are two-sided Wilcoxon signed-rank tests of the differences against zero
(n.s., p ≥ 0.05; \\*, p < 0.05; \\*\\*, p < 0.01; \\*\\*\\*, p < 0.001).
**a**, Flash kernels by the gap from the flash before, against the two gap
classes in use. {gap_class_grid} **b**, Length of the flash kernels, against
150 ms. {flash_length_grid} **c**, Form of the fixation term, against the
log-spaced kernel in use. {hold_grid} **d**, End of the fixation kernel (10
cosines; after the end the kernel keeps its last value to the exit), against
0.7 s. {hold_window_grid} Past 0.7 s few trials give data, and in the design
of 2026-10-13 the kernel to 0.95 s was not stable after 700 ms in split
halves. **e**, The basis of the response period and turn direction, against
8 cosines to 1.3 s; "withdrawal 0–100" is the term at the exit of the design
of 2026-10-13 in place of the response period. {exit_grid} In that design, a
withdrawal kernel from 300 ms before to 300 ms after the exit predicted
better, but with it the fitted fixation level was recovered at
{level_under_test} of its true size, against {level_in_use} with the kernel
of 0 to 100 ms (median unit, {subject} and {held_out}; parameter recovery), so
the term at the exit starts at the exit. **f**, Number of motion-energy PCs.
{pc_sentence} **g**, Session drift, against 4 raised cosines. {drift_grid} The
importance of each regressor is tested on all units (main figure).
"""


def _pc_sentence(entry: dict) -> str:
    """Say what the grid of the number of PCs shows, from its statistics."""
    none = entry["0"]
    more = [entry[name] for name in entry if name.isdigit() and int(name) > 10]
    better = [e for e in more if e["wilcoxon_p"] < 0.05 and e["median_change"] > 0]
    worse = [e for e in more if e["wilcoxon_p"] < 0.05 and e["median_change"] < 0]
    first = (
        "No motion energy is worse."
        if none["wilcoxon_p"] < 0.05 and none["median_change"] < 0
        else "No motion energy is not reliably worse on these units."
    )
    second = (
        "More PCs are not better"
        if not better
        else f"{len(better)} of the larger numbers are reliably better"
    )
    return f"{first} {second}{', and some are worse' if worse else ''}."


def _and(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _others_sentence(entry: dict, labels: dict[str, str] = CAPTION_NAMES) -> str:
    """Say which other forms of a grid are reliably better or worse."""
    names = [n for n in entry if isinstance(entry[n], dict)]
    better = [
        labels.get(n, n)
        for n in names
        if entry[n]["wilcoxon_p"] < 0.05 and entry[n]["median_change"] > 0
    ]
    worse = [
        labels.get(n, n)
        for n in names
        if entry[n]["wilcoxon_p"] < 0.05 and entry[n]["median_change"] < 0
    ]
    parts = []
    if worse:
        parts.append(f"{_and(worse)} {'is' if len(worse) == 1 else 'are'} worse")
    if better:
        parts.append(f"{_and(better)} {'is' if len(better) == 1 else 'are'} better")
    if not parts:
        return "No other form is reliably different."
    rest = (
        ""
        if len(worse) + len(better) == len(names)
        else "; the others are not reliably different"
    )
    text = "; ".join(parts) + rest + "."
    return text[0].upper() + text[1:]


def _stars(p: float) -> str:
    return "n.s." if p >= 0.05 else "*" * (1 + (p < 0.01) + (p < 0.001))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=cli.PoissonGLM().root)
    args = parser.parse_args()
    stats = {}
    rng = np.random.default_rng(0)
    subjects = [session.split("_")[0] for session in SESSIONS]
    with plt.rc_context(FIGURE_STYLE):
        # The dpi is for the rasterized points; the text and the lines are vectors.
        figure = plt.figure(figsize=(7.4, 8.5), layout="constrained", dpi=300)
        axes = figure.subplots(4, 2)
        for axis, (grid, (module, in_use, color, xlabel)) in zip(
            axes.flat, GRIDS.items()
        ):
            path = args.root / grid / f"{grid}.json"
            if not path.exists():
                raise FileNotFoundError(f"{path}: run `{grid}` first.")
            scores = json.loads(path.read_text())
            units = sorted(scores[in_use])
            reference = np.asarray([scores[in_use][unit] for unit in units])
            entry = {"units": len(units), "in_use": in_use.replace("\n", " ")}
            # The designs of the grid module that have scores, in its order.
            names = [name for name in module.DESIGNS if name in scores]
            for position, name in enumerate(names):
                if name == in_use:
                    axis.plot(position, 0, "o", mfc="white", mec=color, ms=6)
                    continue
                change = np.asarray([scores[name][unit] for unit in units]) - reference
                p_value = float(wilcoxon(change).pvalue)
                shown = np.clip(change, -DISPLAY_LIMIT, DISPLAY_LIMIT)
                axis.scatter(
                    position + rng.uniform(-0.22, 0.22, len(shown)),
                    shown,
                    s=2.5,
                    color=color,
                    alpha=0.45,
                    lw=0,
                    rasterized=True,  # 80 points for each parameterization: keeps the file small
                )
                axis.hlines(
                    np.median(change), position - 0.32, position + 0.32, "black"
                )
                axis.text(
                    position,
                    DISPLAY_LIMIT * 1.04,
                    _stars(p_value),
                    ha="center",
                    fontsize=7,
                )
                entry[name.replace("\n", " ")] = {
                    "median_change": float(np.median(change)),
                    "median_change_percent": float(
                        100 * np.median(change) / np.median(reference)
                    ),
                    "units_better": int(np.sum(change > 0)),
                    "wilcoxon_p": p_value,
                    "units_outside_the_display_limit": int(
                        np.sum(np.abs(change) > DISPLAY_LIMIT)
                    ),
                }
            stats[grid] = entry
            axis.axhline(0, color=REFERENCE_COLOR, ls="--", lw=0.8)
            axis.set_ylim(-DISPLAY_LIMIT * 1.05, DISPLAY_LIMIT * 1.18)
            axis.set_xticks(
                range(len(names)),
                [LABELS.get(name, name) for name in names],
                fontsize=6,
            )
            axis.set_xlabel(xlabel, color=color, fontsize=7.5)
            if axis in axes[:, 0]:  # one y label for each row
                axis.set_ylabel(r"$D^2$ minus $D^2$" + "\nof the one in use")
            axis.tick_params(labelsize=7)
            axis.set_yticks([-0.005, 0, 0.005], ["−0.005", "0", "0.005"])
        for axis in axes.flat[len(GRIDS) :]:
            axis.set_axis_off()
        label_panels(axes.flat[: len(GRIDS)], x=-0.2, y=1.04)
        print(save_figure(figure, args.root / "design_optimization"))
    first = stats[next(iter(GRIDS))]
    (args.root / "design_optimization_stats.json").write_text(
        json.dumps(stats, indent=1)
    )
    (args.root / "design_optimization_caption.md").write_text(
        CAPTION.format(
            units=first["units"],
            sets=first["units"] // 20,
            pc_sentence=_pc_sentence(stats["pc_grid"]),
            **{
                grid: _others_sentence(stats[grid])
                for grid in GRIDS
                if grid != "pc_grid"
            },
            level_in_use=" and ".join(
                f"{WITHDRAWAL_RECOVERY[s][0]:.2f}" for s in subjects
            ),
            level_under_test=" and ".join(
                f"{WITHDRAWAL_RECOVERY[s][1]:.2f}" for s in subjects
            ),
            subject=subjects[0],
            held_out=subjects[1],
        )
    )
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
