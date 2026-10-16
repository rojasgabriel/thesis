"""Choose the form of each term of the V1 GLM by prediction.

Question: why does each term of the design have its form (the number of gap
classes, the length of the flash kernels, the fixation term, the window of
the withdrawal kernel, the number of motion-energy PCs, the session drift)?
Whether the weights of the resulting design can be read is a separate
question (`model_trust.py`). This script makes no fit. It draws the results
of the six design grids and writes the caption next to the figure.

Comparison: each grid scores the design in use and the same design with one
term in another form, on the same units of GRB006 (`--root`). A panel shows,
for each other form, the validation D² of each unit minus its D² with the
form in use (paired). D² is relative to the constant-rate model, on the
validation trials.

- Panel a (`gap_class_grid.py`): one flash kernel, the earlier gap terms, and
  three or four gap classes, against the two classes in use.
- Panel b (`flash_length_grid.py`): flash kernels of 200, 250, and 300 ms
  against the 150 ms in use.
- Panel c (`hold_grid.py`): no fixation term, a step, a step with a kernel on
  the first flash, and other forms of the kernel, against the log-spaced
  kernel in use.
- Panel d (`withdrawal_grid.py`): no withdrawal kernel and kernels that also
  cover the time before the exit, against the kernel of 0 to 100 ms in use.
  A form that predicts better also has to keep the other kernels readable:
  the ±300 ms kernel is tested for that with `parameter_recovery.py`.
- Panel e (`pc_grid.py`): no motion energy and more PCs, against the 10 in
  use.
- Panel f (`drift_grid.py`): no session drift, more raised cosines, and a
  cubic spline, against the 4 raised cosines in use.

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

Run the six grids and `parameter_recovery` first; this script stops if a
result file is missing."""

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
    flash_length_grid,
    gap_class_grid,
    hold_grid,
    pc_grid,
    withdrawal_grid,
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
    "withdrawal_grid": (
        withdrawal_grid,
        withdrawal_grid.IN_USE,
        TASK,
        "withdrawal kernel (ms from the center exit)",
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
    "no\nfixation": "no\npre-stimulus",
    "penalty\n÷ 100": "weaker\npenalty",
}
DISPLAY_LIMIT = 0.006
CAPTION = """\
**The form of each term, chosen by prediction.** Each panel compares the
design in use with the same design where one term has another form, on
{units} units of {subject} ({sets} sets of 20 units; no unit is in two sets). A
point is one unit: its validation D² with that form minus its D² with the one
in use (open point at zero). The black line is the median. Marks are
two-sided Wilcoxon signed-rank tests of the differences against zero (n.s.,
p ≥ 0.05; \\*, p < 0.05; \\*\\*, p < 0.01; \\*\\*\\*, p < 0.001). **a**,
Flash kernels by the gap from the flash before. One kernel for all flashes is
worse than the two gap classes in use; more classes are not better. **b**,
Length of the flash kernels. Longer kernels are not better than 150 ms.
**c**, Form of the fixation term. No term and a step are worse than the
kernel in use. **d**, Withdrawal kernel. No kernel is worse. A kernel from
300 ms before to 300 ms after the exit predicts better, but with it the
fitted fixation level is recovered at {level_under_test} of its true size,
against {level_in_use} with the kernel in use (median unit, {subject} and
{held_out}; parameter recovery, as in the figure of `model_trust.py`), so the
design keeps the kernel of 0 to 100 ms. **e**, Number of motion-energy PCs.
{pc_sentence} **f**, Session drift. No drift is worse; more functions are not
better. The importance of each regressor is tested on all units (main
figure).
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
        figure = plt.figure(figsize=(7.4, 6.4), layout="constrained", dpi=300)
        axes = figure.subplots(3, 2)
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
        label_panels(axes.flat, x=-0.2, y=1.04)
        print(save_figure(figure, args.root / "design_optimization"))
    first = stats[next(iter(GRIDS))]
    # The test of the withdrawal kernel of ±300 ms, from `parameter_recovery`.
    recovery = json.loads((args.root.parent / "parameter_recovery.json").read_text())
    (args.root / "design_optimization_stats.json").write_text(
        json.dumps(stats, indent=1)
    )
    (args.root / "design_optimization_caption.md").write_text(
        CAPTION.format(
            units=first["units"],
            sets=first["units"] // 20,
            pc_sentence=_pc_sentence(stats["pc_grid"]),
            level_in_use=" and ".join(
                f"{recovery[s]['fixation level, recovered over true (median)']['in_use']:.2f}"
                for s in SESSIONS
            ),
            level_under_test=" and ".join(
                f"{recovery[s]['fixation level, recovered over true (median)']['under_test']:.2f}"
                for s in SESSIONS
            ),
            subject=subjects[0],
            held_out=subjects[1],
        )
    )
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
