"""Draw the tests of the form of each term of the V1 GLM in one figure.

Question: why does each term of the design have its form (the number of gap
classes, the length of the flash kernels, the form of the fixation term, the
window of the withdrawal kernel, the number of motion-energy PCs, the form of
the session drift)? This script makes no fit. It draws the results of the six
design grids and writes the caption next to the figure.

Comparison: each grid scores the design in use and the same design with one
term in another form, on the same units. A panel shows, for each other form,
the validation D² of each unit minus its D² with the form in use (paired). D²
is relative to the constant-rate model, on the validation trials.

- Panel a (`gap_class_grid.py`): one flash kernel, the earlier gap terms, and
  three or four gap classes, against the two classes in use.
- Panel b (`flash_length_grid.py`): flash kernels of 200, 250, and 300 ms
  against the 150 ms in use.
- Panel c (`hold_grid.py`): no fixation term, a step, a step with a kernel on
  the first flash, and other forms of the kernel, against the log-spaced
  kernel in use.
- Panel d (`withdrawal_grid.py`): no withdrawal kernel and kernels that also
  cover the time before the exit, against the kernel of 0 to 100 ms in use.
- Panel e (`pc_grid.py`): no motion energy and more PCs, against the 10 in
  use.
- Panel f (`drift_grid.py`): no session drift, more raised cosines, and a
  cubic spline, against the 4 raised cosines in use.

Units: the sets of 20 units of the grids (`--samples 4` gives 80 units; no
unit is in two sets). The form of a term is chosen on these units; the
importance of a regressor is tested on all units (`inclusion.py`). Each point
is a unit, and the black line is the median. A point outside `DISPLAY_LIMIT`
is drawn at the limit; it is in the statistics, and the count is in
`design_optimization_stats.json`. The number under each form is the largest
direction that the term of the panel, in that form, shares with another task
regressor: the largest first canonical correlation of a pair of regressors
with the term in it (`collinearity.shared_directions`; the pair is in the
stats file). A form with no such term has no number. It is in red when it is
above the largest value of all pairs of the design in use (`SHARED_TOLERANCE`).

Rule for the form of a term: another form replaces the form in use when it
(1) predicts reliably better on these units, and (2) is separable: the term in that form does not share a direction with
another regressor that is larger than the largest one of the design in use. Condition 2 keeps the kernels readable: a form that predicts
better because it covers the time of another term fails it. The mark over a form is a two-sided Wilcoxon
signed-rank test of the differences against zero. Trials, alignment,
baseline, and exclusions are those of `fit.py`. The sampling unit is the
unit.

    uv run python -m thesis.ephys.analyses.glm.design_optimization

Run the four grids first; this script stops if a result file is missing.
"""

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
    dmat,
    drift_grid,
    flash_length_grid,
    gap_class_grid,
    hold_grid,
    pc_grid,
    withdrawal_grid,
)
from thesis.ephys.analyses.glm.collinearity import shared_directions
from thesis.ephys.analyses.glm.main_figure import DRIFT_COLOR, MOTION, TASK, VISUAL
from thesis.plotting import FIGURE_STYLE, REFERENCE_COLOR, label_panels, save_figure

# name of the grid: (its module, the form in use, color, x label, and the starts
# of the names of the regressors of that term in the design)
GRIDS = {
    "gap_class_grid": (
        gap_class_grid,
        gap_class_grid.IN_USE,
        VISUAL,
        "flash kernels by gap from the flash before (ms)",
        (dmat.FLASH_GAP, dmat.FLASH_COMMON),
    ),
    "flash_length_grid": (
        flash_length_grid,
        flash_length_grid.IN_USE,
        VISUAL,
        "length of the flash kernels (ms)",
        ("flash", dmat.FLASH_COMMON),
    ),
    "hold_grid": (
        hold_grid,
        hold_grid.IN_USE,
        TASK,
        "form of the fixation term",
        (dmat.HOLD, dmat.IN_PORT),
    ),
    "withdrawal_grid": (
        withdrawal_grid,
        withdrawal_grid.IN_USE,
        TASK,
        "withdrawal kernel (ms from the center exit)",
        ("withdrawal",),
    ),
    "pc_grid": (pc_grid, pc_grid.IN_USE, MOTION, "number of motion-energy PCs", ()),
    "drift_grid": (
        drift_grid,
        "4",
        DRIFT_COLOR,
        "session drift (number of raised cosines)",
        (dmat.DRIFT,),
    ),
}
# Tick labels that differ from the name of the design in its grid.
LABELS = {
    "40 | 80+": "40,\n80+",
    "40 | 80-120 | 160+": "40,\n80-120,\n160+",
    "40 | 80 | 120+": "40, 80,\n120+",
    "40 | 80 | 120 | 160+": "40, 80,\n120,\n160+",
    "no\nfixation": "no\ninitiation\ncolumn",
    "penalty\n÷ 100": "weaker\npenalty",
}
DISPLAY_LIMIT = 0.006
# A form is separable when its largest shared direction is not above that of
# the design in use, with this tolerance.
SHARED_TOLERANCE = 0.05
CAPTION = """\
**The form of each term of the design.** Each panel compares the design in use
with the same design where one term has another form, on {units} units ({sets}
sets of 20 units; no unit is in two sets). A point is one unit: its validation
D² with that form minus its D² with the form in use (open point at zero). The
black line is the median. Marks are two-sided Wilcoxon signed-rank tests of
the differences against zero (n.s., p ≥ 0.05; \\*, p < 0.05; \\*\\*, p < 0.01;
\\*\\*\\*, p < 0.001). **a**, Flash kernels by the gap from the flash before.
One kernel for all flashes is worse than the two gap classes in use; more
classes are not better. **b**, Length of the flash kernels. Longer kernels are
not better than 150 ms. **c**, Form of the fixation term. No term and a step
are worse than the kernel in use. **d**, Withdrawal kernel. No kernel is worse.
A kernel from 300 ms before to 300 ms after the exit predicts better, but it
is not separable: it shares a direction of {wide_shared:.2f} with the fixation
kernel, against {limit:.2f} for the largest one in the design in use. **e**,
Number of motion-energy PCs. {pc_sentence} **f**, Session drift. No drift is
worse; more functions are not better. The number under each form is the
largest first canonical correlation of the term of the panel, in that form,
with another task regressor (red: above the largest value of the design in
use). A form replaces the form in use when it
predicts reliably better and is separable; no form does both. The importance of each regressor is tested on all units
(main figure).
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


def shared_direction(
    settings: dict, terms: tuple[str, ...], prepared: dict, trials
) -> tuple[float, str, str] | None:
    """Return the largest direction that a term of one design shares with another.

    `terms` are the starts of the names of the regressors of the term (all
    regressors for an empty tuple). The value is the largest first canonical
    correlation of a pair of task regressors with one of them in the pair,
    with the names of the pair. None when the design has no such regressor.
    """
    saved = {name: getattr(dmat, name) for name in settings}
    for name, value in settings.items():
        setattr(dmat, name, value)
    try:
        values, manifest = dmat.build_task_design(prepared["alignments"], trials)
    finally:
        for name, value in saved.items():
            setattr(dmat, name, value)
    groups, cursor = {}, 0
    for item in manifest:
        groups[item["name"]] = slice(cursor, cursor + item["columns"])
        cursor += item["columns"]
    columns = values[dmat._valid_bin_mask(prepared)]
    columns = (columns - columns.mean(0)) / columns.std(0)
    pairs = shared_directions(columns.T @ columns / len(columns), groups)
    names = list(groups)
    own = np.asarray([name.startswith(terms) or not terms for name in names])
    if not own.any():
        return None
    pairs = np.where(own[:, None] | own[None, :], pairs, np.nan)
    first, second = np.unravel_index(np.nanargmax(pairs), pairs.shape)
    return float(pairs[first, second]), names[first], names[second]


def _stars(p: float) -> str:
    return "n.s." if p >= 0.05 else "*" * (1 + (p < 0.01) + (p < 0.001))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=cli.PoissonGLM().root)
    args = parser.parse_args()
    stats = {}
    rng = np.random.default_rng(0)
    prepared = dmat._load_windows(cli.PoissonGLM(root=args.root).windows)
    trials = dmat._load_selected_trials(prepared)
    limit = shared_direction({}, (), prepared, trials)[0] + SHARED_TOLERANCE
    stats["largest_shared_direction_of_the_design_in_use"] = limit - SHARED_TOLERANCE
    with plt.rc_context(FIGURE_STYLE):
        figure, axes = plt.subplots(3, 2, figsize=(9.5, 9.6), layout="constrained")
        for axis, (grid, (module, in_use, color, xlabel, terms)) in zip(
            axes.flat, GRIDS.items()
        ):
            path = args.root / grid / f"{grid}.json"
            if not path.exists():
                raise FileNotFoundError(f"{path}: run `{grid}` first.")
            scores = json.loads(path.read_text())
            units = sorted(scores[in_use])
            reference = np.asarray([scores[in_use][unit] for unit in units])
            entry = {"units": len(units), "form_in_use": in_use.replace("\n", " ")}
            # The designs of the grid module that have scores, in its order.
            names = [name for name in module.DESIGNS if name in scores]
            shared_of = {}
            for position, name in enumerate(names):
                # The task columns are the same for each number of PCs.
                pair = (
                    shared_direction(module.DESIGNS[name], terms, prepared, trials)
                    if isinstance(module.DESIGNS, dict)
                    else None
                )
                if pair:
                    axis.text(
                        position,
                        -DISPLAY_LIMIT * 1.2,
                        f"{pair[0]:.2f}",
                        ha="center",
                        fontsize=6,
                        color="C3" if pair[0] > limit else "0.4",
                        fontweight="bold" if pair[0] > limit else "normal",
                    )
                    shared_of[name.replace("\n", " ")] = {
                        "largest_shared_direction": pair[0],
                        "between": list(pair[1:]),
                        "separable": bool(pair[0] <= limit),
                    }
                if name == in_use:
                    axis.plot(position, 0, "o", mfc="white", mec=color, ms=6)
                    continue
                change = np.asarray([scores[name][unit] for unit in units]) - reference
                p_value = float(wilcoxon(change).pvalue)
                shown = np.clip(change, -DISPLAY_LIMIT, DISPLAY_LIMIT)
                axis.scatter(
                    position + rng.uniform(-0.22, 0.22, len(shown)),
                    shown,
                    s=5,
                    color=color,
                    alpha=0.45,
                    lw=0,
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
            entry["shared_direction_of_the_term"] = shared_of
            stats[grid] = entry
            axis.axhline(0, color=REFERENCE_COLOR, ls="--", lw=0.8)
            axis.set_ylim(-DISPLAY_LIMIT * 1.3, DISPLAY_LIMIT * 1.18)
            axis.set_xticks(
                range(len(names)),
                [LABELS.get(name, name) for name in names],
                fontsize=6.5,
            )
            axis.set_xlabel(xlabel, color=color)
            axis.set_ylabel(r"$D^2$ minus $D^2$ of the form in use")
        label_panels(axes.flat, x=-0.16, y=1.04)
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
            wide_shared=stats["withdrawal_grid"]["shared_direction_of_the_term"][
                "±300"
            ]["largest_shared_direction"],
            limit=stats["largest_shared_direction_of_the_design_in_use"],
        )
    )
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
