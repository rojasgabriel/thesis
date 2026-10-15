"""Draw the design optimization and the checks of the V1 GLM in one figure.

Question: can the model and its weights be trusted? Why does each term of the
design have its parameterization (the number of gap classes, the length of
the flash kernels, the fixation term, the window of the withdrawal kernel, the
number of motion-energy PCs, the session drift); are the regressors of the
resulting design collinear; does a fit give back known weights and the same
weights from two halves of the trials; and how much of the flash × move effect
does the fit make with no such effect in the data? This script makes no fit.
It draws the results of the six design grids, of `collinearity.py`,
`parameter_recovery.py`, `kernel_reliability.py`, and `leak_test.py`, and
writes the caption next to the figure.

The work has two steps. Step 1 (panels a to f): the parameterization of each
term is chosen by prediction, on GRB006 (`--root`). Step 2 (panels g to k):
the collinearity of the regressors of the resulting design (g to i) and the
parameter recovery of each regressor (j and k), because the weights can be
read only when the model gives them back. Step 2 is shown in GRB006 and in
GRB064 (the other session of `SESSIONS`), which took no part in step 1. In
step 2 the units of both mice are pooled; the black marker is the pooled
median with its hierarchical-bootstrap 95% interval (mice, then units), and
the open circles are the median of each mouse. Step 3 (panels l to o): the
stability of the weights over trials (l) and the leak of the fixation level
into the flash × move kernels (m to o), in the same two mice.

Comparison in step 1: each grid scores the design in use and the same design
with one term in another parameterization, on the same units. A panel shows,
for each other parameterization, the validation D² of each unit minus its D²
with the parameterization in use (paired). D² is relative to the
constant-rate model, on the validation trials.

- Panel a (`gap_class_grid.py`): one flash kernel, the earlier gap terms, and
  three or four gap classes, against the two classes in use.
- Panel b (`flash_length_grid.py`): flash kernels of 200, 250, and 300 ms
  against the 150 ms in use.
- Panel c (`hold_grid.py`): no fixation term, a step, a step with a kernel on
  the first flash, and other parameterizations of the kernel, against the log-spaced
  kernel in use.
- Panel d (`withdrawal_grid.py`): no withdrawal kernel and kernels that also
  cover the time before the exit, against the kernel of 0 to 100 ms in use.
- Panel e (`pc_grid.py`): no motion energy and more PCs, against the 10 in
  use.
- Panel f (`drift_grid.py`): no session drift, more raised cosines, and a
  cubic spline, against the 4 raised cosines in use.

Units: the sets of 20 units of the grids (`--samples 4` gives 80 units; no
unit is in two sets). The parameterization of a term is chosen on these units; the
importance of a regressor is tested on all units (`inclusion.py`). Each point
is a unit, and the black line is the median. A point outside `DISPLAY_LIMIT`
is drawn at the limit; it is in the statistics, and the count is in
`design_optimization_stats.json`.

- Panels g to i (`collinearity.py`): the collinearity of the design in use,
  for the design of each unit. g, the generalized variance inflation factor
  of each regressor (points are units, GRB006 on the left and GRB064 on the
  right of each regressor; the line is the median; dashed lines at 5 and 10).
  h, the condition indices of each unit, sorted (dashed lines at 10 and 30).
  i, the first canonical correlation of each pair of regressors (lower
  triangle GRB006, upper triangle GRB064); for a pair with spike history, the
  median over units. It shows which regressors share a direction; it has no
  published limit.
- Panels j and k (`parameter_recovery.py`): the recovery of each regressor
  of the design in use. Spikes are simulated from the final refit of each
  unit, the model is fit to them, and each recovered kernel is compared with
  the true one. j, the agreement, 1 - |a - b|² / (|a|² + |b|²) (1 for equal
  kernels); k, the size of the recovered kernel over the true size (1 is the
  true size). Points are units, as in g; the line is the median. A point
  outside `RECOVERY_LIMITS` is drawn at the limit; it is in the statistics.

- Panel l (`kernel_reliability.py`): each unit is fit on the even and on the
  odd trials, with the penalty of its final refit. The agreement of the two
  kernels of each regressor, as in j, for the units that need the regressor
  (unique ΔD² above 0 over the folds). A point below the axis is drawn at the
  limit; the count is in the output.
- Panels m to o (`leak_test.py`): spikes are simulated from the final refit
  of each unit with the flash × move weights set to 0, and the full model is
  fit to them. The leak is the flash × move part of the move − still peak of
  that fit (the matched peak measure of `peaks.py`, as in
  `flash_by_period.py`; the excited units). m, the real part against the
  leak; n, real minus leak (two-sided Wilcoxon signed-rank test in each
  mouse); o, the leak against the fixation level of the unit. A point outside
  `PART_LIMITS` is drawn at the limit; it is in the statistics.

Rule for the parameterization of a term: another parameterization replaces
the one in use when it predicts reliably better on these units. Published
limits for step 2: a generalized variance inflation factor (as GVIF^(1/p),
Fox and Monette 1992) above 5, or a condition index above 30 (Belsley, Kuh,
and Welsch 1980).

    uv run python -m thesis.ephys.analyses.glm.design_optimization

Run the six grids, `collinearity`, `parameter_recovery`, `kernel_reliability
--refit`, and `leak_test --refit` first; this script stops if a result file is
missing.
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
    drift_grid,
    flash_length_grid,
    gap_class_grid,
    hold_grid,
    pc_grid,
    peaks,
    withdrawal_grid,
)
from thesis.ephys.analyses.glm.collinearity import (
    INDEX_LIMITS,
    VIF_LIMITS,
)
from thesis.ephys.analyses.glm.flash_by_period import SESSIONS
from thesis.ephys.analyses.glm.leak_test import PART_LIMITS
from thesis.ephys.analyses.glm.main_figure import (
    DRIFT_COLOR,
    MOTION,
    REGRESSORS,
    TASK,
    VISUAL,
)
from thesis.plotting import (
    FIGURE_STYLE,
    REFERENCE_COLOR,
    hierarchical_interval,
    label_panels,
    pooled_points,
    pooled_strip,
    save_figure,
)

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
    "no\nfixation": "no\ninitiation",
    "penalty\n÷ 100": "weaker\npenalty",
}
DISPLAY_LIMIT = 0.006
# Axis limits of the recovery panels; a unit outside them is drawn at the limit.
RECOVERY_LIMITS = {"agreement": (-0.5, 1.05), "size": (0, 2)}
CAPTION = """\
**Design optimization.** **a–f, The parameterization of each term**, chosen
by prediction. Each panel compares the design in use with the same design
where one term has another parameterization, on {units} units ({sets} sets of
20 units; no unit is in two sets). A point is one unit: its validation D²
with that parameterization minus its D² with the one in use (open point at
zero). The black line is the median. Marks are two-sided Wilcoxon signed-rank
tests of the differences against zero (n.s., p ≥ 0.05; \\*, p < 0.05;
\\*\\*, p < 0.01; \\*\\*\\*, p < 0.001). **a**, Flash kernels by the gap from the flash before.
One kernel for all flashes is worse than the two gap classes in use; more
classes are not better. **b**, Length of the flash kernels. Longer kernels are
not better than 150 ms. **c**, Parameterization of the fixation term. No term and a step
are worse than the kernel in use. **d**, Withdrawal kernel. No kernel is worse.
A kernel from 300 ms before to 300 ms after the exit predicts better, but with
it the fitted fixation level is recovered at {level_under_test} of its true size,
against {level_in_use} with the kernel in use (median unit, {subject} and
{held_out}; parameter recovery as in j and k), so the design keeps the kernel of
0 to 100 ms. **e**,
Number of motion-energy PCs. {pc_sentence} **f**, Session drift. No drift is
worse; more functions are not better. The importance of each regressor is
tested on all units (main figure). **g–i, Collinearity of the design in use**, for the design of
each unit, in {subject} (the animal of the choices in a to f) and in {held_out}
(an animal that took no part in them). **g**, The generalized variance inflation factor of each regressor,
as GVIF^(1/p) (points are units; the line is the median); the dashed lines
are the usual limits of 5 and 10. The largest value in either animal is
{gvif:.2f}. **h**, The condition indices of each unit, sorted; the dashed lines
are 10 and 30. The largest is {index:.1f}. **i**, The first canonical correlation of each pair
of regressors: the largest correlation of a weighted sum of the columns of
one with a weighted sum of the columns of the other (0, nothing shared; 1, a
time course that both can make). Lower triangle: {subject}; upper triangle:
{held_out}. For a pair with spike history, the median over units. **j–k, Recovery of each regressor**: spikes are simulated from
the fitted model of each unit ({simulations} simulations), the model is fit
to them, and each recovered kernel is compared with the true one. **j**,
Agreement, 1 − |a − b|² / (|a|² + |b|²): 1 for equal kernels, 0 for unrelated
kernels. **k**, Size of the recovered kernel over the true size. Points are
units (left: {subject}; right: {held_out}); the line is the median. Every
regressor is recovered with a median agreement of {lowest_agreement:.2f} or more in both
animals. The size of a whole kernel is near 1, but the level of the fixation
kernel from 200 to 450 ms is recovered at {level_in_use} of its true size: the
L2 penalty gives a part of it to the flash × move kernels (m to o). **l–o,
Stability and leak.** **l**, Agreement of the kernels fit on the even and on the
odd trials of each unit, for the units that need the regressor (as in j).
Every regressor has a median agreement of {lowest_half:.2f} or more. **m**, The
flash × move part of the move − still flash peak (spikes/s; the excited units)
in the real fit against the leak: the same part in fits to spikes simulated
with no flash × move effect. **n**, Real minus leak. {leak} **o**, The leak
against the fixation level of the unit.
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
    # Steps 2a and 2b: the design in use, in the animal of the choices and in the
    # held-out animal (the other session of `SESSIONS`).
    roots = [args.root] + [
        args.root.parent / session for session in SESSIONS if session != args.root.name
    ]
    measures, recovered, halves, leaks = [], [], [], []
    for root in roots:
        for name, store in (
            ("collinearity", measures),
            ("parameter_recovery", recovered),
            ("kernel_reliability", halves),
            ("leak_test", leaks),
        ):
            path = root / f"{name}.npz"
            if not path.exists():
                raise FileNotFoundError(f"{path}: run `{name}` first.")
            with np.load(path) as saved:
                store.append({key: saved[key] for key in saved.files})
    subjects = [root.name.split("_")[0] for root in roots]
    with plt.rc_context(FIGURE_STYLE):
        # The dpi is for the rasterized points; the text and the lines are vectors.
        figure = plt.figure(figsize=(7.4, 13.6), layout="constrained", dpi=300)
        top, bottom, recovery, leak_row = figure.subfigures(
            4, 1, height_ratios=(6.2, 3.0, 2.5, 2.2)
        )
        axes = top.subplots(3, 2)
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
        gvif_axis, index_axis, pair_axis = bottom.subplots(
            1, 3, width_ratios=(1.35, 0.7, 1.5)
        )
        names = measures[0]["regressors"].tolist()
        order = np.argsort(-np.median(measures[0]["gvif"], 0))

        def by_regressor(axis, values_of, regressors, limits):
            """Units of both mice pooled for each regressor, with the medians.

            Black: the pooled median with its hierarchical-bootstrap 95%
            interval. Open circles: the median of the first mouse (left) and
            of the held-out mouse (right).
            """
            for position, name in enumerate(regressors):
                groups = [values[name] for values in values_of]
                column = np.concatenate(groups)
                axis.scatter(
                    position - 0.1 + rng.uniform(-0.15, 0.15, len(column)),
                    np.clip(column, *limits),
                    s=2,
                    color=REGRESSORS[name][1],
                    alpha=0.45,
                    lw=0,
                    rasterized=True,
                )
                summary = hierarchical_interval(groups)
                low, high = np.clip(summary["interval_95"], *limits)
                median = np.clip(summary["median"], *limits)
                axis.errorbar(
                    position + 0.15,
                    median,
                    yerr=[[median - low], [high - median]],
                    fmt="o",
                    ms=3,
                    color="black",
                    elinewidth=0.8,
                    capsize=1.5,
                    zorder=4,
                )
                for k, mouse in enumerate(summary["median_of_each_mouse"]):
                    axis.scatter(
                        position + 0.27 + 0.1 * k,
                        np.clip(mouse, *limits),
                        s=8,
                        facecolor="white",
                        edgecolor="black",
                        lw=0.6,
                        zorder=4,
                    )
            axis.set_xticks(
                range(len(regressors)),
                [REGRESSORS[name][0] for name in regressors],
                rotation=40,
                ha="right",
                rotation_mode="anchor",
                fontsize=5.5,
            )
            for tick, name in zip(axis.get_xticklabels(), regressors):
                tick.set_color(REGRESSORS[name][1])

        ordered = [names[column] for column in order]
        by_regressor(
            gvif_axis,
            [
                {n: m["gvif"][:, m["regressors"].tolist().index(n)] for n in ordered}
                for m in measures
            ],
            ordered,
            (0.9, 12),
        )
        for value in VIF_LIMITS:
            gvif_axis.axhline(value, color=REFERENCE_COLOR, ls="--", lw=0.8)
        gvif_axis.set_yscale("log")
        gvif_axis.set_ylim(0.9, 12)
        gvif_axis.set_yticks([1, 2, 5, 10], ["1", "2", "5", "10"])
        gvif_axis.set_ylabel("variance inflation,\n" + r"GVIF$^{1/p}$")
        for m in measures:
            for unit in m["condition_indices"]:
                index_axis.plot(unit, color="0.3", alpha=0.12, lw=0.5, rasterized=True)
        index_axis.text(
            0.97,
            0.97,
            f"{sum(len(m['condition_indices']) for m in measures)} units, {len(measures)} mice",
            transform=index_axis.transAxes,
            ha="right",
            va="top",
            fontsize=6,
        )
        for value in INDEX_LIMITS:
            index_axis.axhline(value, color=REFERENCE_COLOR, ls="--", lw=0.8)
        index_axis.set_yscale("log")
        index_axis.set_ylim(0.9, 60)
        index_axis.set_yticks([1, 3, 10, 30], ["1", "3", "10", "30"])
        index_axis.set_xlabel("dimension of the\ndesign, sorted")
        index_axis.set_ylabel("condition index")
        # One animal in each triangle of the matrix of shared directions.
        lower = measures[0]["shared_directions"]
        upper = measures[1]["shared_directions"][
            np.ix_(
                [measures[1]["regressors"].tolist().index(n) for n in names],
                [measures[1]["regressors"].tolist().index(n) for n in names],
            )
        ]
        below = np.tri(len(names), k=-1, dtype=bool)
        pairs = np.where(below, lower, np.where(below.T, upper, np.nan))
        pair_axis.imshow(pairs, cmap="Greys", vmin=0, vmax=1)
        for (row, column), value in np.ndenumerate(pairs):
            if np.isfinite(value):
                pair_axis.text(
                    column,
                    row,
                    f"{value:.2f}"[1:],
                    ha="center",
                    va="center",
                    fontsize=3.8,
                    color="white" if value > 0.55 else "black",
                )
        labels = [REGRESSORS[name][0] for name in names]
        pair_axis.set_xticks(
            range(len(labels)),
            labels,
            rotation=40,
            ha="right",
            rotation_mode="anchor",
            fontsize=5.5,
        )
        pair_axis.set_yticks(range(len(labels)), labels, fontsize=5.5)
        for ticks in (pair_axis.get_xticklabels(), pair_axis.get_yticklabels()):
            for tick, name in zip(ticks, names):
                tick.set_color(REGRESSORS[name][1])
        pair_axis.spines[["left", "bottom"]].set_visible(False)
        pair_axis.tick_params(length=0)
        pair_axis.set_title(
            f"lower: {subjects[0]}, upper: {subjects[1]}", fontsize=6, loc="left"
        )
        stats["collinearity"] = {
            subject: {
                "largest_gvif_per_column": float(m["gvif"].max()),
                "largest_condition_index": float(m["condition_indices"].max()),
                "units_outside_the_gvif_axis_limits": int(
                    np.sum((m["gvif"] > 12).any(1))
                ),
            }
            for subject, m in zip(subjects, measures)
        }
        agreement_axis, size_axis, half_axis = recovery.subplots(1, 3)
        kernels = recovered[0]["regressors"].tolist()
        for axis, key, ylabel in (
            (
                agreement_axis,
                "agreement",
                "agreement of the recovered\nand the true kernel",
            ),
            (size_axis, "size", "size of the recovered kernel\nover the true size"),
        ):
            by_regressor(
                axis,
                [
                    {n: r[key][:, r["regressors"].tolist().index(n)] for n in kernels}
                    for r in recovered
                ],
                kernels,
                RECOVERY_LIMITS[key],
            )
            axis.axhline(1, color=REFERENCE_COLOR, ls="--", lw=0.8)
            axis.set_ylim(*RECOVERY_LIMITS[key])
            axis.set_ylabel(ylabel)
        stats["recovery"] = {
            subject: {
                name: {
                    "median_agreement": float(np.median(r["agreement"][:, i])),
                    "median_size": float(np.median(r["size"][:, i])),
                    "units_outside_the_axis_limits": int(
                        np.sum(
                            (r["agreement"][:, i] < RECOVERY_LIMITS["agreement"][0])
                            | (r["size"][:, i] > RECOVERY_LIMITS["size"][1])
                        )
                    ),
                }
                for i, name in enumerate(r["regressors"].tolist())
            }
            for subject, r in zip(subjects, recovered)
        }
        # Panel l: the agreement of the kernels fit on the even and on the odd
        # trials (`kernel_reliability.py`), for the units that need the regressor.
        stable = [n for n in REGRESSORS if f"agreement__{n}" in halves[0]]
        by_regressor(
            half_axis,
            [
                {n: h[f"agreement__{n}"][h[f"needs__{n}"]] for n in stable}
                for h in halves
            ],
            stable,
            RECOVERY_LIMITS["agreement"],
        )
        gvif_axis.set_title(
            "units of both mice; circles as in j", fontsize=5.5, loc="left"
        )
        agreement_axis.set_title(
            f"units of both mice; open circles: median of {subjects[0]} (left) and"
            f" {subjects[1]}, held out (right); same in k and l",
            fontsize=5.5,
            loc="left",
        )
        half_axis.axhline(1, color=REFERENCE_COLOR, ls="--", lw=0.8)
        half_axis.set_ylim(*RECOVERY_LIMITS["agreement"])
        half_axis.set_ylabel("agreement of the kernels\nof even and odd trials")
        stats["split halves, units that need the regressor"] = {
            name: {
                subject: {
                    "units": int(h[f"needs__{name}"].sum()),
                    "median_agreement": float(
                        np.median(h[f"agreement__{name}"][h[f"needs__{name}"]])
                    ),
                    "units_below_the_axis": int(
                        np.sum(
                            h[f"agreement__{name}"][h[f"needs__{name}"]]
                            < RECOVERY_LIMITS["agreement"][0]
                        )
                    ),
                }
                for subject, h in zip(subjects, halves, strict=True)
            }
            for name in stable
        }
        # Panels m to o: the leak (`leak_test.py`), in the matched peak measure
        # of the excited units, as in `flash_by_period.py`.
        part_axis, rest_axis, level_axis = leak_row.subplots(
            1, 3, width_ratios=(1, 0.6, 1)
        )
        parts = []
        for root, leak in zip(roots, leaks, strict=True):
            subject, _, session = root.name.partition("_")
            excited = np.isin(
                leak["unit_ids"], list(peaks.excited_units(subject, session))
            )
            parts.append(
                {
                    "real": leak["real_part"][excited],
                    "leak": leak["leak_part"][excited].mean(1),
                    "level": leak["simulated_level"][excited],
                }
            )
        part_axis.plot(PART_LIMITS, PART_LIMITS, color=REFERENCE_COLOR, lw=0.8, ls="--")
        part_axis.set(xlim=PART_LIMITS, ylim=PART_LIMITS, aspect="equal")
        pooled_points(
            part_axis,
            [np.clip(p["leak"], *PART_LIMITS) for p in parts],
            [np.clip(p["real"], *PART_LIMITS) for p in parts],
            subjects,
            VISUAL,
        )
        part_axis.set_xlabel(
            "leak: flash × move part with\nno effect simulated (spikes/s)"
        )
        part_axis.set_ylabel("real flash × move part\n(spikes/s)")
        stats["leak, flash × move part, real minus leak (spikes/s)"] = pooled_strip(
            rest_axis,
            0,
            [np.clip(p["real"] - p["leak"], *PART_LIMITS) for p in parts],
            subjects,
            VISUAL,
        ) | {
            subject: {
                "units": len(p["real"]),
                "median": float(np.median(p["real"] - p["leak"])),
                "wilcoxon_p": float(wilcoxon(p["real"] - p["leak"]).pvalue),
                "units_outside_the_axis": int(
                    np.sum(
                        (np.minimum(p["real"], p["leak"]) < PART_LIMITS[0])
                        | (np.maximum(p["real"], p["leak"]) > PART_LIMITS[1])
                    )
                ),
            }
            for subject, p in zip(subjects, parts, strict=True)
        }
        rest_axis.axhline(0, color=REFERENCE_COLOR, lw=0.8)
        rest_axis.set_xticks([])
        rest_axis.set_xlim(-0.5, 0.9)
        rest_axis.set_ylim(*PART_LIMITS)
        rest_axis.set_ylabel("flash × move part,\nreal minus leak (spikes/s)")
        level_axis.set_ylim(*PART_LIMITS)
        pooled_points(
            level_axis,
            [p["level"] for p in parts],
            [np.clip(p["leak"], *PART_LIMITS) for p in parts],
            subjects,
            TASK,
        )
        level_axis.axhline(0, color=REFERENCE_COLOR, lw=0.6)
        level_axis.set_xlabel("fixation level of the unit\n(Δ log rate)")
        level_axis.set_ylabel("leak (spikes/s)")
        label_panels(axes.flat, x=-0.2, y=1.04)
        label_panels(
            [None] * 9 + [agreement_axis, size_axis, half_axis], x=-0.2, y=1.04
        )
        label_panels([None] * 12 + [part_axis, rest_axis, level_axis], x=-0.3, y=1.04)
        label_panels([None] * 6 + [gvif_axis, index_axis, pair_axis], x=-0.3, y=1.04)
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
            gvif=max(
                v["largest_gvif_per_column"] for v in stats["collinearity"].values()
            ),
            simulations=recovery[SESSIONS[0]]["simulations"],
            level_in_use=" and ".join(
                f"{recovery[s]['fixation level, recovered over true (median)']['in_use']:.2f}"
                for s in SESSIONS
            ),
            level_under_test=" and ".join(
                f"{recovery[s]['fixation level, recovered over true (median)']['under_test']:.2f}"
                for s in SESSIONS
            ),
            lowest_agreement=min(
                v["median_agreement"]
                for animal in stats["recovery"].values()
                for v in animal.values()
            ),
            subject=subjects[0],
            held_out=subjects[1],
            index=max(
                v["largest_condition_index"] for v in stats["collinearity"].values()
            ),
            lowest_half=min(
                mouse["median_agreement"]
                for entry in stats[
                    "split halves, units that need the regressor"
                ].values()
                for mouse in entry.values()
            ),
            leak="; ".join(
                f"{subject}: median {entry['median']:+.1f} spikes/s"
                f" ({entry['units']} units, p = {entry['wilcoxon_p']:.2g})"
                for subject, entry in stats[
                    "leak, flash × move part, real minus leak (spikes/s)"
                ].items()
                if subject in subjects
            )
            + ".",
        )
    )
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
