"""Draw the design optimization of the V1 GLM in one figure: the form of each term and the collinearity.

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
direction that the term of the panel, in that form, shares with another
regressor (task or motion energy): the largest first canonical correlation of a pair of regressors
with the term in it (`collinearity.shared_directions`; the pair is in the
stats file). A form with no such term has no number. It is in red when it is
above the largest value of all pairs of the design in use, panel i
(`SHARED_TOLERANCE`).

- Panels g to i (`collinearity.py`): the collinearity of the design in use,
  for the design of each unit of the session. g, the generalized variance
  inflation factor of each regressor (points are units; the line is the
  median; dashed lines at 5 and 10). h, the condition indices of each unit,
  sorted (dashed lines at 10 and 30). i, the shared direction of each pair of
  regressors; for a pair with spike history, the median over units.

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
from thesis.ephys.analyses.glm.collinearity import (
    INDEX_LIMITS,
    VIF_LIMITS,
    shared_directions,
)
from thesis.ephys.analyses.glm.main_figure import (
    DRIFT_COLOR,
    MOTION,
    REGRESSORS,
    TASK,
    VISUAL,
)
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
    "no\nfixation": "no\ninitiation",
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
with another regressor (red: above the largest value of the design in use,
panel i). A form replaces the form in use when it
predicts reliably better and is separable; no form does both. The importance of each regressor is tested on all units
(main figure). **g–i, Collinearity of the design in use**, for the design of
each unit. **g**, The generalized variance inflation factor of each regressor,
as GVIF^(1/p) (points are units; the line is the median); the dashed lines
are the usual limits of 5 and 10. The largest value is {gvif:.2f}. **h**, The
condition indices of each unit, sorted; the dashed lines are 10 and 30. The
largest is {index:.1f}. **i**, The shared direction of each pair of
regressors: the largest correlation of a weighted sum of the columns of one
with a weighted sum of the columns of the other (0, nothing shared; 1, a time
course that both can make). For a pair with spike history, the median over
units.
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
    settings: dict, terms: tuple[str, ...], prepared: dict, trials, video: np.ndarray
) -> tuple[float, str, str] | None:
    """Return the largest direction that a term of one design shares with another.

    `terms` are the starts of the names of the regressors of the term (all
    regressors for an empty tuple). The value is the largest first canonical
    correlation of a pair of regressors (task and motion energy; `video` has
    the motion-energy columns) with one of them in the pair, with the names of
    the pair. None when the design has no such regressor.
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
    groups["video"] = slice(cursor, cursor + video.shape[1])
    columns = np.hstack((values, video))[dmat._valid_bin_mask(prepared)]
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
    model = cli.PoissonGLM(root=args.root)
    prepared = dmat._load_windows(model.windows)
    trials = dmat._load_selected_trials(prepared)
    metadata = json.loads(model.design.with_suffix(".json").read_text())
    base = int(metadata["base_columns"])
    video = np.asarray(
        np.load(model.design, mmap_mode="r")[
            :, base : base + max(dmat.VIDEO_COMPONENT_COUNTS)
        ],
        dtype=float,
    )
    measures_path = args.root / "collinearity.npz"
    if not measures_path.exists():
        raise FileNotFoundError(f"{measures_path}: run `collinearity` first.")
    with np.load(measures_path) as saved:
        measures = {key: saved[key] for key in saved.files}
    # The limit of the rule: the largest pair of all regressors of the design.
    largest = float(np.nanmax(measures["shared_directions"]))
    limit = largest + SHARED_TOLERANCE
    stats["largest_shared_direction_of_the_design_in_use"] = largest
    with plt.rc_context(FIGURE_STYLE):
        # The dpi is for the rasterized points; the text and the lines are vectors.
        figure = plt.figure(figsize=(7.4, 10.0), layout="constrained", dpi=300)
        top, bottom = figure.subfigures(2, 1, height_ratios=(6.9, 3.1))
        axes = top.subplots(3, 2)
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
                    shared_direction(
                        module.DESIGNS[name], terms, prepared, trials, video
                    )
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
                    s=2.5,
                    color=color,
                    alpha=0.45,
                    lw=0,
                    rasterized=True,  # 80 points for each form: keeps the file small
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
                fontsize=6,
            )
            axis.set_xlabel(xlabel, color=color, fontsize=7.5)
            if axis in axes[:, 0]:  # one y label for each row
                axis.set_ylabel(r"$D^2$ minus $D^2$" + "\nof the form in use")
            axis.tick_params(labelsize=7)
            axis.set_yticks([-0.005, 0, 0.005], ["−0.005", "0", "0.005"])
        gvif_axis, index_axis, pair_axis = bottom.subplots(
            1, 3, width_ratios=(1.35, 0.7, 1.5)
        )
        names = measures["regressors"].tolist()
        order = np.argsort(-np.median(measures["gvif"], 0))
        for position, column in enumerate(order):
            values = measures["gvif"][:, column]
            gvif_axis.scatter(
                position + rng.uniform(-0.25, 0.25, len(values)),
                values,
                s=2.5,
                color=REGRESSORS[names[column]][1],
                alpha=0.5,
                lw=0,
                rasterized=True,
            )
            gvif_axis.hlines(
                np.median(values), position - 0.35, position + 0.35, "black"
            )
        for value in VIF_LIMITS:
            gvif_axis.axhline(value, color=REFERENCE_COLOR, ls="--", lw=0.8)
        gvif_axis.set_yscale("log")
        gvif_axis.set_ylim(0.9, 12)
        gvif_axis.set_yticks([1, 2, 5, 10], ["1", "2", "5", "10"])
        gvif_axis.set_xticks(
            range(len(names)),
            [REGRESSORS[names[column]][0] for column in order],
            rotation=40,
            ha="right",
            rotation_mode="anchor",
            fontsize=5.5,
        )
        for tick, column in zip(gvif_axis.get_xticklabels(), order):
            tick.set_color(REGRESSORS[names[column]][1])
        gvif_axis.set_ylabel("variance inflation,\n" + r"GVIF$^{1/p}$")
        for unit in measures["condition_indices"]:
            index_axis.plot(unit, color="0.3", alpha=0.15, lw=0.5, rasterized=True)
        for value in INDEX_LIMITS:
            index_axis.axhline(value, color=REFERENCE_COLOR, ls="--", lw=0.8)
        index_axis.set_yscale("log")
        index_axis.set_ylim(0.9, 60)
        index_axis.set_yticks([1, 3, 10, 30], ["1", "3", "10", "30"])
        index_axis.set_xlabel("dimension of the\ndesign, sorted")
        index_axis.set_ylabel("condition index")
        # The matrix is symmetric: only the lower triangle is drawn.
        pairs = np.where(
            np.tri(len(names), k=-1, dtype=bool), measures["shared_directions"], np.nan
        )[1:, :-1]
        pair_axis.imshow(pairs, cmap="Greys", vmin=0, vmax=1)
        for (row, column), value in np.ndenumerate(pairs):
            if np.isfinite(value):
                pair_axis.text(
                    column,
                    row,
                    f"{value:.2f}"[1:],
                    ha="center",
                    va="center",
                    fontsize=4.3,
                    color="white" if value > 0.55 else "black",
                )
        labels = [REGRESSORS[name][0] for name in names]
        pair_axis.set_xticks(
            range(len(labels) - 1),
            labels[:-1],
            rotation=40,
            ha="right",
            rotation_mode="anchor",
            fontsize=5.5,
        )
        pair_axis.set_yticks(range(len(labels) - 1), labels[1:], fontsize=5.5)
        for ticks, shown in (
            (pair_axis.get_xticklabels(), names[:-1]),
            (pair_axis.get_yticklabels(), names[1:]),
        ):
            for tick, name in zip(ticks, shown):
                tick.set_color(REGRESSORS[name][1])
        pair_axis.spines[["left", "bottom"]].set_visible(False)
        pair_axis.tick_params(length=0)
        stats["collinearity"] = {
            "largest_gvif_per_column": float(measures["gvif"].max()),
            "largest_condition_index": float(measures["condition_indices"].max()),
            "units_outside_the_gvif_axis_limits": int(
                np.sum((measures["gvif"] > 12).any(1))
            ),
        }
        label_panels(axes.flat, x=-0.2, y=1.04)
        label_panels([None] * 6 + [gvif_axis, index_axis, pair_axis], x=-0.3, y=1.04)
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
            gvif=stats["collinearity"]["largest_gvif_per_column"],
            index=stats["collinearity"]["largest_condition_index"],
        )
    )
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
