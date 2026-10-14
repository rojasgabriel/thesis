"""Show the flash kernels of every unit, by the gap from the flash before and by period.

Question: how does the response of V1 units to a flash, as the model has it,
depend on the gap from the flash before and on the period of the trial (the
fixation or the time after the center exit)?

Comparison: the flash kernel of each unit in four conditions, from the final
refit. No new fit. The comparisons are paired: each unit has a kernel in each
condition.

- Kernel of a condition (`results.flash_kernel_at`): the flash kernel of the
  gap class of that condition, plus the flash × move kernel at the value of
  the period. Conditions (`CONDITIONS`): a flash in the fixation (still) and after the center exit
  (move), in each of the two gap classes. The flash × move × gap kernel is
  added at the product of the centered period and the centered gap class.
- Panels a and b (the two kinds of unit): the median kernel over the units
  that fire more after a flash (a) and that fire less (b). Left: a flash in
  the fixation in the two gap classes. Right: a flash in the fixation and
  after the exit in the class of the longer gaps, with the quartiles as bands. Here the kind of a
  unit is from its mean response over the two conditions in the fixation
  (above `KIND_LIMIT` or below -`KIND_LIMIT`), so the sign of the curves is by
  selection; the panels show the shape.
- Panel c (every unit): one map for each condition, with one row for each unit
  and the same order of units in all maps (by the mean response over the
  two conditions in the fixation). The units do not have one kind of
  response, so a median curve over all units is near zero and is not shown.
  The color scale stops at ±`COLOR_LIMIT`; larger values have the color of
  the limit.
- Response of a unit: the mean of its kernel from `RESPONSE_MS`.
- Panel d (gap): the response at a longer gap against the response at the
  40 ms gap when still, one point for each unit. Panels e and f (period):
  the response in move against the response when still, at a longer gap (e)
  and at the 40 ms gap (f). Error bars
  are the SD over the 10 cross-validation folds, in both directions. A point
  outside `AXIS_LIMITS` is not drawn; it is in the statistics, and the number
  of those points is in the output.
- Statistics: two-sided Wilcoxon signed-rank test of the paired difference
  over all units, and over the units that fire more (response above
  `KIND_LIMIT`) and fire less (below -`KIND_LIMIT`) apart, because a change
  that makes every response larger has opposite signs in the two kinds. The
  kind of a unit comes from conditions that are not in the pair (`PAIRS`): a
  kind from one of the two compared responses makes that response more extreme
  by selection alone.

Units (`--units`), trials, alignment (first flash), baseline, and exclusions
are those of `fit.py`. The sampling unit is the unit, and each test is within
one animal (`SESSIONS`); the units of the two animals are not pooled. Panels
a and b, c and d, g to i are GRB006; the others are GRB064, in the same
order. The script writes `flash_kernels.svg` and `flash_kernels.json` (by
session) in `--out`, and `flash_kernels.npz` (the response of each unit) in
each session directory.

    uv run python -m thesis.ephys.analyses.glm.flash_kernels
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

from thesis.ephys.analyses.glm import cli, dmat
from thesis.ephys.analyses.glm.fit import load_unit_records
from thesis.ephys.analyses.glm.results import flash_kernel_at, flash_kernel_levels
from thesis.plotting import FIGURE_STYLE, REFERENCE_COLOR, save_figure

# The sessions of the figure, one row of panels for each. Each test is within one
# animal; the units of the two animals are not pooled.
SESSIONS = ("GRB006_20240821_121447", "GRB064_20260916_132939")

# name: (gap in ms that selects the gap class, after the center exit)
# "Still" is a flash in the fixation and "move" is a flash after the center exit.
CONDITIONS = {
    "still, 40 ms gap": (40, False),
    "still, 80+ ms gap": (100, False),
    "move, 40 ms gap": (40, True),
    "move, 80+ ms gap": (100, True),
}
FIXATION = ("still, 40 ms gap", "still, 80+ ms gap")
# name: (first condition, second condition, conditions that give the kind of unit)
# The kind of a unit comes from conditions that are not in the pair. A kind
# that comes from one of the two compared responses makes that response more
# extreme than the other one by selection alone.
PAIRS = {
    "gap": (
        "still, 40 ms gap",
        "still, 80+ ms gap",
        ("move, 40 ms gap", "move, 80+ ms gap"),
    ),
    "period": ("still, 80+ ms gap", "move, 80+ ms gap", ("still, 40 ms gap",)),
    "period, 40 ms gap": (
        "still, 40 ms gap",
        "move, 40 ms gap",
        ("still, 80+ ms gap",),
    ),
}
RESPONSE_MS = (40, 100)
KIND_LIMIT = 0.2
COLOR_LIMIT = 1.0
AXIS_LIMITS = (-1.2, 1.6)
# The two kinds of unit, in the colors of the two ends of the map of panel c.
KIND_COLORS = {"fire more": "#b2182b", "fire less": "#2166ac"}
OTHER_COLOR = "0.7"


def _paired(first: np.ndarray, second: np.ndarray) -> dict:
    change = second - first
    return {
        "units": len(change),
        "median_change": float(np.median(change)),
        "units_higher": int(np.sum(change > 0)),
        "wilcoxon_p": float(wilcoxon(change).pvalue) if len(change) > 5 else None,
    }


def _mean_of(response: np.ndarray, conditions) -> np.ndarray:
    """Return the mean response of each unit over some conditions."""
    return response[:, [list(CONDITIONS).index(name) for name in conditions]].mean(1)


def session_kernels(root: Path, unit_set: str) -> dict:
    """Return the kernels and responses of the units of one session, with the statistics.

    Also writes the response of each unit to `flash_kernels.npz` in the session
    directory, for the modules that group units by it.
    """
    model = cli.PoissonGLM(root=root)
    metadata = json.loads(model.design.with_suffix(".json").read_text())
    names = metadata["column_names"]
    scale = np.asarray(metadata["training_scale"])
    basis = dmat.task_temporal_bases()[dmat.FLASH_AFTER_EXIT]
    lags = 1000 * basis.basis_time
    window = (lags >= RESPONSE_MS[0]) & (lags < RESPONSE_MS[1])
    prepared = dmat._load_windows(model.windows)
    levels = flash_kernel_levels(dmat._load_selected_trials(prepared))
    columns = {
        term: [
            i for i, column in enumerate(names) if column.startswith(f"{term}_basis")
        ]
        for term in (
            *dmat.flash_gap_names(),
            dmat.FLASH_AFTER_EXIT,
            dmat.FLASH_MOVE_GAP,
        )
    }

    def kernels(coefficients) -> np.ndarray:
        """Return the kernel of each condition for one set of coefficients."""
        weights = np.asarray(coefficients, dtype=float)
        terms = {
            term: (lags, (weights[index] / scale[index]) @ basis.basis.T)
            for term, index in columns.items()
        }
        return np.stack(
            [
                flash_kernel_at(terms, levels, gap, after)
                for gap, after in CONDITIONS.values()
            ]
        )

    checkpoints = model.fit_dir(unit_set) / "checkpoints"
    final = load_unit_records(checkpoints, "unit_*_final.json")
    folds = {
        int(item["unit_id"]): item["folds"]
        for item in load_unit_records(checkpoints, "unit_*_folds.json")
    }
    kernel = np.asarray([kernels(item["coefficients"]) for item in final])
    response = kernel[:, :, window].mean(-1)  # (unit, condition)
    spread = np.asarray(
        [
            np.std(
                [
                    kernels(fold["coefficients"])[:, window].mean(-1)
                    for fold in folds[int(item["unit_id"])]
                ],
                axis=0,
            )
            for item in final
        ]
    )
    index = {name: position for position, name in enumerate(CONDITIONS)}
    stats = {"units": len(final), "response_window_ms": RESPONSE_MS}
    for name, (first, second, kind_from) in PAIRS.items():
        x, y = response[:, index[first]], response[:, index[second]]
        reference = _mean_of(response, kind_from)
        kinds = {
            "all units": np.ones(len(final), dtype=bool),
            "units that fire more after a flash": reference > KIND_LIMIT,
            "units that fire less after a flash": reference < -KIND_LIMIT,
        }
        outside = (
            (x < AXIS_LIMITS[0])
            | (x > AXIS_LIMITS[1])
            | (y < AXIS_LIMITS[0])
            | (y > AXIS_LIMITS[1])
        )
        stats[f"{name}: {second!r} minus {first!r}"] = {
            "kind_of_unit_from": list(kind_from),
            **{kind: _paired(x[mask], y[mask]) for kind, mask in kinds.items()},
            "units_outside_the_axis_limits": int(outside.sum()),
        }
    np.savez(
        root / "flash_kernels.npz",
        unit_ids=np.asarray([int(item["unit_id"]) for item in final]),
        conditions=np.asarray(list(CONDITIONS)),
        response=response,
        response_sd_over_folds=spread,
        fixation_response=_mean_of(response, FIXATION),
    )
    return {
        "subject": root.name.split("_")[0],
        "lags": lags,
        "kernel": kernel,
        "response": response,
        "spread": spread,
        "stats": stats,
    }


def draw_curves(figure, slot, data: dict, legend: bool) -> list:
    """Draw the median kernel of each kind of unit; return the first axis of each kind."""
    lags, kernel = data["lags"], data["kernel"]
    index = {name: position for position, name in enumerate(CONDITIONS)}
    fixation_mean = _mean_of(data["response"], FIXATION)
    curves = slot.subgridspec(1, 4, wspace=0.12)
    axes = []
    for k, (kind, members) in enumerate(
        (
            ("fire more", fixation_mean > KIND_LIMIT),
            ("fire less", fixation_mean < -KIND_LIMIT),
        )
    ):
        color = KIND_COLORS[kind]
        by_gap = figure.add_subplot(curves[2 * k], sharey=axes[0] if axes else None)
        by_period = figure.add_subplot(curves[2 * k + 1], sharey=by_gap)
        for name, shade in zip(FIXATION, (0.5, 1.0), strict=True):
            by_gap.plot(
                lags,
                np.median(kernel[members, index[name]], 0),
                color=color,
                alpha=shade,
            )
        for name, style in zip(PAIRS["period"][:2], ("-", "--"), strict=True):
            values = kernel[members, index[name]]
            low, median, high = np.percentile(values, [25, 50, 75], axis=0)
            by_period.plot(lags, median, color=color, linestyle=style)
            by_period.fill_between(lags, low, high, color=color, alpha=0.1, lw=0)
        for axis in (by_gap, by_period):
            axis.axhline(0, color=REFERENCE_COLOR, linestyle="--", linewidth=0.8)
            axis.set_xlabel("time from flash (ms)")
            axis.tick_params(labelleft=not axes and axis is by_gap)
        by_gap.text(
            0.03 if k == 0 else 0.97,
            0.03 if k == 0 else 0.97,
            f"{kind} after a flash\n{int(members.sum())} units",
            color=color,
            transform=by_gap.transAxes,
            ha="left" if k == 0 else "right",
            va="bottom" if k == 0 else "top",
            fontsize=6,
        )
        if not axes:
            by_gap.set_ylabel(
                f"{data['subject']}\nmedian flash kernel\n" + r"($\Delta$ log rate)"
            )
        if legend and not axes:
            for row, (name, shade) in enumerate(
                zip(("40 ms gap", "80+ ms gap"), (0.5, 1.0), strict=True)
            ):
                by_gap.text(
                    0.03,
                    1.12 - 0.105 * row,
                    name,
                    color="black",
                    alpha=shade,
                    transform=by_gap.transAxes,
                    va="top",
                    fontsize=6,
                )
            for row, label in enumerate(("still", "move (dashed)")):
                by_period.text(
                    0.03,
                    1.12 - 0.105 * row,
                    label,
                    color="black",
                    transform=by_period.transAxes,
                    va="top",
                    fontsize=6,
                )
        axes.append(by_gap)
    return axes


def draw_maps(figure, slot, data: dict, first: bool) -> list:
    """Draw the kernel of every unit in each condition, one map for each condition."""
    lags, kernel = data["lags"], data["kernel"]
    # One order for all maps: the mean response over the fixation conditions.
    order = np.argsort(_mean_of(data["response"], FIXATION))[::-1]
    maps = slot.subgridspec(1, len(CONDITIONS), wspace=0.15)
    axes = []
    for position, name in enumerate(CONDITIONS):
        axis = figure.add_subplot(maps[position])
        axis.imshow(
            kernel[order, position],
            aspect="auto",
            cmap="RdBu_r",
            vmin=-COLOR_LIMIT,
            vmax=COLOR_LIMIT,
            extent=(lags[0], lags[-1], len(order), 0),
            interpolation="nearest",
        )
        axis.text(
            0,
            1.02,
            name.replace(", ", ",\n"),
            transform=axis.transAxes,
            va="bottom",
            fontsize=6,
        )
        if not position:
            axis.set_xlabel("time from flash (ms)", loc="left")
        if position:
            axis.tick_params(labelleft=False)
        else:
            axis.set_ylabel(
                f"{data['subject']}: unit, sorted by\nits mean response when still"
                if first
                else f"{data['subject']}: unit"
            )
        axes.append(axis)
    return axes


def draw_pairs(figure, slot, data: dict) -> list:
    """Draw the paired responses of `PAIRS`, one point for each unit."""
    response, spread = data["response"], data["spread"]
    index = {name: position for position, name in enumerate(CONDITIONS)}
    pairs = slot.subgridspec(1, 3, wspace=0.75)
    axes = []
    for position, (first, second, kind_from) in enumerate(PAIRS.values()):
        axis = figure.add_subplot(pairs[position])
        x, y = response[:, index[first]], response[:, index[second]]
        # The kind of a unit comes from the conditions outside the pair.
        reference = _mean_of(response, kind_from)
        for members, color in (
            (np.abs(reference) <= KIND_LIMIT, OTHER_COLOR),
            (reference > KIND_LIMIT, KIND_COLORS["fire more"]),
            (reference < -KIND_LIMIT, KIND_COLORS["fire less"]),
        ):
            axis.errorbar(
                x[members],
                y[members],
                xerr=spread[members, index[first]],
                yerr=spread[members, index[second]],
                fmt="o",
                color=color,
                ms=2.5,
                elinewidth=0.4,
                alpha=0.8,
            )
        axis.plot(AXIS_LIMITS, AXIS_LIMITS, color=REFERENCE_COLOR, ls="--", lw=0.8)
        for line in (axis.axhline, axis.axvline):
            line(0, color=REFERENCE_COLOR, linewidth=0.6)
        axis.set_xlim(AXIS_LIMITS)
        axis.set_ylim(AXIS_LIMITS)
        axis.set_aspect("equal")
        axis.set_xlabel(f"response, {first}")
        axis.set_ylabel(
            f"{data['subject']}: response,\n{second}"
            if position == 0
            else f"response,\n{second}"
        )
        axes.append(axis)
    return axes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("figures/glm"))
    parser.add_argument("--units", type=cli._unit_set, default="all")
    args = parser.parse_args()
    sessions = [session_kernels(args.out / session, args.units) for session in SESSIONS]
    with plt.rc_context(FIGURE_STYLE):
        figure = plt.figure(figsize=(7.2, 10.4))
        outer = figure.add_gridspec(
            5, 1, height_ratios=(0.8, 0.8, 1.7, 1, 1), hspace=0.7
        )
        curve_axes = [
            axis
            for row, data in enumerate(sessions)
            for axis in draw_curves(figure, outer[row], data, legend=row == 0)
        ]
        halves = outer[2].subgridspec(1, 2, wspace=0.2)
        map_groups = [
            draw_maps(figure, halves[k], data, first=k == 0)
            for k, data in enumerate(sessions)
        ]
        bar = figure.colorbar(
            map_groups[-1][0].images[0],
            ax=[axis for group in map_groups for axis in group],
            fraction=0.015,
            pad=0.015,
        )
        bar.set_label(r"flash kernel ($\Delta$ log rate)")
        pair_axes = [
            axis
            for row, data in enumerate(sessions)
            for axis in draw_pairs(figure, outer[3 + row], data)
        ]
        figure.canvas.draw()
        to_figure = figure.transFigure.inverted()
        lettered = (*curve_axes, *(group[0] for group in map_groups), *pair_axes)
        for letter, axis in zip("abcdefghijkl", lettered, strict=True):
            box = axis.yaxis.get_tightbbox()  # None for an axis with no tick labels
            edge = (
                box.transformed(to_figure).x0 if box else axis.get_position().x0 - 0.01
            )
            figure.text(
                edge - 0.01,
                axis.get_position().y1 + 0.02,
                letter,
                fontweight="bold",
                fontsize=10,
            )
        print(save_figure(figure, args.out / "flash_kernels"))
    stats = {
        session: data["stats"] for session, data in zip(SESSIONS, sessions, strict=True)
    }
    output = args.out / "flash_kernels.json"
    output.write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
