"""Show the flash kernels of every unit, by the gap from the flash before and by period.

Question: how does the response of V1 units to a flash, as the model has it,
depend on the gap from the flash before and on the period of the trial (the
fixation or the time after the center exit)?

Comparison: the flash kernel of each unit in four conditions, from the final
refit. No new fit. The comparisons are paired: each unit has a kernel in each
condition.

- Kernel of a condition (`results.flash_kernel_at`): the flash kernel of the
  gap class of that condition, plus the flash × fixation kernel at the value of
  the period. Conditions (`CONDITIONS`): a flash in the fixation in each of
  the three gap classes, and a flash after the center exit in the middle gap
  class.
- Panel a (every unit): one map for each condition, with one row for each unit
  and the same order of units in all maps (by the mean response over the
  three conditions in the fixation). The units do not have one kind of response: some fire more after a
  flash and some fire less, so a median curve over units is near zero and is
  not shown. The color scale stops at ±`COLOR_LIMIT`; larger values have the
  color of the limit.
- Response of a unit: the mean of its kernel from `RESPONSE_MS`.
- Panel b (gap): the response at the longest gap against the response at the
  shortest gap, one point for each unit. Panel c (period): the response after
  the exit against the response in the fixation, at the middle gap. Error bars
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
are those of `fit.py`. The sampling unit is the unit. The script writes
`flash_kernels.svg`, `flash_kernels.json`, and `flash_kernels.npz` (the response
of each unit) in the session directory.

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
from thesis.plotting import CONCEPT_COLORS, FIGURE_STYLE, REFERENCE_COLOR, save_figure

# name: (gap in ms that selects the gap class, after the center exit)
CONDITIONS = {
    "40 ms gap": (40, False),
    "80 to 120 ms gap": (100, False),
    "160+ ms gap": (250, False),
    "80 to 120 ms gap,\nafter exit": (100, True),
}
FIXATION = ("40 ms gap", "80 to 120 ms gap", "160+ ms gap")
# name: (first condition, second condition, conditions that give the kind of unit)
# The kind of a unit comes from conditions that are not in the pair. A kind
# that comes from one of the two compared responses makes that response more
# extreme than the other one by selection alone.
PAIRS = {
    "gap": ("40 ms gap", "160+ ms gap", ("80 to 120 ms gap",)),
    "period": (
        "80 to 120 ms gap",
        "80 to 120 ms gap,\nafter exit",
        ("40 ms gap", "160+ ms gap"),
    ),
}
RESPONSE_MS = (40, 100)
KIND_LIMIT = 0.2
COLOR_LIMIT = 1.0
AXIS_LIMITS = (-1.2, 1.6)
VISUAL = CONCEPT_COLORS["visual"]


def _paired(first: np.ndarray, second: np.ndarray) -> dict:
    change = second - first
    return {
        "units": len(change),
        "median_change": float(np.median(change)),
        "units_higher": int(np.sum(change > 0)),
        "wilcoxon_p": float(wilcoxon(change).pvalue) if len(change) > 5 else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=cli.PoissonGLM().root)
    parser.add_argument("--units", type=cli._unit_set, default="all")
    args = parser.parse_args()
    model = cli.PoissonGLM(root=args.root)
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
        for term in (*dmat.flash_gap_names(), dmat.FLASH_AFTER_EXIT)
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
                flash_kernel_at(terms, None, levels, gap, after)
                for gap, after in CONDITIONS.values()
            ]
        )

    checkpoints = model.fit_dir(args.units) / "checkpoints"
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
    mean_of = lambda names: response[:, [index[name] for name in names]].mean(1)  # noqa: E731
    stats = {"units": len(final), "response_window_ms": RESPONSE_MS}
    for name, (first, second, kind_from) in PAIRS.items():
        x, y = response[:, index[first]], response[:, index[second]]
        reference = mean_of(kind_from)
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
        stats[f"{name}: {second!r} minus {first!r}".replace("\\n", " ")] = {
            "kind_of_unit_from": list(kind_from),
            **{kind: _paired(x[mask], y[mask]) for kind, mask in kinds.items()},
            "units_outside_the_axis_limits": int(outside.sum()),
        }

    # One order for all maps: the mean response over the fixation conditions.
    order = np.argsort(mean_of(FIXATION))[::-1]
    with plt.rc_context(FIGURE_STYLE):
        figure = plt.figure(figsize=(7.2, 5.6))
        outer = figure.add_gridspec(2, 1, height_ratios=(1.55, 1), hspace=0.42)
        maps = outer[0].subgridspec(1, len(CONDITIONS), wspace=0.18)
        below = outer[1].subgridspec(1, 3, width_ratios=(1, 1, 0.35), wspace=0.55)
        map_axes = []
        for position, name in enumerate(CONDITIONS):
            axis = figure.add_subplot(maps[position])
            image = axis.imshow(
                kernel[order, position],
                aspect="auto",
                cmap="RdBu_r",
                vmin=-COLOR_LIMIT,
                vmax=COLOR_LIMIT,
                extent=(lags[0], lags[-1], len(order), 0),
                interpolation="nearest",
            )
            axis.text(0, 1.02, name, transform=axis.transAxes, va="bottom", fontsize=7)
            axis.set_xlabel("time from flash (ms)")
            if position:
                axis.tick_params(labelleft=False)
            else:
                axis.set_ylabel("unit, sorted by its mean\nresponse in the fixation")
            map_axes.append(axis)
        bar = figure.colorbar(image, ax=map_axes, fraction=0.018, pad=0.015)
        bar.set_label(r"flash kernel ($\Delta$ log rate)")
        pair_axes = []
        for position, (first, second, _) in enumerate(PAIRS.values()):
            axis = figure.add_subplot(below[position])
            x, y = response[:, index[first]], response[:, index[second]]
            axis.errorbar(
                x,
                y,
                xerr=spread[:, index[first]],
                yerr=spread[:, index[second]],
                fmt="o",
                color=VISUAL,
                markersize=2.3,
                elinewidth=0.4,
                alpha=0.65,
            )
            axis.plot(AXIS_LIMITS, AXIS_LIMITS, color=REFERENCE_COLOR, ls="--", lw=0.8)
            for line in (axis.axhline, axis.axvline):
                line(0, color=REFERENCE_COLOR, linewidth=0.5)
            axis.set_xlim(AXIS_LIMITS)
            axis.set_ylim(AXIS_LIMITS)
            axis.set_aspect("equal")
            axis.set_xlabel("response, " + first.replace(",\n", ", "))
            axis.set_ylabel("response,\n" + second.replace(",\n", ", "))
            pair_axes.append(axis)
        figure.canvas.draw()
        to_figure = figure.transFigure.inverted()
        for letter, axis in zip("abc", (map_axes[0], *pair_axes), strict=True):
            edge = axis.yaxis.get_tightbbox().transformed(to_figure).x0
            figure.text(
                edge - 0.01,
                axis.get_position().y1 + 0.035,
                letter,
                fontweight="bold",
                fontsize=10,
            )
        print(save_figure(figure, args.root / "flash_kernels"))
    # The response of each unit, for the modules that group units by it.
    np.savez(
        args.root / "flash_kernels.npz",
        unit_ids=np.asarray([int(item["unit_id"]) for item in final]),
        conditions=np.asarray([name.replace(",\n", ", ") for name in CONDITIONS]),
        response=response,
        fixation_response=mean_of(FIXATION),
    )
    output = args.root / "flash_kernels.json"
    output.write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
