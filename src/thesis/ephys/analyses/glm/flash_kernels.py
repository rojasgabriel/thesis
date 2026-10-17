"""Show the flash kernels of every unit, and the flash response at the two gaps.

Question: how does the response of V1 units to a flash depend on the gap from
the flash before? (The period, still against move, is in `flash_by_period.py`,
with the same matched measure.)

Comparison: no new fit. The comparisons are paired: each unit has a value in
each condition. Each test is within one mouse (`SESSIONS`); the units of the
two mice are not pooled. Both mice are in each axis: solid lines and circles
GRB006, dashed lines and triangles GRB064.

- Kernel of a condition (`results.flash_kernel_at`): the flash kernel of the
  gap class of that condition, plus the flash × move kernel at the value of
  the period. Conditions (`CONDITIONS`): a flash in the fixation (still) and
  after the center exit (move), in each of the two gap classes. The flash ×
  move × gap kernel is added at the product of the centered period and the
  centered gap class.
- Panels a and b: the median kernel, when still, at the 40 ms gap (light) and
  the 80+ ms gap (dark), over the units that fire more after a flash (a) and
  that fire less (b). The kind of a unit is from its mean kernel response over
  the two still conditions (above `KIND_LIMIT` or below -`KIND_LIMIT`; the
  response is the mean of the kernel from `RESPONSE_MS`), so the sign of the
  curves is by selection; the panels show the shape.
- Panels c and d (every unit, one map for each mouse, because the units
  differ): the kernel in each condition, one row for each unit, in the same
  order in all maps of a mouse (by the mean response when still). The color
  scale stops at ±`COLOR_LIMIT`.
- Panels e to g (the gap, with the matched measure of `peaks.py`): the peak of
  the excited units (StimulusResponse) after flashes in the fixation at the
  80+ ms gap and at the 40 ms gap (not the first flash of a trial), for the
  observed rate (e) and the model rate (f), and 40 ms minus 80+ ms in both
  (g). The baseline is the 80+ ms rate from `peaks.BASELINE_MS`. Small markers
  are units; large markers the median with its bootstrap 95% interval. A
  point outside `GAP_LIMITS` (linear axes) is not drawn; it is in the
  statistics, and the count is in the output.
- Statistics: two-sided Wilcoxon signed-rank tests of the paired difference.
  For the kernels (in the output), also for the units that fire more and that
  fire less apart, with the kind from conditions outside the pair (`PAIRS`).

Units (`--units`), trials, alignment (first flash), baseline, and exclusions
are those of `fit.py`. The sampling unit is the unit. The script writes
`flash_kernels.svg` and `flash_kernels.json` (by session) in `--out`, and
`flash_kernels.npz` (the kernel response of each unit) in each session
directory.

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

from thesis.ephys.analyses.glm import cli, dmat, peaks
from thesis.ephys.analyses.glm.fit import load_unit_records
from thesis.ephys.analyses.glm.results import flash_kernel_at, flash_kernel_levels
from thesis.plotting import (
    FIGURE_STYLE,
    MODEL_COLOR,
    OBSERVED_COLOR,
    REFERENCE_COLOR,
    hierarchical_interval,
    peak_axes,
    pooled_points,
    pooled_strip,
    save_figure,
)

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
# The matched peak (`peaks.py`): the observed rate and the model rate.
RATES = ("observed", "model")
RATE_COLORS = (OBSERVED_COLOR, MODEL_COLOR)
# The gap peaks are small and often at or below 0, so their axes are linear.
GAP_LIMITS = (-15, 25)
PEAK_LIMITS = GAP_LIMITS


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
            dmat.FLASH_COMMON,
            dmat.FLASH_JUST_BEFORE,
            dmat.FLASH_AFTER_EXIT,
            dmat.FLASH_MOVE_GAP,
        )
    }
    # Only the terms that the design in use has.
    columns = {term: index for term, index in columns.items() if index}

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


def gap_flashes(trials) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return the trial rows and times of the flashes of each gap class, still.

    Flashes in the fixation whose window (`peaks.BASELINE_MS[0]` to
    `peaks.PEAK_MS[1]`) is in the fixation, except the first flash of a trial
    (it has no gap). The 80+ ms gap class first: it is the reference of the
    baseline.
    """
    before_s, after_s = -peaks.BASELINE_MS[0] / 1000, peaks.PEAK_MS[1] / 1000
    rows, times, short = [], [], []
    for row, (flashes, entry, leave) in enumerate(
        zip(
            trials["stim_pulse_times_s"],
            trials["center_entry_s"],
            trials["center_exit_s"],
            strict=True,
        )
    ):
        flashes = np.asarray(flashes, dtype=float)
        gaps = np.diff(flashes, prepend=-np.inf)
        keep = (flashes - before_s >= entry) & (flashes + after_s <= leave)
        keep &= np.isfinite(gaps)
        rows.append(np.full(int(keep.sum()), row))
        times.append(flashes[keep])
        short.append(gaps[keep] < dmat.short_gap_s())
    rows, times, short = (np.concatenate(values) for values in (rows, times, short))
    return [(rows[~short], times[~short]), (rows[short], times[short])]


def draw_curves(axes, pooled: dict) -> None:
    """Draw the median kernel of each kind of unit, by gap, over the pooled units."""
    index = {name: position for position, name in enumerate(CONDITIONS)}
    fixation_mean = _mean_of(pooled["response"], FIXATION)
    for axis, (kind, sign) in zip(
        axes, (("fire more", 1), ("fire less", -1)), strict=True
    ):
        color = KIND_COLORS[kind]
        members = sign * fixation_mean > KIND_LIMIT
        for name, shade in zip(FIXATION, (0.45, 1.0), strict=True):
            axis.plot(
                pooled["lags"],
                np.median(pooled["kernel"][members, index[name]], 0),
                color=color,
                alpha=shade,
            )
        axis.axhline(0, color=REFERENCE_COLOR, linestyle="--", linewidth=0.8)
        axis.set_xlabel("time from flash (ms)")
        axis.set_title(
            f"{kind} after a flash ({int(members.sum())} units)",
            color=color,
            fontsize=6.5,
            loc="left",
        )
    axes[0].set_ylabel(r"median flash kernel" + "\n" + r"($\Delta$ log rate)")
    for row, (text, shade) in enumerate((("40 ms gap", 0.45), ("80+ ms gap", 1.0))):
        axes[0].text(
            0.03,
            0.96 - 0.1 * row,
            text,
            color="black",
            alpha=shade,
            transform=axes[0].transAxes,
            va="top",
            fontsize=6,
        )


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
            axis.set_ylabel("unit, sorted by its\nmean response when still")
        axes.append(axis)
    return axes


def gap_peaks(out: Path, session: str, unit_set: str) -> dict:
    """Return the matched peaks of the excited units at the two gaps, still."""
    model = cli.PoissonGLM(root=out / session)
    trials = dmat._load_selected_trials(dmat._load_windows(model.windows))
    return peaks.session_peaks(out / session, unit_set, gap_flashes(trials))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("figures/glm"))
    parser.add_argument("--units", type=cli._unit_set, default="all")
    args = parser.parse_args()
    sessions = [session_kernels(args.out / session, args.units) for session in SESSIONS]
    matched = [gap_peaks(args.out, session, args.units) for session in SESSIONS]
    stats = {}
    for session, data, gap in zip(SESSIONS, sessions, matched, strict=True):
        change = gap["peaks"][:, :, 1] - gap["peaks"][:, :, 0]  # 40 minus 80+
        stats[session] = {
            "kernels": data["stats"],
            "matched peak (spikes/s), 40 ms minus 80+ ms gap, still": {
                "flashes": gap["events"],
                **{
                    rate: _paired(gap["peaks"][:, k, 0], gap["peaks"][:, k, 1])
                    for k, rate in enumerate(RATES)
                },
                "model minus observed change": _paired(change[:, 0], change[:, 1]),
                "units_outside_the_peak_axis": int(
                    np.sum(
                        (gap["peaks"][:, :2] < PEAK_LIMITS[0])
                        | (gap["peaks"][:, :2] > PEAK_LIMITS[1])
                    )
                ),
            },
        }
    stats["pooled, hierarchical bootstrap: 40 ms minus 80+ ms gap (spikes/s)"] = {
        rate: hierarchical_interval(
            [g["peaks"][:, k, 1] - g["peaks"][:, k, 0] for g in matched]
        )
        for k, rate in enumerate(RATES)
    }
    with plt.rc_context(FIGURE_STYLE):
        figure = plt.figure(figsize=(7.4, 8.4))
        outer = figure.add_gridspec(3, 1, height_ratios=(0.9, 1.6, 1.05), hspace=0.55)
        pooled = {
            "lags": sessions[0]["lags"],
            "kernel": np.concatenate([d["kernel"] for d in sessions]),
            "response": np.concatenate([d["response"] for d in sessions]),
        }
        curve_axes = outer[0].subgridspec(1, 2, wspace=0.25).subplots(sharey=True)
        draw_curves(curve_axes, pooled)
        map_axes = draw_maps(figure, outer[1], pooled, first=True)
        bar = figure.colorbar(
            map_axes[0].images[0], ax=map_axes, fraction=0.015, pad=0.015
        )
        bar.set_label(r"flash kernel ($\Delta$ log rate)")
        gap_axes = (
            outer[2].subgridspec(1, 3, wspace=0.55, width_ratios=(1, 1, 1.1)).subplots()
        )
        subjects = [gap["subject"] for gap in matched]
        for axis, k in zip(gap_axes[:2], range(2), strict=True):
            peak_axes(axis, GAP_LIMITS)
            pooled_points(
                axis,
                [g["peaks"][:, k, 0] for g in matched],
                [g["peaks"][:, k, 1] for g in matched],
                subjects,
                RATE_COLORS[k],
            )
            axis.set_xlabel(f"{RATES[k]} peak, 80+ ms gap\n(spikes/s)")
            axis.set_ylabel(f"{RATES[k]} peak,\n40 ms gap (spikes/s)")
        change_axis = gap_axes[2]
        for k in range(2):
            pooled_strip(
                change_axis,
                k,
                [g["peaks"][:, k, 1] - g["peaks"][:, k, 0] for g in matched],
                subjects,
                RATE_COLORS[k],
            )
        change_axis.axhline(0, color=REFERENCE_COLOR, lw=0.8)
        change_axis.set_xticks(range(2), list(RATES))
        for tick, color in zip(change_axis.get_xticklabels(), RATE_COLORS, strict=True):
            tick.set_color(color)
        change_axis.set_xlim(-0.5, 1.8)
        change_axis.set_ylabel("peak, 40 ms minus\n80+ ms gap (spikes/s)")
        change_axis.set_title(
            f"{sum(len(g['peaks']) for g in matched)} units, {len(matched)} mice",
            fontsize=6.5,
            loc="left",
        )
        figure.canvas.draw()
        to_figure = figure.transFigure.inverted()
        lettered = (*curve_axes, map_axes[0], *gap_axes)
        for letter, axis in zip("abcdef", lettered, strict=True):
            box = axis.yaxis.get_tightbbox()
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
    output = args.out / "flash_kernels.json"
    output.write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
