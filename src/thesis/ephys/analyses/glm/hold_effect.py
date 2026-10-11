"""Show the suppression of V1 firing while the animal holds with the stimulus on.

Question: V1 units fire less from about 80 ms after the first flash until the
animal leaves the center port. Does that follow the stimulus (adaptation to the
flashes) or the hold? The flashes go on from the first flash to the response
entry, so the time after the center exit has the stimulus without the hold.

Comparison: the hold (first flash to center exit) against the time from the
center exit to the response entry, in the same trials and the same units. The
comparison is paired: each unit has a value in both periods.

- Panel a (observation): raw firing rate, no model, against time from the first
  flash and from the center exit. The rate of a unit is divided by its mean
  rate over the valid bins; the line is the mean over units and the band is its
  SEM. The dashed line is the rate in the 100 ms before the first flash. The
  top row is the fraction of trials with a flash in each 40 ms.
- Panel b (characterization): the gain that a descriptive model gives to each
  time, exp(kernel). The model is the design of `dmat.py` with one change: the
  withdrawal kernel (0 to 100 ms) is replaced with a kernel from the center
  exit to the response entry (`AFTER_BASES` raised cosines with log spacing,
  `AFTER_WINDOW_S`). Both kernels are relative to the time before the first
  flash. Thin lines are units; the thick line and band are the median and the
  interquartile range.
- Panel c (the paired comparison): each unit's mean gain over `LEVEL_MS` of
  the hold against its mean gain over the same time after the center exit.
  Error bars are the SD over the 10 cross-validation folds, in both directions.
  Two-sided Wilcoxon signed-rank test of the log gains.
- Panel d (control for stillness): raw rate against motion energy in the
  center port and at a side port (`port_vs_stillness.py`).

The descriptive model is fit here (`glm fit` stages, with cross-validation) in
the directory with the suffix `ROOT_SUFFIX` next to the session directory. It
is not the model of the main figure: the longer kernel after the exit does not
predict reliably better than the withdrawal kernel (`docs/glm.md`).

Units, trials, alignment (first flash), baseline, and exclusions are those of
`fit.py`. Bins after the response entry are excluded. The sampling unit is the
unit. The script writes `hold_effect.svg` and `hold_effect_stats.json` in the
session directory.

    uv run python -m thesis.ephys.analyses.glm.hold_effect --units all

The fit of all units takes about 20 minutes; a fit that is there is not run
again. Run `port_vs_stillness` first.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem, wilcoxon

from thesis.ephys.analyses.glm import cli, dmat, port_vs_stillness
from thesis.ephys.analyses.glm.adaptation_grid import INPUTS
from thesis.ephys.analyses.glm.fit import (
    analysis_unit_indices,
    build_unit_counts,
    crossvalidate,
    fit_models,
    load_unit_records,
    refit_final,
)
from thesis.ephys.units import fetch_unit_table
from thesis.plotting import (
    CONCEPT_COLORS,
    FIGURE_STYLE,
    OBSERVED_COLOR,
    REFERENCE_COLOR,
    save_figure,
    separate_axes,
)

ROOT_SUFFIX = "_after_exit_kernel"
AFTER = "after_exit"
AFTER_WINDOW_S = 0.6
AFTER_BASES = 8
LEVEL_MS = (150, 500)
RATE_BIN_MS = 20
FLASH_SLOT_S = 0.04
FLASH_WINDOW_MS = (-100, 500)
EXIT_WINDOW_MS = (-200, 500)
TASK, VISUAL = CONCEPT_COLORS["task"], CONCEPT_COLORS["visual"]


def after_exit_basis() -> dmat.StretchedCosineBasis:
    return dmat.StretchedCosineBasis(
        AFTER_BASES,
        AFTER_WINDOW_S + dmat.BINWIDTH_S,
        dmat.BINWIDTH_S,
        dmat.HOLD_STRETCH_S,
    )


def fit_descriptive(source: Path, unit_set: str) -> Path:
    """Fit the design with a kernel from the center exit to the response entry."""
    root = source.with_name(source.name + ROOT_SUFFIX)
    model = cli.PoissonGLM(root=root)
    directory = model.fit_dir(unit_set)
    root.mkdir(exist_ok=True)
    for name in INPUTS:
        if not (root / name).exists():
            shutil.copy(source / name, root / name)
    basis = after_exit_basis().basis
    original = dmat.build_task_design

    def with_after_kernel(alignments, trials):
        values, manifest = original(alignments, trials)
        centers, _, _ = dmat.construct_timebins(
            dmat.PRE_S, dmat.POST_S, dmat.BINWIDTH_S
        )
        leave = trials["center_exit_s"].to_numpy(dtype=float)
        since = centers[None, :] - (leave - np.asarray(alignments))[:, None]
        limit = trials["response_port_entry_s"].to_numpy(dtype=float) - leave
        index = np.rint(since / dmat.BINWIDTH_S).astype(int)
        inside = (index >= 0) & (index < len(basis)) & (since < limit[:, None])
        block = np.where(
            inside[..., None], basis[np.clip(index, 0, len(basis) - 1)], 0.0
        )
        manifest.append(
            {
                "name": AFTER,
                "group": "task",
                "events": len(alignments),
                "columns": basis.shape[1],
                "coding": "kernel on the center exit, zero from the response entry on",
            }
        )
        return np.column_stack((values, block.reshape(-1, basis.shape[1]))), manifest

    # ponytail: the design reads module attributes; set them for this fit and
    # restore them. Make them arguments if the design gets more callers.
    dmat.build_task_design, dmat.WITHDRAWAL_KERNEL = with_after_kernel, False
    try:
        if not model.design.exists():
            dmat.prepare_common_design(model.windows, model.video, model.design)
        if not (directory / "cross_validation_summary.json").exists():
            fit_models(model.windows, model.design, unit_set, directory)
            summary = json.loads((directory / "validation_summary.json").read_text())
            components = int(summary["selected_video_components"])
            crossvalidate(model.windows, model.design, unit_set, directory, components)
            refit_final(model.windows, model.design, unit_set, directory, components)
    finally:
        dmat.build_task_design, dmat.WITHDRAWAL_KERNEL = original, True
    return root


def kernel_gains(root: Path, unit_set: str) -> dict:
    """Return exp(kernel) of the hold and after-exit kernels: final refit and folds."""
    model = cli.PoissonGLM(root=root)
    metadata = json.loads(model.design.with_suffix(".json").read_text())
    names = metadata["column_names"]
    scale = np.asarray(metadata["training_scale"])
    checkpoints = model.fit_dir(unit_set) / "checkpoints"
    final = load_unit_records(checkpoints, "unit_*_final.json")
    folds = {
        item["unit_id"]: item
        for item in load_unit_records(checkpoints, "unit_*_folds.json")
    }
    out = {"unit_ids": np.asarray([item["unit_id"] for item in final])}
    for name, basis in (
        (dmat.HOLD, dmat.task_temporal_bases()[dmat.HOLD]),
        (AFTER, after_exit_basis()),
    ):
        columns = [i for i, column in enumerate(names) if column.startswith(name)]
        times = 1000 * basis.basis_time
        level = (times >= LEVEL_MS[0]) & (times < LEVEL_MS[1])
        gain = lambda weights: np.exp(  # noqa: E731
            (np.asarray(weights)[columns] / scale[columns]) @ basis.basis.T
        )
        curves = np.asarray([gain(item["coefficients"]) for item in final])
        fold_levels = [
            [
                gain(fold["coefficients"])[level].mean()
                for fold in folds[item["unit_id"]]["folds"]
            ]
            for item in final
        ]
        out[name] = {
            "times": times,
            "gain": curves,
            "level": curves[:, level].mean(axis=1),
            "level_sd": np.asarray([np.std(values) for values in fold_levels]),
        }
    return out


def raw_rates(source: Path, unit_set: str) -> dict:
    """Return each unit's normalized rate from the first flash and from the center exit."""
    prepared = dmat._load_windows(source / "stimulus_windows_me.npz")
    trials = dmat._load_selected_trials(prepared)
    alignments, relative = prepared["alignments"], prepared["relative_centers"]
    grid = (len(alignments), len(relative))
    valid = dmat._valid_bin_mask(prepared).reshape(grid)
    leave = trials["center_exit_s"].to_numpy(dtype=float) - alignments
    step = RATE_BIN_MS  # 1-ms bins per rate bin

    def window(events, limits_ms, keep):
        """Index of each 1-ms bin around the events, and the bins that count."""
        fine = (np.arange(*limits_ms) + 0.5) / 1000
        columns = np.rint(
            (events[:, None] + fine[None, :] - relative[0]) / dmat.BINWIDTH_S
        ).astype(int)
        inside = (columns >= 0) & (columns < grid[1])
        columns = np.clip(columns, 0, grid[1] - 1)
        rows = np.arange(grid[0])[:, None]
        used = inside & valid[rows, columns] & keep(relative[columns])
        centers = np.arange(*limits_ms, step) + step / 2
        return rows, columns, used, centers

    in_hold = lambda times: times < leave[:, None]  # noqa: E731
    after_flash = lambda times: times >= 0  # noqa: E731
    windows = {
        "flash": window(np.zeros(grid[0]), FLASH_WINDOW_MS, in_hold),
        "exit": window(leave, EXIT_WINDOW_MS, after_flash),
    }
    units = fetch_unit_table(
        prepared["metadata"]["subject_name"],
        prepared["metadata"]["session_name"],
        unit_criteria_id=1,
        stability_param_id=0,
        include_metrics=False,
    )
    units = units.iloc[analysis_unit_indices(units, unit_set)]
    rates = {name: [] for name in windows}
    for _, unit in units.iterrows():
        spikes = np.asarray(unit["spike_times_s"], dtype=float)
        counts = build_unit_counts(alignments, spikes).reshape(grid)
        mean_count = counts[valid].mean()
        if mean_count <= 0:
            continue
        for name, (rows, columns, used, _) in windows.items():
            shape = (grid[0], -1, step)
            spikes_in = (counts[rows, columns] * used).reshape(shape).sum((0, 2))
            exposure = used.reshape(shape).sum((0, 2))
            rates[name].append(spikes_in / np.maximum(exposure, 1) / mean_count)
    flashes = [
        np.asarray(times, dtype=float) - first
        for times, first in zip(trials["stim_pulse_times_s"], alignments, strict=True)
    ]

    def flash_fraction(events, limits_ms):
        edges = np.arange(limits_ms[0] / 1000, limits_ms[1] / 1000 + 1e-9, FLASH_SLOT_S)
        hits = [
            np.histogram(times - event, bins=edges)[0] > 0
            for times, event in zip(flashes, events, strict=True)
        ]
        return 1000 * edges[:-1], np.mean(hits, axis=0)

    return {
        "flash": (windows["flash"][3], np.asarray(rates["flash"])),
        "exit": (windows["exit"][3], np.asarray(rates["exit"])),
        "flash_fraction": flash_fraction(np.zeros(grid[0]), FLASH_WINDOW_MS),
        "exit_fraction": flash_fraction(leave, EXIT_WINDOW_MS),
        "median_exit_ms": float(1000 * np.median(leave)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=cli.PoissonGLM().root)
    parser.add_argument("--units", type=cli._unit_set, default="all")
    args = parser.parse_args()
    stillness_path = args.root / "port_vs_stillness.npz"
    if not stillness_path.exists():
        raise FileNotFoundError(f"{stillness_path}: run `port_vs_stillness` first.")
    with np.load(stillness_path, allow_pickle=False) as data:
        stillness = {key: data[key] for key in data.files}
    gains = kernel_gains(fit_descriptive(args.root, args.units), args.units)
    rates = raw_rates(args.root, args.units)
    hold, after = gains[dmat.HOLD], gains[AFTER]
    before = float(rates["flash"][1][:, rates["flash"][0] < 0].mean())

    with plt.rc_context(FIGURE_STYLE):
        figure = plt.figure(figsize=(7.4, 7.0))
        top = figure.add_gridspec(
            2, 2, left=0.11, right=0.97, top=0.95, bottom=0.71, height_ratios=(1, 4)
        )
        middle = figure.add_gridspec(1, 2, left=0.11, right=0.97, top=0.6, bottom=0.4)
        bottom = figure.add_gridspec(
            1, 2, left=0.11, right=0.86, top=0.29, bottom=0.07, wspace=0.35
        )
        panel_axes = []
        shared = None
        for column, (name, xlabel) in enumerate(
            (("flash", "time from first flash (ms)"), ("exit", "time from center exit"))
        ):
            slots, fraction = rates[f"{name}_fraction"]
            flash_axis = figure.add_subplot(top[0, column])
            flash_axis.step(slots, fraction, where="post", color=VISUAL)
            flash_axis.set_ylim(0, 1.05)
            flash_axis.tick_params(labelbottom=False)
            axis = figure.add_subplot(top[1, column], sharex=flash_axis, sharey=shared)
            shared = shared or axis
            centers, values = rates[name]
            mean, error = values.mean(axis=0), sem(values, axis=0)
            axis.fill_between(
                centers, mean - error, mean + error, color=OBSERVED_COLOR, alpha=0.25
            )
            axis.plot(centers, mean, color=OBSERVED_COLOR)
            axis.axhline(before, color=REFERENCE_COLOR, linestyle="--", linewidth=0.8)
            axis.set_xlabel(xlabel + (" (ms)" if name == "exit" else ""))
            if column == 0:
                flash_axis.text(
                    0.3,
                    0.75,
                    "fraction of trials with a flash",
                    color=VISUAL,
                    transform=flash_axis.transAxes,
                    fontsize=6.5,
                )
                axis.set_ylabel("normalized firing rate")
                axis.text(
                    0.98,
                    before,
                    "before the first flash",
                    transform=axis.get_yaxis_transform(),
                    ha="right",
                    va="bottom",
                    fontsize=6.5,
                )
                panel_axes.append(flash_axis)
            else:
                flash_axis.tick_params(labelleft=False)
                axis.tick_params(labelleft=False)

            gain_axis = figure.add_subplot(
                middle[column], sharey=panel_axes[1] if column else None
            )
            kernel = hold if name == "flash" else after
            shown = (
                kernel["times"]
                <= (FLASH_WINDOW_MS if column == 0 else EXIT_WINDOW_MS)[1]
            )
            times = kernel["times"][shown]
            gain_axis.plot(
                times, kernel["gain"][:, shown].T, color="0.8", linewidth=0.3
            )
            low, median, high = np.percentile(
                kernel["gain"][:, shown], [25, 50, 75], axis=0
            )
            gain_axis.fill_between(times, low, high, color=TASK, alpha=0.3, linewidth=0)
            gain_axis.plot(times, median, color=TASK, linewidth=1.6)
            gain_axis.axhline(1, color=REFERENCE_COLOR, linestyle="--", linewidth=0.8)
            gain_axis.set_xlim(axis.get_xlim())
            gain_axis.set_xlabel(xlabel + (" (ms)" if name == "exit" else ""))
            if column == 0:
                gain_axis.set_yscale("log")
                gain_axis.set_ylim(0.2, 5)
                gain_axis.set_yticks(
                    [0.25, 0.5, 1, 2, 4], ["0.25", "0.5", "1", "2", "4"]
                )
                gain_axis.minorticks_off()
                gain_axis.set_ylabel("gain (model)")
                gain_axis.text(
                    0.98,
                    0.96,
                    "units",
                    color="0.55",
                    transform=gain_axis.transAxes,
                    ha="right",
                    va="top",
                )
                gain_axis.text(
                    0.98,
                    0.86,
                    "median and IQR",
                    color=TASK,
                    transform=gain_axis.transAxes,
                    ha="right",
                    va="top",
                )
                panel_axes.append(gain_axis)
            else:
                gain_axis.tick_params(labelleft=False)

        paired = figure.add_subplot(bottom[0])
        paired.errorbar(
            hold["level"],
            after["level"],
            xerr=hold["level_sd"],
            yerr=after["level_sd"],
            fmt="o",
            color=TASK,
            markersize=2.5,
            elinewidth=0.4,
            alpha=0.7,
        )
        limits = (0.15, 6)
        paired.plot(
            limits, limits, color=REFERENCE_COLOR, linestyle="--", linewidth=0.8
        )
        for line in (paired.axhline, paired.axvline):
            line(1, color=REFERENCE_COLOR, linewidth=0.5)
        paired.set_xscale("log")
        paired.set_yscale("log")
        paired.set_xlim(limits)
        paired.set_ylim(limits)
        ticks = [0.25, 0.5, 1, 2, 4]
        paired.set_xticks(ticks, [f"{tick:g}" for tick in ticks])
        paired.set_yticks(ticks, [f"{tick:g}" for tick in ticks])
        paired.minorticks_off()
        paired.set_aspect("equal")
        window = f"{LEVEL_MS[0]} to {LEVEL_MS[1]} ms"
        paired.set_xlabel(f"gain in the hold,\n{window} from first flash")
        paired.set_ylabel(f"gain after the center exit,\n{window} from exit")
        stillness_axis = figure.add_subplot(bottom[1])
        port_vs_stillness.plot_rates(stillness_axis, stillness, states=(0, 2))
        panel_axes += [paired, stillness_axis]
        for axis in panel_axes[1:]:
            separate_axes(axis)
        figure.canvas.draw()
        to_figure = figure.transFigure.inverted()
        for letter, axis in zip("abcd", panel_axes, strict=True):
            left = axis.yaxis.get_tightbbox().transformed(to_figure).x0
            figure.text(
                left,
                axis.get_position().y1 + 0.015,
                letter,
                fontweight="bold",
                fontsize=10,
            )
        output = save_figure(figure, args.root / "hold_effect")

    outside = int(
        np.sum(
            (hold["level"] < limits[0])
            | (hold["level"] > limits[1])
            | (after["level"] < limits[0])
            | (after["level"] > limits[1])
        )
    )
    centers, values = rates["exit"]
    stats = {
        "units": len(hold["level"]),
        "rate_before_first_flash": before,
        "rate_last_200_ms_of_hold": float(values[:, centers < 0].mean()),
        "rate_100_to_400_ms_after_exit": float(
            values[:, (centers > 100) & (centers < 400)].mean()
        ),
        "median_gain_hold": float(np.median(hold["level"])),
        "median_gain_after_exit": float(np.median(after["level"])),
        "fraction_units_hold_below_1": float(np.mean(hold["level"] < 1)),
        "fraction_units_after_exit_below_1": float(np.mean(after["level"] < 1)),
        "units_gain_higher_after_exit": int(np.sum(after["level"] > hold["level"])),
        "wilcoxon_p": float(
            wilcoxon(np.log(after["level"]), np.log(hold["level"])).pvalue
        ),
        "units_outside_the_axis_limits_of_panel_c": outside,
        "median_center_exit_ms_from_first_flash": rates["median_exit_ms"],
    }
    output.with_name("hold_effect_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))
    print(output)


if __name__ == "__main__":
    main()
