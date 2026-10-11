"""Show the suppression of V1 firing while the animal holds with the stimulus on.

Question: V1 units fire less from about 80 ms after the first flash until the
animal leaves the center port. Does that follow the stimulus (adaptation to the
flashes) or the hold? The flashes go on from the first flash to the response
entry, so the time after the center exit has the stimulus without the hold.

Comparison: the hold (first flash to center exit) against the time from the
center exit to the response entry, in the same trials and the same units. The
comparison is paired: each unit has a value in both periods.

- Panel a (examples): spikes of two units against time from the first flash,
  with the trials sorted by the length of the hold. Left: the example unit of
  the main figure. Right: the unit with the highest hold level among the units
  that fire at `EXAMPLE_MIN_RATE` or more.
- Panel b (observation): raw firing rate, no model, against time from the first
  flash and from the center exit. The rate of a unit is divided by its mean
  rate over the valid bins; the line is the mean over units and the band is its
  SEM. The dashed line is the rate in the 100 ms before the first flash. The
  top row is the fraction of trials with a flash in each 40 ms.
- Panel c (characterization): the kernels of a descriptive model, as a change
  in log rate. The model is the design of `dmat.py` with one change: the
  withdrawal kernel (0 to 100 ms) is replaced with a kernel from the center
  exit to the response entry (`AFTER_BASES` raised cosines, `AFTER_WINDOW_S`).
  Both kernels are relative to the time before the first flash. Thin lines are
  units; the thick line and band are the median and the interquartile range.
- Panel d (the paired comparison): each unit's mean kernel value over
  `LEVEL_MS` of the hold against its mean over the same time after the center
  exit. Error bars are the SD over the 10 cross-validation folds, in both
  directions. Two-sided Wilcoxon signed-rank test of the two values. Points
  outside `LEVEL_LIMITS` are not drawn; they are in the statistics, and their
  number is in `hold_effect_stats.json`.
- Panel e (control for the order of fitting): does the hold kernel take
  variance from the other terms? Each unit's hold level in the main model (all
  terms fit together) against its level in a two-step fit: the design with no
  hold kernel first, then the hold kernel alone on top of that fixed
  prediction (`two_step_levels`). The two-step value is a lower bound. Both are
  in-sample.
- Panel f (control for stillness): raw rate against motion energy in the
  center port and at a side port (`port_vs_stillness.py`).

The descriptive model and the design with no hold kernel are fit here (`glm
fit` stages, with cross-validation) in the directories with the suffixes
`ROOT_SUFFIX` and `NO_HOLD_SUFFIX` next to the session directory. It
is not the model of the main figure: the longer kernel after the exit does not
predict reliably better than the withdrawal kernel (`docs/glm.md`).

Units, trials, alignment (first flash), baseline, and exclusions are those of
`fit.py`. Bins after the response entry are excluded. The sampling unit is the
unit. The script writes `hold_effect.svg` and `hold_effect_stats.json` in the
session directory.

    uv run python -m thesis.ephys.analyses.glm.hold_effect --units all

Each fit of all units takes about 20 minutes; a fit that is there is not run
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
from sklearn.linear_model import PoissonRegressor

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
from thesis.ephys.analyses.glm.results import EXAMPLE_UNIT_ID
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
NO_HOLD_SUFFIX = "_no_hold_term"
# The excited example unit must fire at this rate or more (spikes/s).
EXAMPLE_MIN_RATE = 2.0
EXAMPLE_WINDOW_MS = (-100, 900)
AFTER = "after_exit"
AFTER_WINDOW_S = 0.6
AFTER_BASES = 8
LEVEL_MS = (150, 500)
# Axis limits of the kernel panels (Δ log rate).
LEVEL_LIMITS = (-1.6, 1.6)
RATE_BIN_MS = 20
FLASH_SLOT_S = 0.04
FLASH_WINDOW_MS = (-100, 500)
EXIT_WINDOW_MS = (-200, 500)
TASK, VISUAL = CONCEPT_COLORS["task"], CONCEPT_COLORS["visual"]


def after_exit_basis() -> dmat.RaisedCosineBasis:
    return dmat.RaisedCosineBasis(
        AFTER_BASES, 0, AFTER_WINDOW_S + dmat.BINWIDTH_S, dmat.BINWIDTH_S
    )


def _fit_variant(source: Path, unit_set: str, suffix: str, attributes: dict) -> Path:
    """Fit the design with some `dmat` attributes replaced, in a sibling directory."""
    root = source.with_name(source.name + suffix)
    model = cli.PoissonGLM(root=root)
    directory = model.fit_dir(unit_set)
    root.mkdir(exist_ok=True)
    for name in INPUTS:
        if not (root / name).exists():
            shutil.copy(source / name, root / name)
    # ponytail: the design reads module attributes; set them for this fit and
    # restore them. Make them arguments if the design gets more callers.
    saved = {name: getattr(dmat, name) for name in attributes}
    for name, value in attributes.items():
        setattr(dmat, name, value)
    try:
        if not model.design.exists():
            dmat.prepare_common_design(model.windows, model.video, model.design)
        # Each stage keeps its finished units, so a stopped fit goes on.
        checkpoints = directory / "checkpoints"
        if not list(checkpoints.glob("unit_*_final.json")) or len(
            list(checkpoints.glob("unit_*_final.json"))
        ) < len(list(checkpoints.glob("unit_*_validation.json"))):
            fit_models(model.windows, model.design, unit_set, directory)
            summary = json.loads((directory / "validation_summary.json").read_text())
            components = int(summary["selected_video_components"])
            crossvalidate(model.windows, model.design, unit_set, directory, components)
            refit_final(model.windows, model.design, unit_set, directory, components)
    finally:
        for name, value in saved.items():
            setattr(dmat, name, value)
    return root


def _time_in_trial() -> np.ndarray:
    centers, _, _ = dmat.construct_timebins(dmat.PRE_S, dmat.POST_S, dmat.BINWIDTH_S)
    return centers


def _kernel_block(basis: np.ndarray, since: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Place the basis rows at each bin's time since an event; zero where not kept."""
    index = np.rint(since / dmat.BINWIDTH_S).astype(int)
    inside = (index >= 0) & (index < len(basis)) & keep
    block = np.where(inside[..., None], basis[np.clip(index, 0, len(basis) - 1)], 0.0)
    return block.reshape(-1, basis.shape[1])


def fit_descriptive(source: Path, unit_set: str) -> Path:
    """Fit the design with a kernel from the center exit to the response entry."""
    basis = after_exit_basis().basis
    original = dmat.build_task_design

    def with_after_kernel(alignments, trials):
        values, manifest = original(alignments, trials)
        leave = trials["center_exit_s"].to_numpy(dtype=float)
        since = _time_in_trial()[None, :] - (leave - np.asarray(alignments))[:, None]
        limit = trials["response_port_entry_s"].to_numpy(dtype=float) - leave
        manifest.append(
            {
                "name": AFTER,
                "group": "task",
                "events": len(alignments),
                "columns": basis.shape[1],
                "coding": "kernel on the center exit, zero from the response entry on",
            }
        )
        block = _kernel_block(basis, since, since < limit[:, None])
        return np.column_stack((values, block)), manifest

    return _fit_variant(
        source,
        unit_set,
        ROOT_SUFFIX,
        {"build_task_design": with_after_kernel, "WITHDRAWAL_KERNEL": False},
    )


def two_step_levels(source: Path, unit_set: str) -> dict:
    """Return each unit's hold level from the joint fit and from a two-step fit.

    Step 1 is the design with no hold kernel. Step 2 fits the hold kernel alone
    on top of the fixed prediction of step 1 (Poisson, no intercept, no
    penalty), so the other terms keep all that they can explain. Both fits use
    all valid bins; the values are in-sample.
    """
    cache = source / f"hold_two_step_{unit_set}.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}
    first = cli.PoissonGLM(
        root=_fit_variant(source, unit_set, NO_HOLD_SUFFIX, {"HOLD_KERNEL": False})
    )
    joint = cli.PoissonGLM(root=source)
    prepared = dmat._load_windows(joint.windows)
    trials = dmat._load_selected_trials(prepared)
    alignments = prepared["alignments"]
    valid = dmat._valid_bin_mask(prepared)
    basis = dmat.task_temporal_bases()[dmat.HOLD]
    times = 1000 * basis.basis_time
    level = (times >= LEVEL_MS[0]) & (times < LEVEL_MS[1])
    leave = trials["center_exit_s"].to_numpy(dtype=float) - alignments
    since = np.broadcast_to(_time_in_trial(), (len(alignments), len(_time_in_trial())))
    hold_columns = _kernel_block(basis.basis, since, since < leave[:, None])[valid]
    metadata = json.loads(first.design.with_suffix(".json").read_text())
    common = np.load(first.design, mmap_mode="r", allow_pickle=False)
    joint_metadata = json.loads(joint.design.with_suffix(".json").read_text())
    columns = [
        index
        for index, name in enumerate(joint_metadata["column_names"])
        if name.startswith(f"{dmat.HOLD}_basis_")
    ]
    scale = np.asarray(joint_metadata["training_scale"])[columns]
    joint_records = {
        item["unit_id"]: item
        for item in load_unit_records(
            joint.fit_dir(unit_set) / "checkpoints", "unit_*_final.json"
        )
    }
    units = fetch_unit_table(
        prepared["metadata"]["subject_name"],
        prepared["metadata"]["session_name"],
        unit_criteria_id=1,
        stability_param_id=0,
        include_metrics=False,
    ).set_index("unit_id")
    out = {"unit_ids": [], "joint": [], "two_step": [], "rate": []}
    for record in load_unit_records(
        first.fit_dir(unit_set) / "checkpoints", "unit_*_final.json"
    ):
        unit_id = record["unit_id"]
        spikes = np.asarray(units.loc[unit_id, "spike_times_s"], dtype=float)
        counts = build_unit_counts(alignments, spikes).astype(float)[valid]
        history, _, _ = dmat.training_zscore(
            dmat.build_unit_history(alignments, spikes), valid
        )
        width = int(metadata["base_columns"]) + int(record["components"])
        weights = np.asarray(record["coefficients"], dtype=float)
        predicted = np.exp(
            record["intercept"]
            + np.asarray(common[:, :width])[valid] @ weights[:width]
            + history[valid] @ weights[width:]
        )
        second = PoissonRegressor(alpha=0.0, fit_intercept=False, max_iter=1000).fit(
            hold_columns, counts / predicted, sample_weight=predicted
        )
        joint_weights = np.asarray(joint_records[unit_id]["coefficients"])[columns]
        out["unit_ids"].append(unit_id)
        out["rate"].append(joint_records[unit_id]["fit_mean_count"] / dmat.BINWIDTH_S)
        out["two_step"].append((second.coef_ @ basis.basis.T)[level].mean())
        out["joint"].append(((joint_weights / scale) @ basis.basis.T)[level].mean())
        print(f"Two-step fit of unit {unit_id}.", flush=True)
    out = {key: np.asarray(value) for key, value in out.items()}
    np.savez(cache, **out)
    return out


def kernel_values(root: Path, unit_set: str) -> dict:
    """Return the hold and after-exit kernels (Δ log rate): final refit and folds."""
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
        values_of = lambda weights: (  # noqa: E731
            (np.asarray(weights)[columns] / scale[columns]) @ basis.basis.T
        )
        curves = np.asarray([values_of(item["coefficients"]) for item in final])
        fold_levels = [
            [
                values_of(fold["coefficients"])[level].mean()
                for fold in folds[item["unit_id"]]["folds"]
            ]
            for item in final
        ]
        out[name] = {
            "times": times,
            "values": curves,
            "level": curves[:, level].mean(axis=1),
            "level_sd": np.asarray([np.std(values) for values in fold_levels]),
        }
    return out


def raw_rates(source: Path, unit_set: str, examples: tuple[int, ...]) -> dict:
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
    shown = (relative >= EXAMPLE_WINDOW_MS[0] / 1000) & (
        relative < EXAMPLE_WINDOW_MS[1] / 1000
    )
    order = np.argsort(leave)
    rasters = {}
    for _, unit in units.iterrows():
        spikes = np.asarray(unit["spike_times_s"], dtype=float)
        counts = build_unit_counts(alignments, spikes).reshape(grid)
        mean_count = counts[valid].mean()
        if mean_count <= 0:
            continue
        if int(unit["unit_id"]) in examples:
            rows, bins = np.nonzero(counts[order][:, shown] > 0)
            rasters[int(unit["unit_id"])] = (1000 * relative[shown][bins], rows)
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
        "rasters": rasters,
        "sorted_exit_ms": 1000 * leave[order],
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
    kernels = kernel_values(fit_descriptive(args.root, args.units), args.units)
    steps = two_step_levels(args.root, args.units)
    # Example units: the example of the main figure (the most suppressed unit
    # if it was not fit), and the most excited unit that fires enough.
    ids, joint = steps["unit_ids"], steps["joint"]
    active = steps["rate"] >= EXAMPLE_MIN_RATE
    suppressed = (
        EXAMPLE_UNIT_ID
        if EXAMPLE_UNIT_ID in ids
        else int(ids[active][np.argmin(joint[active])])
    )
    excited = int(ids[active][np.argmax(joint[active])])
    rates = raw_rates(args.root, args.units, (suppressed, excited))
    hold, after = kernels[dmat.HOLD], kernels[AFTER]
    before = float(rates["flash"][1][:, rates["flash"][0] < 0].mean())

    with plt.rc_context(FIGURE_STYLE):
        figure = plt.figure(figsize=(7.4, 9.2))
        examples = figure.add_gridspec(
            1, 2, left=0.11, right=0.97, top=0.975, bottom=0.87
        )
        top = figure.add_gridspec(
            2, 2, left=0.11, right=0.97, top=0.795, bottom=0.615, height_ratios=(1, 4)
        )
        middle = figure.add_gridspec(
            1, 2, left=0.11, right=0.97, top=0.53, bottom=0.375
        )
        bottom = figure.add_gridspec(
            1, 3, left=0.11, right=0.88, top=0.265, bottom=0.06, wspace=0.75
        )
        panel_axes = []
        exits = rates["sorted_exit_ms"]
        for column, (unit_id, label) in enumerate(
            ((suppressed, "suppressed"), (excited, "excited"))
        ):
            axis = figure.add_subplot(examples[column])
            spike_ms, rows = rates["rasters"][unit_id]
            axis.plot(spike_ms, rows, "|", color=OBSERVED_COLOR, markersize=1.5)
            axis.plot(exits, np.arange(len(exits)), color=TASK, linewidth=1)
            axis.set_xlim(*EXAMPLE_WINDOW_MS)
            axis.set_xlabel("time from first flash (ms)")
            axis.text(
                0.99,
                1.02,
                f"unit {unit_id}, {label}",
                transform=axis.transAxes,
                ha="right",
                va="bottom",
                fontsize=6.5,
            )
            if column == 0:
                axis.set_ylabel("trial, sorted\nby hold length")
                axis.text(
                    exits[0] - 15,
                    0,
                    "center exit",
                    color=TASK,
                    ha="right",
                    va="bottom",
                    fontsize=6.5,
                )
                panel_axes.append(axis)
            else:
                axis.tick_params(labelleft=False)
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

            kernel_axis = figure.add_subplot(
                middle[column], sharey=panel_axes[-1] if column else None
            )
            kernel = hold if name == "flash" else after
            shown = (
                kernel["times"]
                <= (FLASH_WINDOW_MS if column == 0 else EXIT_WINDOW_MS)[1]
            )
            times = kernel["times"][shown]
            kernel_axis.plot(
                times,
                kernel["values"][:, shown].T,
                color="0.6",
                linewidth=0.3,
                alpha=0.3,
                zorder=1,
            )
            low, median, high = np.percentile(
                kernel["values"][:, shown], [25, 50, 75], axis=0
            )
            kernel_axis.fill_between(
                times, low, high, color=TASK, alpha=0.45, linewidth=0, zorder=2
            )
            kernel_axis.plot(times, median, color=TASK, linewidth=1.6, zorder=3)
            kernel_axis.axhline(0, color=REFERENCE_COLOR, linestyle="--", linewidth=0.8)
            kernel_axis.set_xlim(axis.get_xlim())
            kernel_axis.set_xlabel(xlabel + (" (ms)" if name == "exit" else ""))
            if column == 0:
                kernel_axis.set_ylim(*LEVEL_LIMITS)
                kernel_axis.set_ylabel(r"$\Delta$ log rate")
                kernel_axis.text(
                    0.98,
                    0.96,
                    "units",
                    color="0.55",
                    transform=kernel_axis.transAxes,
                    ha="right",
                    va="top",
                )
                kernel_axis.text(
                    0.98,
                    0.86,
                    "median and IQR",
                    color=TASK,
                    transform=kernel_axis.transAxes,
                    ha="right",
                    va="top",
                )
                panel_axes.append(kernel_axis)
            else:
                kernel_axis.tick_params(labelleft=False)

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
        limits = LEVEL_LIMITS
        paired.plot(
            limits, limits, color=REFERENCE_COLOR, linestyle="--", linewidth=0.8
        )
        for line in (paired.axhline, paired.axvline):
            line(0, color=REFERENCE_COLOR, linewidth=0.5)
        paired.set_xlim(limits)
        paired.set_ylim(limits)
        paired.set_aspect("equal")
        window = f"{LEVEL_MS[0]} to {LEVEL_MS[1]} ms"
        paired.set_xlabel(
            r"$\Delta$ log rate in the hold," + f"\n{window} from first flash"
        )
        paired.set_ylabel(r"$\Delta$ log rate after exit," + f"\n{window} from exit")
        order_axis = figure.add_subplot(bottom[1])
        order_axis.plot(
            steps["joint"],
            steps["two_step"],
            "o",
            color=TASK,
            markersize=2.5,
            alpha=0.7,
        )
        order_axis.plot(
            limits, limits, color=REFERENCE_COLOR, linestyle="--", linewidth=0.8
        )
        for line in (order_axis.axhline, order_axis.axvline):
            line(0, color=REFERENCE_COLOR, linewidth=0.5)
        order_axis.set_xlim(limits)
        order_axis.set_ylim(limits)
        order_axis.set_aspect("equal")
        order_axis.set_xlabel(
            r"$\Delta$ log rate in the hold," + "\nall terms fit together"
        )
        order_axis.set_ylabel(
            r"$\Delta$ log rate in the hold," + "\nhold kernel fit last"
        )
        stillness_axis = figure.add_subplot(bottom[2])
        port_vs_stillness.plot_rates(stillness_axis, stillness, states=(0, 2))
        panel_axes += [paired, order_axis, stillness_axis]
        for axis in panel_axes[2:]:
            separate_axes(axis)
        figure.canvas.draw()
        to_figure = figure.transFigure.inverted()
        for letter, axis in zip("abcdef", panel_axes, strict=True):
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
        "median_delta_log_rate_hold": float(np.median(hold["level"])),
        "median_delta_log_rate_after_exit": float(np.median(after["level"])),
        "fraction_units_hold_below_0": float(np.mean(hold["level"] < 0)),
        "fraction_units_after_exit_below_0": float(np.mean(after["level"] < 0)),
        "units_higher_after_exit": int(np.sum(after["level"] > hold["level"])),
        "wilcoxon_p": float(wilcoxon(after["level"], hold["level"]).pvalue),
        "units_outside_the_axis_limits_of_panel_d": outside,
        "example_units_suppressed_and_excited": [suppressed, excited],
        "two_step": {
            "median_delta_log_rate_joint": float(np.median(steps["joint"])),
            "median_delta_log_rate_two_step": float(np.median(steps["two_step"])),
            "fraction_units_two_step_below_0": float(np.mean(steps["two_step"] < 0)),
            "wilcoxon_p_two_step_against_0": float(wilcoxon(steps["two_step"]).pvalue),
            "units_outside_the_axis_limits_of_panel_e": int(
                np.sum(
                    (np.abs(steps["joint"]) > limits[1])
                    | (np.abs(steps["two_step"]) > limits[1])
                )
            ),
        },
        "median_center_exit_ms_from_first_flash": rates["median_exit_ms"],
    }
    output.with_name("hold_effect_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))
    print(output)


if __name__ == "__main__":
    main()
