"""Show the suppression of V1 firing while the animal holds with the stimulus on.

Question: V1 units fire less from about 80 ms after the first flash until the
animal leaves the center port. Does that follow the stimulus (adaptation to the
flashes) or the fixation (the time in the center port)? The flashes go on from the first flash to the response
entry, so the time after the center exit has the stimulus without the fixation.

Comparison: the fixation (first flash to center exit) against the time from the
center exit to the response entry, in the same trials and the same units. The
comparison is paired: each unit has a value in both periods.

- Panel a (examples): spikes of two units against time from the first flash
  (left) and from the center exit (right), on the time axes of panels b and c,
  with the trials sorted by the length of the fixation. The vertical line is the
  event at time 0. Top: the unit with
  the lowest fixation level among the units with a rate in `SUPPRESSED_RATES`.
  Bottom: the unit with the highest fixation level among the units that fire at
  `EXAMPLE_MIN_RATE` or more.
- Panel b (observation): raw firing rate, no model, against time from the first
  flash and from the center exit. The rate of a unit is divided by its mean
  rate over the valid bins; the line is the mean over units and the band is its
  SEM. The
  top row is the fraction of trials with a flash in each 40 ms.
- Panel c (characterization): the kernels of a descriptive model, as a change
  in log rate. The model is the design of `dmat.py` with one change: the
  withdrawal kernel (0 to 100 ms) is replaced with a kernel from the center
  exit to the response entry (`AFTER_BASES` raised cosines, `AFTER_WINDOW_S`).
  Both kernels are relative to the time before the first flash. Thin lines are
  units; the thick line and band are the median and the interquartile range.
- Panel d (the paired comparison): each unit's mean kernel value over
  `LEVEL_MS` of the fixation against its mean over the same time after the center
  exit. Error bars are the SD over the 10 cross-validation folds, in both
  directions. Two-sided Wilcoxon signed-rank test of the two values. Points
  outside `LEVEL_LIMITS` are not drawn; they are in the statistics, and their
  number is in `hold_effect_stats.json`.
- Panel e (control for the order of fitting): does the fixation kernel take
  variance from the other terms? Each unit's fixation level in the main model (all
  terms fit together) against its level in a two-step fit: the design with no
  fixation kernel first, then the fixation kernel alone on top of that fixed
  prediction (`two_step_levels`). The two-step value is a lower bound. Both are
  in-sample.
- Panel f (control for stillness): raw rate against motion energy in the
  center port and at a side port (`port_vs_stillness.py`).
- Panel g (control for the movement after the exit): the rate around a flash
  in the fixation, and around a flash after the exit with less and with more
  motion energy of the video (lower and upper half), from
  `matched_flashes.py`. The first flash of a trial is not used. The rate of a
  unit is divided by its mean over the flashes in the fixation and after the
  exit; lines are the mean and bands the SEM over units. The two groups after
  the exit are matched for the time from the first flash, the gap from the
  flash before, and the time from the exit. The fixation is not matched to
  them for time: the fixation has almost the same length in all trials.
- Panel h (which units): each unit's fixation level (mean of its fixation
  kernel over `LEVEL_MS`, main model) against the unique ΔD² of its visual
  flash regressor. A value below `UNIQUE_FLOOR` is drawn there, and a level
  outside `LEVEL_LIMITS` is not drawn; all units are in the Spearman
  correlation of `matched_flashes.json`.

The descriptive model and the design with no fixation kernel are fit here (`glm
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
again. Run `port_vs_stillness` and `matched_flashes` first.
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
from scipy.ndimage import gaussian_filter1d
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
from thesis.ephys.units import fetch_unit_table
from thesis.plotting import (
    CONCEPT_COLORS,
    FIGURE_STYLE,
    OBSERVED_COLOR,
    REFERENCE_COLOR,
    save_figure,
    separate_axes,
)

CAPTION = """\
**V1 firing is lower during fixation and recovers at the withdrawal, while the
flashes go on.** The fixation is the time from the first flash to the
withdrawal from the center port. The flashes continue until the animal enters
a response port, so the time after the withdrawal has the stimulus without the
fixation. {units} units, one session. **a**, Spikes of two example units
against time from the first flash (left) and from the withdrawal (right), with
the trials sorted by the length of the fixation. The vertical line is the event
at time 0; in the left panels the orange line is the withdrawal of each trial.
Top: a unit with a lower rate in fixation. Bottom: a unit with a higher rate
in fixation. **b**, Observation, with no model. Top: the fraction of trials
with a flash in each 40 ms. Bottom: the firing rate, with the rate of each unit
divided by its mean rate (mean and SEM over units). The rate is {before:.2f}
before the first flash, {late:.2f} in the last 200 ms of the fixation, and
{after_rate:.2f} from 100 to 400 ms after the withdrawal. **c**, The kernels
of a descriptive model, as a change in log rate relative to the time before
the first flash. This model is the GLM of the main figure with one change: a
kernel from the withdrawal to the response entry in place of the 100-ms
withdrawal kernel. Thin lines are units; the thick line and the band are the
median and the interquartile range. **d**, The paired comparison. For each
unit, the mean kernel value from {lo} to {hi} ms after the first flash against
the mean over the same time after the withdrawal (error bars: SD over the 10
cross-validation folds). Median Δ log rate {hold:.2f} in fixation (below zero
in {hold_below:.0%} of the units) and {after:+.2f} after the withdrawal;
higher after the withdrawal in {higher} of {units} units (two-sided Wilcoxon
signed-rank test, p = {p:.1g}). {outside_d} units are outside the axis limits
and are in the test. **e**, Control for the order of fitting. The level in
fixation when all terms are fit together against the level when the fixation
kernel is fit last, on what the other terms do not explain (a lower bound).
Median {joint:.2f} and {two_step:.2f}; with the kernel fit last, the level is
below zero in {two_below:.0%} of the units (p = {p_two:.1g}). {outside_e} units
are outside the axis limits. **f**, Control for stillness. Raw firing rate
against motion energy of the video, in the center port and at a side port.
At the same low motion energy, the rate is lower in the center port (median
ratio {still_ratio:.2f}, lower in {still_fraction:.0%} of the units,
p = {still_p:.1g}).
**g**, Control for the movement after the withdrawal. The firing rate
around a flash in fixation, and around a flash after the withdrawal with less
and with more motion energy of the video (lower and upper half; mean and SEM
over units; the first flash of a trial is not used). The response after the
withdrawal does not grow with motion energy (ratio of the response, more over
less motion energy: {energy_ratio:.2f}, p = {energy_p:.1g}). **h**, The units
with a larger flash response have a lower rate in fixation: the level in
fixation against the unique ΔD² of the visual flash (Spearman ρ = {rho:.2f},
p = {rho_p:.1g}; {outside_h} units are outside the axis limits).
"""
ROOT_SUFFIX = "_after_exit_kernel"
NO_HOLD_SUFFIX = "_no_hold_term"
# The excited example unit must fire at this rate or more (spikes/s).
EXAMPLE_MIN_RATE = 2.0
# The suppressed example unit must have a rate in this range (spikes/s): enough
# spikes to see the hold, few enough to see single spikes.
SUPPRESSED_RATES = (5.0, 20.0)
AFTER = "after_exit"
AFTER_WINDOW_S = 0.6
AFTER_BASES = 8
LEVEL_MS = (150, 500)
# Axis limits of the kernel panels (Δ log rate).
LEVEL_LIMITS = (-1.6, 1.6)
UNIQUE_FLOOR = 1e-4  # panel h: a smaller unique ΔD² is drawn here
RATE_BIN_MS = 20
FLASH_SLOT_S = 0.04
FLASH_WINDOW_MS = (-100, 500)
EXIT_WINDOW_MS = (-200, 500)
TASK, VISUAL = CONCEPT_COLORS["task"], CONCEPT_COLORS["visual"]
MOTION = CONCEPT_COLORS["motion energy"]


def after_exit_basis() -> dmat.BasisFunction:
    """Bases of the kernel after the center exit, spaced like the hold kernel."""
    post = AFTER_WINDOW_S + dmat.BINWIDTH_S
    if dmat.HOLD_LOG_STRETCH_S:
        return dmat.StretchedCosineBasis(
            AFTER_BASES, post, dmat.BINWIDTH_S, dmat.HOLD_LOG_STRETCH_S
        )
    return dmat.RaisedCosineBasis(AFTER_BASES, 0, post, dmat.BINWIDTH_S)


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
    order = np.argsort(leave)
    rasters = {}
    for _, unit in units.iterrows():
        spikes = np.asarray(unit["spike_times_s"], dtype=float)
        counts = build_unit_counts(alignments, spikes).reshape(grid)
        mean_count = counts[valid].mean()
        if mean_count <= 0:
            continue
        if int(unit["unit_id"]) in examples:
            # Spike times of each trial from the first flash and from the exit.
            rows, bins = np.nonzero(counts[order] > 0)
            spike_ms = 1000 * relative[bins]
            rasters[int(unit["unit_id"])] = {
                "flash": (spike_ms, rows),
                "exit": (spike_ms - 1000 * leave[order][rows], rows),
            }
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
    matched_path = args.root / "matched_flashes.npz"
    if not matched_path.exists():
        raise FileNotFoundError(f"{matched_path}: run `matched_flashes` first.")
    with np.load(matched_path, allow_pickle=False) as data:
        matched = {key: data[key] for key in data.files}
    stillness_path = args.root / "port_vs_stillness.npz"
    if not stillness_path.exists():
        raise FileNotFoundError(f"{stillness_path}: run `port_vs_stillness` first.")
    with np.load(stillness_path, allow_pickle=False) as data:
        stillness = {key: data[key] for key in data.files}
    kernels = kernel_values(fit_descriptive(args.root, args.units), args.units)
    steps = two_step_levels(args.root, args.units)
    # Example units: the most suppressed and the most excited unit, each among
    # the units with a rate that makes its raster readable.
    ids, joint = steps["unit_ids"], steps["joint"]
    active = steps["rate"] >= EXAMPLE_MIN_RATE
    readable = (steps["rate"] >= SUPPRESSED_RATES[0]) & (
        steps["rate"] <= SUPPRESSED_RATES[1]
    )
    suppressed = int(ids[readable][np.argmin(joint[readable])])
    excited = int(ids[active][np.argmax(joint[active])])
    rates = raw_rates(args.root, args.units, (suppressed, excited))
    hold, after = kernels[dmat.HOLD], kernels[AFTER]
    before = float(rates["flash"][1][:, rates["flash"][0] < 0].mean())

    with plt.rc_context(FIGURE_STYLE):
        figure = plt.figure(figsize=(7.4, 12.6))
        examples = figure.add_gridspec(
            2, 2, left=0.11, right=0.97, top=0.9794, bottom=0.8349, hspace=0.25
        )
        top = figure.add_gridspec(
            2,
            2,
            left=0.11,
            right=0.97,
            top=0.7895,
            bottom=0.6492,
            height_ratios=(1, 4),
        )
        middle = figure.add_gridspec(
            1, 2, left=0.11, right=0.97, top=0.5832, bottom=0.4635
        )
        bottom = figure.add_gridspec(
            1, 3, left=0.11, right=0.88, top=0.3851, bottom=0.22, wspace=0.75
        )
        last = figure.add_gridspec(
            1, 2, left=0.11, right=0.9, top=0.15, bottom=0.045, wspace=1.25
        )
        panel_axes = []
        # Panel a has the time axes of panels b and c: from the first flash
        # (left) and from the center exit (right).
        exits = rates["sorted_exit_ms"]
        trial_rows = np.arange(len(exits))
        for row, (unit_id, label) in enumerate(
            ((suppressed, "suppressed"), (excited, "excited"))
        ):
            for column, (name, limits) in enumerate(
                (("flash", FLASH_WINDOW_MS), ("exit", EXIT_WINDOW_MS))
            ):
                axis = figure.add_subplot(examples[row, column])
                spike_ms, rows = rates["rasters"][unit_id][name]
                keep = (spike_ms >= limits[0]) & (spike_ms < limits[1])
                axis.plot(
                    spike_ms[keep],
                    rows[keep],
                    "|",
                    color=OBSERVED_COLOR,
                    markersize=1.5,
                )
                # The event at time 0, and in the left column the exit of each trial.
                event = VISUAL if name == "flash" else TASK
                axis.axvline(0, color=event, linewidth=0.8)
                if name == "flash":
                    axis.plot(exits, trial_rows, color=TASK, linewidth=1)
                axis.set_xlim(*limits)
                axis.tick_params(labelbottom=False)
                if column == 0:
                    axis.set_ylabel("trial, sorted\nby fixation length")
                    axis.text(
                        0.01,
                        1.02,
                        f"unit {unit_id}, {label}",
                        transform=axis.transAxes,
                        va="bottom",
                        fontsize=6.5,
                    )
                else:
                    axis.tick_params(labelleft=False)
                if row == 0:
                    axis.text(
                        0.99,
                        1.02,
                        "first flash" if name == "flash" else "center exit",
                        color=event,
                        transform=axis.transAxes,
                        ha="right",
                        va="bottom",
                        fontsize=6.5,
                    )
                if row == 0 and column == 0:
                    panel_axes.append(axis)
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
        window = f"{LEVEL_MS[0]} to {LEVEL_MS[1]} ms"
        paired.set_xlabel(
            r"$\Delta$ log rate in fixation," + f"\n{window} from first flash"
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
        order_axis.set_xlabel(
            r"$\Delta$ log rate in fixation," + "\nall terms fit together"
        )
        order_axis.set_ylabel(
            r"$\Delta$ log rate in fixation," + "\nfixation kernel fit last"
        )
        stillness_axis = figure.add_subplot(bottom[2])
        port_vs_stillness.plot_rates(stillness_axis, stillness, states=(0, 2))
        # Panel g: the response to a flash in the fixation and after the exit,
        # and after the exit with less and with more motion energy.
        flash_axis = figure.add_subplot(last[0])
        lags = matched["lags_ms"]
        reference = matched["rate__fixation against after exit, matched gap"]
        by_energy = matched[
            "rate__after exit, lower against upper half of motion energy"
        ]
        scale = reference.mean((0, 2))
        used = scale > 0
        curves = (
            (reference[0], TASK, "in fixation"),
            (by_energy[0], "0.35", "after exit,\nless motion energy"),
            (by_energy[1], MOTION, "after exit,\nmore motion energy"),
        )
        for (values, color, label), height in zip(
            curves, (0.3, 0.98, 0.72), strict=True
        ):
            values = gaussian_filter1d(values[used] / scale[used, None], 3, axis=-1)
            mean, error = values.mean(0), sem(values, axis=0)
            flash_axis.plot(lags, mean, color=color)
            flash_axis.fill_between(
                lags, mean - error, mean + error, color=color, alpha=0.25, lw=0
            )
            flash_axis.text(
                1.02,
                height,
                label,
                color=color,
                transform=flash_axis.transAxes,
                va="top",
            )
        flash_axis.axvline(0, color=VISUAL, linewidth=0.8)
        flash_axis.set_ylim(bottom=0)
        flash_axis.set_xlabel("time from flash (ms)")
        flash_axis.set_ylabel("normalized firing rate")
        # Panel h: units with a larger flash response have a lower fixation level.
        relation = figure.add_subplot(last[1])
        relation.scatter(
            np.maximum(matched["flash_unique"], UNIQUE_FLOOR),
            matched["fixation_level"],
            s=6,
            color=TASK,
            alpha=0.7,
            linewidths=0,
        )
        relation.axhline(0, color=REFERENCE_COLOR, linestyle="--", linewidth=0.8)
        relation.set_xscale("log")
        relation.set_ylim(limits)
        relation.set_xlabel(r"unique $\Delta D^2$ of the visual flash")
        relation.set_ylabel(r"$\Delta$ log rate in fixation")
        panel_axes += [paired, order_axis, stillness_axis, flash_axis, relation]
        for axis in panel_axes[2:]:
            separate_axes(axis)
        figure.canvas.draw()
        to_figure = figure.transFigure.inverted()
        for letter, axis in zip("abcdefgh", panel_axes, strict=True):
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
    least = json.loads(stillness_path.with_suffix(".json").read_text())[
        "least_motion_group"
    ]
    two = stats["two_step"]
    flashes = json.loads(matched_path.with_suffix(".json").read_text())
    by_energy = flashes["after exit, lower against upper half of motion energy"][
        "response, second / first group"
    ]
    relation = flashes["fixation level against unique dD2, flash (Spearman)"]
    output.with_name("hold_effect_caption.md").write_text(
        CAPTION.format(
            energy_ratio=by_energy["median_ratio"],
            energy_p=by_energy["wilcoxon_p"],
            rho=relation["rho"],
            rho_p=relation["p"],
            outside_h=int(
                np.sum(
                    (matched["fixation_level"] < limits[0])
                    | (matched["fixation_level"] > limits[1])
                )
            ),
            units=stats["units"],
            before=stats["rate_before_first_flash"],
            late=stats["rate_last_200_ms_of_hold"],
            after_rate=stats["rate_100_to_400_ms_after_exit"],
            lo=LEVEL_MS[0],
            hi=LEVEL_MS[1],
            hold=stats["median_delta_log_rate_hold"],
            hold_below=stats["fraction_units_hold_below_0"],
            after=stats["median_delta_log_rate_after_exit"],
            higher=stats["units_higher_after_exit"],
            p=stats["wilcoxon_p"],
            outside_d=outside,
            joint=two["median_delta_log_rate_joint"],
            two_step=two["median_delta_log_rate_two_step"],
            two_below=two["fraction_units_two_step_below_0"],
            p_two=two["wilcoxon_p_two_step_against_0"],
            outside_e=two["units_outside_the_axis_limits_of_panel_e"],
            still_ratio=least["median_center_over_side"],
            still_fraction=least["fraction_units_lower_in_center_port"],
            still_p=least["wilcoxon_p"],
        )
    )
    print(json.dumps(stats, indent=1))
    print(output)


if __name__ == "__main__":
    main()
