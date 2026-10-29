"""Show the suppression of V1 firing while the animal holds with the stimulus on.

Question: V1 units fire less from about 80 ms after the first flash until the
animal leaves the center port. Does that follow the stimulus (adaptation to the
flashes) or the fixation (the time in the center port)? The flashes go on from the first flash to the response
entry, so the time after the center exit has the stimulus without the fixation.

Comparison: the fixation (first flash to center exit) against the time from the
center exit to the response entry, in the same trials and the same units. The
comparison is paired: each unit has a value in both periods. Panel a is
GRB006. In the other panels the units of both mice (`SESSIONS`) are pooled (standard
practice); in d and i the large marker is the pooled median with its
hierarchical-bootstrap 95% interval (mice, then units), and the open markers
are the median of each mouse (circle: GRB006, triangle: GRB064). The tests are
within each mouse.

Panels a to d observe and characterize the effect, e and f ask what starts
and ends it, g and h are controls, and i asks which units. The hold kernel
passes the inclusion test (`candidate_terms.py`, unique deviance with all other
terms in the model), so it is not a part of the rate that other terms explain.

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
- Panel c (characterization): the fixation kernel and the response-period
  kernel (from the center exit) of the model of the main figure, as a change
  in log rate. Both kernels are relative to the time before the center entry. Thin lines are
  units; the thick line and band are the median and the interquartile range.
- Panel d (the paired comparison): each unit's mean kernel value over
  `LEVEL_MS` of the fixation against its mean over the same time after the center
  exit. Error bars are the SD over the 10 cross-validation folds, in both
  directions. Two-sided Wilcoxon signed-rank test of the two values. Points
  outside `LEVEL_LIMITS` are not drawn; they are in the statistics, and their
  number is in `hold_effect_stats.json`.
- Panel e (does the drop start before the stimulus?), from
  `suppression_onset.py`, no model. Left: the population rate of the trials
  with a short and a long delay from the center entry to the first flash (the
  lower and upper half of the delays), aligned to the center entry; mean and
  SEM over the units of both mice. Right: in the long delays, the rate
  of each unit from `ENTRY_LAG_MS` after the entry to the first flash, and in
  `LATE_MS` after the first flash, over its rate before the entry (the gray
  bands of the left panel, labeled "before entry" and "in port, no flash";
  a linear axis from 0 to 1/8 and a log axis above it). Zero ratios stay in
  the plot and tests. A ratio below 1 before the flash is a
  drop with no stimulus. The bars above the traces are the 5th to 95th
  percentile of the first flash of each half (tick: the median), in the
  color of its trace. The test uses the long delays; the second band ends at
  their median first flash, and in each trial the window of the test ends at
  its own first flash.
- Panel f (does the rise at the exit need the stimulus?): the rate around the
  center exit in completed trials (the flashes go on) and in early
  withdrawals (the flashes stop), mean and SEM over the units of both mice,
  from `exit_without_stimulus.py`.
- Panel g (control for the movement after the exit): the rate around a flash
  in the fixation, and around a flash after the exit with less and with more
  motion energy of the video (lower and upper half), from
  `matched_flashes.py`. The first flash of a trial is not used. The rate of a
  unit is divided by its mean over the flashes in the fixation and after the
  exit; lines are the mean and bands the SEM over units. The two groups after
  the exit are matched for the time from the first flash, the gap from the
  flash before, and the time from the exit. The fixation is not matched to
  them for time: the fixation has almost the same length in all trials.
- Panel h (control for stillness): raw rate against motion energy in the
  center port and at a side port (`port_vs_stillness.py`), with its slope test.
  The two animals have different camera views, so motion energy is not the
  same measure.
- Panel i (which units): each unit's fixation level (mean of its fixation
  kernel over `LEVEL_MS`, main model) against the unique gain (held-out bits
  per spike) of its visual flash terms. A value below `UNIQUE_FLOOR` is drawn there, and a level
  outside `LEVEL_LIMITS` is not drawn; all units are in the Spearman
  correlation of `matched_flashes.json`.

Units, trials, alignment (first flash), baseline, and exclusions are those of
`fit.py`. Bins after the response entry are excluded. The sampling unit is the
unit. The script writes `hold_effect.svg`, `hold_effect_stats.json` (by
session), and `hold_effect_caption.md` in `--out`.

    uv run python -m thesis.ephys.analyses.glm.hold_effect

It makes no fit. Run `port_vs_stillness` and `matched_flashes` for each
session, and `suppression_onset` and `exit_without_stimulus`, first.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import sem, wilcoxon

from thesis.ephys.analyses import exit_without_stimulus
from thesis.ephys.analyses import suppression_onset as onset
from thesis.ephys.analyses.glm import cli, dmat, port_vs_stillness
from thesis.ephys.analyses.glm.fit import (
    analysis_unit_indices,
    build_unit_counts,
    load_unit_records,
)
from thesis.ephys.analyses.glm.flash_kernels import SESSIONS
from thesis.ephys.units import fetch_unit_table
from thesis.plotting import (
    CONCEPT_COLORS,
    FIGURE_STYLE,
    OBSERVED_COLOR,
    REFERENCE_COLOR,
    hierarchical_interval,
    pooled_points,
    pooled_strip,
    save_figure,
    separate_axes,
)

CAPTION = """\
**V1 firing changes during fixation and at the withdrawal, while the flashes
go on.** The fixation is the time from the first flash to the withdrawal from
the center port. The flashes continue until the animal enters a response port,
so the time after the withdrawal has the stimulus without the fixation. Two
mice: {units}. Except in a and f, the units of both mice are pooled; in d and
i the large marker is the pooled median with its hierarchical-bootstrap 95%
interval (the mice, then the units of each), and the open markers are the
median of each mouse ({subject} circle, {other} triangle). The tests are within each mouse. **a**, Spikes of two example units of
{subject} against time from the first flash (left) and from the withdrawal
(right), with the trials sorted by the length of the fixation. The vertical
line is the event at time 0; in the left panels the orange line is the
withdrawal of each trial. Top: a unit with a lower rate in fixation. Bottom: a
unit with a higher rate in fixation. **b**, Observation, with no model. Top:
the fraction of trials with a flash in each 40 ms. Bottom: the firing rate,
with the rate of each unit divided by its mean rate (mean and SEM over units).
The rate is {before} before the first flash, {late} in the last 200 ms of the
fixation, and {after_rate} from 100 to 400 ms after the withdrawal. **c**, The
fixation kernel and the response-period kernel of the GLM of the main figure,
as a change in log rate relative to the time before the center entry. Thin lines are units; the thick lines and the
bands are the median and the interquartile range. **d**, The paired comparison. For
each unit, the mean kernel value from {lo} to {hi} ms after the first flash
against the mean over the same time after the withdrawal (error bars: SD over
the 10 cross-validation folds). {paired} {outside_d} {bias} **e**, Does the drop
start before the stimulus? Left: the population rate, with no model, of the
trials with a short and a long delay from the center entry to the first flash
(the lower and upper half of the delays), aligned to the center entry; the
bars above are the 5th to 95th percentile of the first flash of each half. Right: {onset} **f**, Does the rise at the
withdrawal need the stimulus? The rate around the withdrawal in completed
trials, where the flashes go on, and in early withdrawals, where they stop
(units of both mice pooled, mean and SEM over units). {exit} **g**, Control for the movement after the withdrawal. The firing rate
around a flash in fixation, and around a flash after the withdrawal with less
and with more motion energy of the video (lower and upper half; mean and SEM
over units; the first flash of a trial is not used). {energy}
**h**, Control
for stillness. Raw firing rate against motion energy of the video, in the
center port and at a side port. {stillness} The two animals
have different camera views, so motion energy is not the same measure in
both. **i**, The level in fixation against the unique gain of the visual flash terms (bits per spike).
{relation} {outside_i}
"""
# The excited example unit must fire at this rate or more (spikes/s).
EXAMPLE_MIN_RATE = 2.0
# The suppressed example unit must have a rate in this range (spikes/s): enough
# spikes to see the hold, few enough to see single spikes.
SUPPRESSED_RATES = (5.0, 20.0)
EXAMPLE_PSTH_BIN_MS = 20  # panel a: the bin of the PSTH under each raster
AFTER = "after_exit"
LEVEL_MS = (150, 500)
# Axis limits of the kernel panels (Δ log rate).
LEVEL_LIMITS = (-1.6, 1.6)
UNIQUE_FLOOR = 1e-3  # panel i: a smaller unique gain (bits per spike) is drawn here
RATE_BIN_MS = 20
FLASH_SLOT_S = 0.04
FLASH_WINDOW_MS = (-100, 500)
EXIT_WINDOW_MS = (-200, 500)
TASK, VISUAL = CONCEPT_COLORS["task"], CONCEPT_COLORS["visual"]
DELAY_COLORS = {"short delay": "0.6", "long delay": OBSERVED_COLOR}
MOTION = CONCEPT_COLORS["motion energy"]
# The raw analyses of panels e and f (no model), in `figures/<name>/<session>/`.
ONSET_NAME = "suppression_onset"
EXIT_NAME = "exit_without_stimulus"
ONSET_WINDOW_MS = (-250, 290)


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
    out = {
        "unit_ids": np.asarray([item["unit_id"] for item in final]),
        "rate": np.asarray([item["fit_mean_count"] for item in final])
        / dmat.BINWIDTH_S,
    }
    for name, basis in (
        (dmat.HOLD, dmat.task_temporal_bases()[dmat.HOLD]),
        (AFTER, dmat.exit_basis()),
    ):
        # The kernel after the exit is the response period of the main model.
        key = dmat.RESPONSE_PERIOD if name == AFTER else name
        columns = [i for i, column in enumerate(names) if column.startswith(key)]
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
        fine = np.arange(*limits_ms) / 1000
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


def session_effect(root: Path, unit_set: str) -> dict:
    """Return the measures of one session (no fit)."""
    loaded = {}
    for name, command in (
        ("matched_flashes", "matched_flashes"),
        ("port_vs_stillness", "port_vs_stillness"),
    ):
        path = root / f"{name}.npz"
        if not path.exists():
            raise FileNotFoundError(f"{path}: run `{command} --root {root}` first.")
        with np.load(path, allow_pickle=False) as data:
            loaded[name] = {key: data[key] for key in data.files}
        loaded[f"{name}_json"] = json.loads(path.with_suffix(".json").read_text())
    kernels = kernel_values(root, unit_set)
    # Example units: the most suppressed and the most excited unit, each among
    # the units with a rate that makes its raster readable.
    ids, joint = kernels["unit_ids"], kernels[dmat.HOLD]["level"]
    active = kernels["rate"] >= EXAMPLE_MIN_RATE
    readable = (kernels["rate"] >= SUPPRESSED_RATES[0]) & (
        kernels["rate"] <= SUPPRESSED_RATES[1]
    )
    readable = readable if readable.any() else active
    suppressed = int(ids[readable][np.argmin(joint[readable])])
    excited = int(ids[active][np.argmax(joint[active])])
    rates = raw_rates(root, unit_set, (suppressed, excited))
    hold, after = kernels[dmat.HOLD], kernels[AFTER]
    limits = LEVEL_LIMITS
    centers, values = rates["exit"]
    matched = loaded["matched_flashes"]
    stats = {
        "units": len(hold["level"]),
        "rate_before_first_flash": float(
            rates["flash"][1][:, rates["flash"][0] < 0].mean()
        ),
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
        "units_outside_the_axis_limits_of_panel_d": int(
            np.sum(
                (np.minimum(hold["level"], after["level"]) < limits[0])
                | (np.maximum(hold["level"], after["level"]) > limits[1])
            )
        ),
        "example_units_suppressed_and_excited": [suppressed, excited],
        "units_outside_the_axis_limits_of_panel_i": int(
            np.sum(
                (matched["fixation_level"] < limits[0])
                | (matched["fixation_level"] > limits[1])
            )
        ),
        "median_center_exit_ms_from_first_flash": rates["median_exit_ms"],
    }
    flashes = loaded["matched_flashes_json"]
    return {
        "subject": root.name.split("_")[0],
        "kernels": kernels,
        "rates": rates,
        "stillness": loaded["port_vs_stillness"],
        "matched": matched,
        "examples": (suppressed, excited),
        "stats": stats,
        "least": loaded["port_vs_stillness_json"]["least_motion_group"],
        "slope": loaded["port_vs_stillness_json"]["slope_against_motion_energy_group"],
        "by_energy": flashes["after exit, lower against upper half of motion energy"][
            "response, second / first group"
        ],
        "relation": flashes[
            "fixation level against unique bits per spike, flash (Spearman)"
        ],
    }


def _draw_examples(figure, examples, data: dict) -> list:
    """Draw panel a: the rasters of the two example units; return its first axis."""
    rates = data["rates"]
    suppressed, excited = data["examples"]
    exits = rates["sorted_exit_ms"]
    trial_rows = np.arange(len(exits))
    first = None
    for row, (unit_id, label) in enumerate(
        # The label is the rate in the fixation, not the response to a flash.
        ((suppressed, "lower rate in fixation"), (excited, "higher rate in fixation"))
    ):
        for column, (name, limits) in enumerate(
            (("flash", FLASH_WINDOW_MS), ("exit", EXIT_WINDOW_MS))
        ):
            axis = figure.add_subplot(examples[2 * row, column])
            spike_ms, rows = rates["rasters"][unit_id][name]
            keep = (spike_ms >= limits[0]) & (spike_ms < limits[1])
            axis.plot(
                spike_ms[keep], rows[keep], "|", color=OBSERVED_COLOR, markersize=1.5
            )
            # The PSTH of the same spikes: mean and SEM over trials.
            psth = figure.add_subplot(examples[2 * row + 1, column], sharex=axis)
            edges = np.arange(limits[0], limits[1] + 1e-9, EXAMPLE_PSTH_BIN_MS)
            counts = np.histogram2d(
                rows[keep], spike_ms[keep], bins=(np.arange(len(exits) + 1), edges)
            )[0] / (EXAMPLE_PSTH_BIN_MS / 1000)
            mean = counts.mean(0)
            sem = counts.std(0, ddof=1) / np.sqrt(len(counts))
            centers = (edges[:-1] + edges[1:]) / 2
            psth.plot(centers, mean, color=OBSERVED_COLOR, linewidth=1)
            psth.fill_between(
                centers, mean - sem, mean + sem, color=OBSERVED_COLOR, alpha=0.2, lw=0
            )
            psth.axvline(0, color=VISUAL if name == "flash" else TASK, linewidth=0.8)
            psth.set_ylim(bottom=0)
            if column == 0:
                psth.set_ylabel("spikes/s")
            axis.tick_params(labelbottom=False)
            # The event at time 0, and in the left column the exit of each trial.
            event = VISUAL if name == "flash" else TASK
            axis.axvline(0, color=event, linewidth=0.8)
            if name == "flash":
                axis.plot(exits, trial_rows, color=TASK, linewidth=1)
            axis.set_xlim(*limits)
            if row == 0:
                psth.tick_params(labelbottom=False)
            else:
                psth.set_xlabel(
                    "time from first flash (ms)"
                    if name == "flash"
                    else "time from center exit (ms)"
                )
            if column == 0:
                axis.set_ylabel("trial")
                axis.text(
                    0.01,
                    1.02,
                    f"{data['subject']}, unit {unit_id}, {label}",
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
            first = first or axis
    return first


def _draw_onset(figure, slots, results: list[dict]) -> tuple[list, dict]:
    """Draw panel e: the population rate from the center entry, and the paired
    test of the drop before the first flash; return the axes and its statistics."""
    time = np.asarray(results[0]["time_ms"])
    shown = (time >= ONSET_WINDOW_MS[0]) & (time <= ONSET_WINDOW_MS[1])
    trace = figure.add_subplot(slots[0])
    # The units of both mice pooled: mean and SEM over units.
    delays = {}
    for (delay, color), height in zip(DELAY_COLORS.items(), (0.96, 0.87), strict=True):
        values = np.concatenate(
            [
                np.asarray(data["aligned to the center entry"]["rate_by_unit"][delay])
                for data in results
            ]
        )[:, shown]
        mean, error = values.mean(0), sem(values, axis=0)
        trace.fill_between(
            time[shown], mean - error, mean + error, color=color, alpha=0.25, lw=0
        )
        trace.plot(time[shown], mean, color=color, lw=1)
        trace.text(
            0.02,
            height,
            delay,
            color=color,
            transform=trace.transAxes,
            ha="left",
            va="top",
            fontsize=5.5,
        )
        delays[delay] = np.concatenate(
            [
                d[f"{delay.split()[0]}-delay trials"]["delay_ms_by_trial"]
                for d in results
            ]
        )
    # The two windows of the test, and the median first flash of the long delays.
    for limits in (
        onset.BEFORE_ENTRY_MS,
        (
            onset.ENTRY_LAG_MS,
            np.median([d["median_delay_ms_short_and_long"][1] for d in results]),
        ),
    ):
        trace.axvspan(*limits, color=REFERENCE_COLOR, alpha=0.25, lw=0)
    # The two windows of the ratio in the right panel, labeled in place.
    for x, label, align in (
        (np.mean(onset.BEFORE_ENTRY_MS), "before\nentry", "center"),
        (onset.ENTRY_LAG_MS + 4, "in port,\nno flash", "left"),
    ):
        trace.text(
            x,
            0.04,
            label,
            transform=trace.get_xaxis_transform(),
            ha=align,
            va="bottom",
            color="0.35",
            fontsize=5.5,
        )
    trace.axvline(0, color=TASK, lw=0.8)
    # The first flash of each delay group, both mice: 5th to 95th percentile
    # and the median, in the color of its trace.
    top = trace.get_ylim()[1]
    for (delay, color), height in zip(
        DELAY_COLORS.items(), (top * 1.04, top), strict=True
    ):
        low, middle, high = np.percentile(delays[delay], [5, 50, 95])
        trace.plot([low, high], [height] * 2, color=color, lw=2, solid_capstyle="butt")
        trace.plot(middle, height, "|", color=color, ms=6, mew=1.5)
    trace.text(
        np.percentile(delays["long delay"], 95) + 8,
        top * 1.02,
        "first flash",
        va="center",
        fontsize=5.5,
    )
    trace.set_ylim(top=top * 1.07)
    trace.set_xlabel("time from center entry (ms)")
    trace.set_ylabel("normalized\nfiring rate")
    strip = figure.add_subplot(slots[1])
    subjects = [data["subject"] for data in results]
    summary = {}
    for position, (key, label) in enumerate(
        (
            ("entry to first flash over before entry", "entry to\nfirst flash"),
            (
                "late over before entry",
                f"{onset.LATE_MS[0]}-{onset.LATE_MS[1]} ms\nafter flash",
            ),
        )
    ):
        groups = []
        for data in results:
            ratio = np.asarray(
                data["long-delay trials"]["ratio_by_unit"][key], dtype=float
            )
            groups.append(ratio[np.isfinite(ratio) & (ratio >= 0)])
        summary[key] = pooled_strip(
            strip, position, groups, subjects, TASK if position else OBSERVED_COLOR
        )
        summary[key]["p_bootstrap"] = hierarchical_interval(
            [group - 1 for group in groups]
        )["p_bootstrap"]
        summary[key]["null_ratio"] = 1
        summary[key]["units"] = sum(map(len, groups))
    strip.axhline(1, color=REFERENCE_COLOR, lw=0.8)
    strip.set_xticks([0, 1], ["entry to\nfirst flash", "late\nfixation"])
    strip.set_xlim(-0.5, 1.7)
    strip.set_yscale("symlog", linthresh=1 / 8, base=2)
    strip.set_ylim(bottom=-0.02)
    strip.set_yticks(
        [0, 1 / 8, 1 / 2, 1, 2, 4, 16], ["0", "1/8", "1/2", "1", "2", "4", "16"]
    )
    strip.set_ylabel("rate over rate\nbefore entry")
    strip.set_title("long delays", fontsize=6, loc="left")
    return [trace, strip], summary


def _draw_exit(axis, results: list[dict]) -> None:
    """Draw panel f: the rate around the center exit with and without the stimulus.

    The units of the mice are pooled: mean and SEM over all units.
    """
    time = np.asarray(results[0]["time_from_exit_ms"])
    # Short labels, so that they stay to the left of the exit.
    labels = {
        "completed": "completed:\nflashes go on",
        "early withdrawal": "early\nwithdrawal:\nflashes stop",
    }
    for (name, label), height in zip(labels.items(), (0.98, 0.8), strict=True):
        color = exit_without_stimulus.COLORS[name]
        rates = np.concatenate(
            [np.asarray(r[name]["normalized_rate_by_unit"]) for r in results]
        )
        mean, error = rates.mean(0), rates.std(0, ddof=1) / np.sqrt(len(rates))
        axis.plot(time, mean, color=color)
        axis.fill_between(
            time, mean - error, mean + error, color=color, alpha=0.25, lw=0
        )
        axis.text(
            0.03,
            height,
            label,
            color=color,
            transform=axis.transAxes,
            va="top",
            fontsize=5.5,
        )
    axis.axvline(0, color=TASK, linewidth=0.8)
    axis.set_ylim(bottom=0)
    axis.set_xlabel("time from center exit (ms)")
    axis.set_ylabel("normalized firing rate")
    axis.text(
        0.98,
        0.02,
        f"{sum(r['units'] for r in results)} units, {len(results)} mice",
        transform=axis.transAxes,
        ha="right",
        fontsize=5.5,
    )


def _draw_matched(axis, data: dict, labels: bool, style: str = "-") -> None:
    """Draw panel i: the rate around flashes in three conditions."""
    matched = data["matched"]
    lags = matched["lags_ms"]
    reference = matched["rate__fixation against after exit, matched gap"]
    by_energy = matched["rate__after exit, lower against upper half of motion energy"]
    scale = reference.mean((0, 2))
    used = scale > 0
    curves = (
        (reference[0], TASK, "in port"),
        (by_energy[0], "0.35", "after exit, less motion energy"),
        (by_energy[1], MOTION, "after exit, more motion energy"),
    )
    # The labels after the exit are in the band between the curves; the label of
    # the fixation is below its curve.
    for (values, color, label), height in zip(curves, (0.33, 0.56, 0.48), strict=True):
        values = gaussian_filter1d(values[used] / scale[used, None], 3, axis=-1)
        mean, error = values.mean(0), sem(values, axis=0)
        axis.plot(lags, mean, color=color, ls=style)
        axis.fill_between(
            lags, mean - error, mean + error, color=color, alpha=0.2, lw=0
        )
        if labels:
            axis.text(
                0.03,
                height,
                label,
                color=color,
                transform=axis.transAxes,
                va="top",
                fontsize=6,
            )
    axis.axvline(0, color=VISUAL, linewidth=0.8)
    axis.set_ylim(bottom=0)
    axis.set_xlabel("time from flash (ms)")


def _pair(sessions, text) -> str:
    """Return one value for each session, as 'x (A) and y (B)'."""
    return " and ".join(f"{text(s)} ({s['subject']})" for s in sessions)


def _direction(p_value: float, ratio: float) -> str:
    """Return the direction of a ratio as a word, or "not reliably different"."""
    if p_value >= 0.05:
        return "not reliably different"
    return "higher" if ratio > 1 else "lower"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("results/glm"))
    parser.add_argument("--units", type=cli._unit_set, default="all")
    args = parser.parse_args()
    sessions = [session_effect(args.out / s, args.units) for s in SESSIONS]
    limits = LEVEL_LIMITS

    raw = {
        name: [
            # The model-free controls are not GLM outputs; they stay in figures/<name>.
            json.loads((Path("figures") / name / session / f"{name}.json").read_text())
            for session in SESSIONS
        ]
        for name in (ONSET_NAME, EXIT_NAME)
    }

    with plt.rc_context(FIGURE_STYLE):
        figure = plt.figure(figsize=(7.4, 11.4))
        rows = figure.add_gridspec(
            5,
            1,
            left=0.09,
            right=0.98,
            top=0.98,
            bottom=0.05,
            height_ratios=(2.4, 1.35, 1.9, 1.9, 1.9),
            hspace=0.55,
        )
        # Each raster has its PSTH (mean rate over trials) below it.
        examples = rows[0].subgridspec(
            4, 2, height_ratios=(1, 0.6, 1, 0.6), hspace=0.45, wspace=0.16
        )
        # A narrow empty column keeps the letter of c away from b.
        top = rows[1].subgridspec(
            2, 5, height_ratios=(1, 4), width_ratios=(1, 1, 0.12, 1, 1), wspace=0.3
        )
        grid = [rows[k].subgridspec(1, 3, wspace=0.62) for k in (2, 3, 4)]
        panel_axes = [_draw_examples(figure, examples, sessions[0])]
        shared = None
        kernel_axes = []
        for column, (name, xlabel) in enumerate(
            (("flash", "time from first flash (ms)"), ("exit", "time from center exit"))
        ):
            xlabel += " (ms)" if name == "exit" else ""
            flash_axis = figure.add_subplot(top[0, column])
            axis = figure.add_subplot(top[1, column], sharex=flash_axis, sharey=shared)
            shared = shared or axis
            kernel_axis = figure.add_subplot(
                top[1, 3 + column], sharey=kernel_axes[0] if column else None
            )
            kernel_axes.append(kernel_axis)
            # The units of both mice pooled (standard practice).
            slots, _ = sessions[0]["rates"][f"{name}_fraction"]
            fraction = np.mean(
                [data["rates"][f"{name}_fraction"][1] for data in sessions], axis=0
            )
            flash_axis.step(slots, fraction, where="post", color=VISUAL)
            centers = sessions[0]["rates"][name][0]
            values = np.concatenate([data["rates"][name][1] for data in sessions])
            mean, error = values.mean(axis=0), sem(values, axis=0)
            axis.fill_between(
                centers,
                mean - error,
                mean + error,
                color=OBSERVED_COLOR,
                alpha=0.25,
                lw=0,
            )
            axis.plot(centers, mean, color=OBSERVED_COLOR)
            key = dmat.HOLD if name == "flash" else AFTER
            times_all = sessions[0]["kernels"][key]["times"]
            shown = times_all <= (FLASH_WINDOW_MS if column == 0 else EXIT_WINDOW_MS)[1]
            times = times_all[shown]
            curves = np.concatenate(
                [data["kernels"][key]["values"][:, shown] for data in sessions]
            )
            kernel_axis.plot(
                times, curves.T, color="0.6", linewidth=0.3, alpha=0.25, zorder=1
            )
            low, median, high = np.percentile(curves, [25, 50, 75], axis=0)
            kernel_axis.fill_between(
                times, low, high, color=TASK, alpha=0.4, linewidth=0, zorder=2
            )
            kernel_axis.plot(times, median, color=TASK, linewidth=1.6, zorder=3)
            flash_axis.set_ylim(0, 1.05)
            flash_axis.tick_params(labelbottom=False)
            axis.set_xlabel(xlabel)
            kernel_axis.axhline(0, color=REFERENCE_COLOR, linestyle="--", linewidth=0.8)
            kernel_axis.set_xlim(axis.get_xlim())
            kernel_axis.set_xlabel(xlabel)
            if column == 0:
                flash_axis.text(
                    0.3,
                    0.75,
                    "fraction of trials\nwith a flash",
                    color=VISUAL,
                    transform=flash_axis.transAxes,
                    fontsize=6,
                )
                axis.set_ylabel("normalized\nfiring rate")
                axis.text(
                    0.98,
                    0.96,
                    f"{len(values)} units, {len(sessions)} mice",
                    transform=axis.transAxes,
                    ha="right",
                    va="top",
                    fontsize=6,
                )
                panel_axes.append(flash_axis)
                kernel_axis.set_ylim(*limits)
                kernel_axis.set_ylabel(r"$\Delta$ log rate")
                kernel_axis.text(
                    0.98,
                    0.96,
                    "units",
                    color="0.55",
                    transform=kernel_axis.transAxes,
                    ha="right",
                    va="top",
                    fontsize=6,
                )
                kernel_axis.text(
                    0.98,
                    0.84,
                    "median, IQR",
                    color=TASK,
                    transform=kernel_axis.transAxes,
                    ha="right",
                    va="top",
                    fontsize=6,
                )
                panel_axes.append(kernel_axis)
            else:
                flash_axis.tick_params(labelleft=False)
                axis.tick_params(labelleft=False)
                kernel_axis.tick_params(labelleft=False)

        window = f"{LEVEL_MS[0]} to {LEVEL_MS[1]} ms"
        paired = figure.add_subplot(grid[0][0])
        onset_axes, onset_stats = _draw_onset(
            figure, (grid[0][1], grid[0][2]), raw[ONSET_NAME]
        )
        exit_axis = figure.add_subplot(grid[1][0])
        _draw_exit(exit_axis, raw[EXIT_NAME])
        matched_axis = figure.add_subplot(grid[1][1])
        stillness_axis = figure.add_subplot(grid[1][2])
        relation = figure.add_subplot(grid[2][0])
        subjects = [data["subject"] for data in sessions]
        holds = [data["kernels"][dmat.HOLD] for data in sessions]
        afters = [data["kernels"][AFTER] for data in sessions]
        # Panel d: each unit with its SD over the folds in both directions.
        paired.errorbar(
            np.concatenate([h["level"] for h in holds]),
            np.concatenate([a["level"] for a in afters]),
            xerr=np.concatenate([h["level_sd"] for h in holds]),
            yerr=np.concatenate([a["level_sd"] for a in afters]),
            fmt="none",
            ecolor=TASK,
            elinewidth=0.4,
            alpha=0.4,
        )
        paired.set_xlim(limits)
        paired.set_ylim(limits)
        pooled_points(
            paired,
            [h["level"] for h in holds],
            [a["level"] for a in afters],
            subjects,
            TASK,
        )
        relation.set_xscale("log")
        relation.set_ylim(limits)
        pooled_points(
            relation,
            [
                np.maximum(data["matched"]["flash_unique"], UNIQUE_FLOOR)
                for data in sessions
            ],
            [data["matched"]["fixation_level"] for data in sessions],
            subjects,
            TASK,
        )
        paired.plot(
            limits, limits, color=REFERENCE_COLOR, linestyle="--", linewidth=0.8
        )
        for line in (paired.axhline, paired.axvline):
            line(0, color=REFERENCE_COLOR, linewidth=0.5)
        paired.set_xlim(limits)
        paired.set_ylim(limits)
        paired.set_xlabel(
            r"$\Delta$ log rate in fixation," + f"\n{window} from first flash"
        )
        paired.set_ylabel(r"$\Delta$ log rate after exit," + f"\n{window} from exit")
        # Panels g and h: the units of both mice pooled.
        port_vs_stillness.plot_rates(
            stillness_axis,
            {
                "rate": np.concatenate([d["stillness"]["rate"] for d in sessions]),
                "frames": np.sum([d["stillness"]["frames"] for d in sessions], axis=0),
            },
            states=(0, 2),
        )
        stillness_axis.set_xlabel("motion energy\n(percentile in each mouse)")
        keys = (
            "rate__fixation against after exit, matched gap",
            "rate__after exit, lower against upper half of motion energy",
        )
        _draw_matched(
            matched_axis,
            {
                "matched": {
                    "lags_ms": sessions[0]["matched"]["lags_ms"],
                    **{
                        key: np.concatenate(
                            [d["matched"][key] for d in sessions], axis=1
                        )
                        for key in keys
                    },
                }
            },
            labels=True,
        )
        matched_axis.set_ylabel("normalized firing rate")
        relation.axhline(0, color=REFERENCE_COLOR, linestyle="--", linewidth=0.8)
        relation.set_xlabel("unique gain of the flash terms\n(bits per spike)")
        relation.set_ylabel(r"$\Delta$ log rate in fixation")
        panel_axes += [
            paired,
            onset_axes[0],
            exit_axis,
            matched_axis,
            stillness_axis,
            relation,
        ]
        for axis in panel_axes[2:] + onset_axes[1:]:
            separate_axes(axis)
        figure.canvas.draw()
        to_figure = figure.transFigure.inverted()
        for letter, axis in zip("abcdefghi", panel_axes, strict=True):
            left = axis.yaxis.get_tightbbox().transformed(to_figure).x0
            figure.text(
                left,
                axis.get_position().y1 + 0.012,
                letter,
                fontweight="bold",
                fontsize=10,
            )
        output = save_figure(figure, args.out / "hold_effect")

    stats = {s: data["stats"] for s, data in zip(SESSIONS, sessions, strict=True)}
    stats["rate over the rate before the entry, long delays (panel e)"] = onset_stats
    (args.out / "hold_effect_stats.json").write_text(json.dumps(stats, indent=1))
    first, other = sessions[0]["subject"], sessions[1]["subject"]
    st = lambda data: data["stats"]  # noqa: E731
    paired_text = " ".join(
        f"{d['subject']}: median Δ log rate {st(d)['median_delta_log_rate_hold']:.2f}"
        f" in fixation (below zero in {st(d)['fraction_units_hold_below_0']:.0%} of"
        f" the units) and {st(d)['median_delta_log_rate_after_exit']:+.2f} after the"
        f" withdrawal; higher after the withdrawal in"
        f" {st(d)['units_higher_after_exit']} of {st(d)['units']} units (two-sided"
        f" Wilcoxon signed-rank test, p = {st(d)['wilcoxon_p']:.1g})."
        for d in sessions
    )
    stillness_text = (
        "At the same low motion energy, the rate in the center port against the"
        " rate at a side port is "
        + "; ".join(
            f"{_direction(d['least']['wilcoxon_p'], d['least']['median_center_over_side'])}"
            f" in {d['subject']} (median ratio"
            f" {d['least']['median_center_over_side']:.2f}, lower in"
            f" {d['least']['fraction_units_lower_in_center_port']:.0%} of the units,"
            f" p = {d['least']['wilcoxon_p']:.2g})"
            for d in sessions
        )
        + ". Over all motion-energy groups, the rate at a side port goes up with"
        " motion energy and the rate in the center port does not: the slope of"
        " each unit over the groups is higher at the side port in "
        + "; ".join(
            f"{d['slope']['side minus center']['units_higher_at_side_port']} of"
            f" {d['slope']['side minus center']['units']} units of {d['subject']}"
            f" (p = {d['slope']['side minus center']['wilcoxon_p']:.1g})"
            for d in sessions
        )
        + ". So the low rate in the port is not only the low rate of a still"
        " animal."
    )
    energy_text = (
        "The response after the withdrawal, more over less motion energy: "
        + "; ".join(
            f"{d['by_energy']['median_ratio']:.2f} in {d['subject']} (p ="
            f" {d['by_energy']['wilcoxon_p']:.2g})"
            for d in sessions
        )
        + "."
    )
    relation_text = (
        "Spearman correlation: "
        + "; ".join(
            f"ρ = {d['relation']['rho']:.2f}, p = {d['relation']['p']:.2g} in"
            f" {d['subject']}"
            for d in sessions
        )
        + "."
    )

    onset_text = (
        "in the long delays, the rate of each unit from"
        f" {onset.ENTRY_LAG_MS} ms after the entry to the first flash, over its"
        f" rate {-onset.BEFORE_ENTRY_MS[0]} to {-onset.BEFORE_ENTRY_MS[1]} ms"
        " before the entry (gray bands in the left panel), and the same for"
        f" {onset.LATE_MS[0]} to {onset.LATE_MS[1]} ms after the first flash: "
        + "; ".join(
            f"{d['subject']}: median {e['entry to first flash over before entry']['median_ratio']:.2f}"
            f" before the flash (lower in {e['entry to first flash over before entry']['units_lower']}"
            f" of {e['entry to first flash over before entry']['units']} units, two-sided"
            f" Wilcoxon signed-rank test, p = {e['entry to first flash over before entry']['wilcoxon_p']:.1g})"
            f" and {e['late over before entry']['median_ratio']:.2f} late"
            for d in raw[ONSET_NAME]
            for e in [d["long-delay trials"]]
        )
        + ". Zero rates stay in the plot and tests. The axis is linear from 0"
        " to 1/8 and logarithmic above it; all finite ratios are shown."
    )
    exit_text = (
        "Each mouse, two-sided Wilcoxon signed-rank tests over units: "
        + "; ".join(
            f"{r['subject']} ({r['early withdrawal']['trials']} early withdrawals),"
            f" after over before the exit {r['completed']['after / before exit']['median_ratio']:.2f}"
            f" in completed trials (p = {r['completed']['after / before exit']['wilcoxon_p']:.1g})"
            f" and {r['early withdrawal']['after / before exit']['median_ratio']:.2f} in early"
            f" withdrawals (p = {r['early withdrawal']['after / before exit']['wilcoxon_p']:.1g});"
            " after the exit, early withdrawal over completed"
            f" {r['early withdrawal / completed, after exit']['median_ratio']:.2f}"
            f" (p = {r['early withdrawal / completed, after exit']['wilcoxon_p']:.2g})"
            for r in raw[EXIT_NAME]
        )
        + "."
    )

    # The fixation level is shrunk by the L2 penalty (`parameter_recovery.py`, on
    # the model of the main figure).
    recovery = json.loads((args.out / "parameter_recovery.json").read_text())
    bias_text = (
        "These levels are lower bounds: in the model of the main figure, spikes"
        " simulated with a known fixation level give it back at "
        + " and ".join(
            f"{recovery[s]['fixation level, recovered over true (median)']['in_use']:.2f}"
            for s in SESSIONS
        )
        + " of its true size (median unit, "
        + " and ".join(s.split("_")[0] for s in SESSIONS)
        + "), because the penalty gives a part of it to the terms after the exit."
    )

    def outside(key: str) -> str:
        return (
            _pair(sessions, lambda d: str(st(d)[key]))
            + " units are outside the axis limits and are in the tests."
        )

    output.with_name("hold_effect_caption.md").write_text(
        CAPTION.format(
            units=_pair(sessions, lambda d: f"{st(d)['units']} units"),
            subject=first,
            other=other,
            before=_pair(sessions, lambda d: f"{st(d)['rate_before_first_flash']:.2f}"),
            late=_pair(sessions, lambda d: f"{st(d)['rate_last_200_ms_of_hold']:.2f}"),
            after_rate=_pair(
                sessions, lambda d: f"{st(d)['rate_100_to_400_ms_after_exit']:.2f}"
            ),
            lo=LEVEL_MS[0],
            hi=LEVEL_MS[1],
            paired=paired_text,
            bias=bias_text,
            outside_d=outside("units_outside_the_axis_limits_of_panel_d"),
            onset=onset_text,
            exit=exit_text,
            stillness=stillness_text,
            energy=energy_text,
            relation=relation_text,
            outside_i=outside("units_outside_the_axis_limits_of_panel_i"),
        )
    )
    print(json.dumps(stats, indent=1))
    print(output)


if __name__ == "__main__":
    main()
