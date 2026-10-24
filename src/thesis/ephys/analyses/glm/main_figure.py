"""Draw the main figure of the V1 GLM: what predicts V1 spikes in the task.

Question: which variables of the task predict the spikes of V1 units, and how
much does each one explain?

Comparison: no new fits. The script reads one fit set (`--units`, `all` by
default) of each session of `SESSIONS`, after `glm fit` and `glm unique`.
Panels b to d are the example unit of the first session (GRB006). In panels e
and f the units of both mice are pooled (standard practice; the caption gives
the units and mice); a summary is the pooled median with its
hierarchical-bootstrap 95% interval (mice, then units;
`plotting.hierarchical_interval`). The inclusion tests are within each mouse.

- Panel a (the design): each term of the model over one schematic trial
  (`SCHEMATIC`: the median times of the two mice), from the switches of
  `dmat.py`: ticks at the flashes that a flash kernel models, bars for the time
  in which a term is on, and the number of columns of each term. Before the
  center entry no event term is on: each event kernel is relative to that
  time. Motion energy, spike history, the session drift, and the previous
  choice are on there too, so the baseline is the rate before the entry at
  the mean of those terms.
- Panel b (the task and the model): one held-out trial of the example unit,
  against time from the first flash. The events (flashes, the pre-stimulus
  time from the center entry to the first flash in a lighter color, the
  fixation from the first flash to the center exit, and the center exit); the
  first motion-energy PC; the term that each group of regressors adds to the
  log rate in the final refit (visual, task, and motion energy in one row,
  spike history in its own row, in their colors); and the observed spikes.
- Panel c (what the model learns): the kernels of the example unit from the
  final refit, as a change in log rate. Top: the flash kernel for a flash in
  the fixation in the two gap classes, and for a flash in the fixation and
  after the center exit in the class of the longer gaps
  (`results.flash_condition_kernels`); the fixation kernel. Bottom: the term
  after the center exit for a right and a left turn (the response period plus
  and minus turn direction), and the spike-history kernel. The pre-stimulus
  column, the previous choice, and the session drift have no kernel over time
  and are not drawn.
- Panel d (prediction): observed and predicted firing rate of the example unit
  over the held-out test trials, in `RATE_BIN_MS` bins. The prediction of a
  trial comes from the cross-validation fold that did not train on it. A time
  bin is shown while at least `MIN_TRIAL_FRACTION` of the test trials have not
  reached the response port.
- Panel e (population): cross-validated D² of each unit against its
  bits/spike, in three groups of equal size by firing rate
  (`figures._plot_bits_deviance_by_rate`), with the distributions of the two
  measures and their medians.
- Panel f (what explains the spikes): for each regressor, the unique ΔD² and
  the maximal ΔD²: the pooled median with its hierarchical-bootstrap 95%
  interval. The axis is logarithmic and starts at `FLOOR`. The numbers at the
  right are the fraction of the units of both mice that need the regressor; a
  dagger marks a regressor that fails the two tests of `inclusion.py` in one
  mouse, and n.s. one that fails in both (`main_figure_stats.json`).

Units, trials, alignment (first flash), baseline (constant rate), and
exclusions (bins after response entry) are those of `fit.py`. The example unit
is the one that `results.example_unit_id` selects (unit 197 when it was fit).
The sampling unit is the unit in panels e and f. The script writes
`main_figure.svg`, `main_figure_stats.json` (by session), and
`main_figure_caption.md` in `--out`.

    uv run python -m thesis.ephys.analyses.glm.main_figure
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from thesis.ephys.analyses.glm import cli, dmat
from thesis.ephys.analyses.glm.dmat import BINWIDTH_S
from thesis.ephys.analyses.glm.figures import (
    _plot_bits_deviance_by_rate,
    training_rate_and_deviance,
)
from thesis.ephys.analyses.glm.fit import load_unit_records
from thesis.ephys.analyses.glm.flash_kernels import SESSIONS
from thesis.ephys.analyses.glm.inclusion import decide, minimum_units, regressor_tests
from thesis.ephys.analyses.glm.results import FLASH_GAPS_MS
from thesis.ephys.analyses.glm.unique_dev import MAXIMAL, RELIABLE_ALPHA, UNIQUE
from thesis.plotting import (
    CONCEPT_COLORS,
    FIGURE_STYLE,
    MODEL_COLOR,
    REFERENCE_COLOR,
    hierarchical_interval,
    save_figure,
    separate_axes,
)

FLOOR = 1e-4
RATE_BIN_MS = 20
MIN_TRIAL_FRACTION = 0.5
VISUAL, TASK, MOTION, HISTORY = (
    CONCEPT_COLORS[name] for name in ("visual", "task", "motion energy", "history")
)
DRIFT_COLOR = "0.4"  # not a variable of the task
REGRESSORS = {
    "history": ("spike history", HISTORY),
    "visual_flash": ("flash", VISUAL),
    "flash_just_before": ("flash × short gap", VISUAL),
    "flash_gap_1": ("flash, 40 ms gap", VISUAL),
    "flash_gap_2": ("flash, 80+ ms gap", VISUAL),
    "flash_gap_3": ("flash, 160+ ms gap", VISUAL),
    "flash_after_exit": ("flash × after exit", VISUAL),
    "flash_recovery": ("flash × recovery", VISUAL),
    "fixation": ("pre-stimulus", TASK),
    "hold": ("fixation", TASK),
    "withdrawal": ("withdrawal", TASK),
    "previous_side": ("previous choice", TASK),
    "video": ("motion energy", MOTION),
    "session_drift": ("session drift", DRIFT_COLOR),
    "response_period": ("response period", TASK),
    "turn_direction": ("turn direction", TASK),
    "previous_outcome": ("previous outcome", TASK),
}

CAPTION = """\
**A Poisson GLM predicts V1 spikes from the visual stimulus, the task, movement,
and spike history.** The model predicts the spike count of each unit in 1-ms
bins. Each group of regressors adds a term to the log of the firing rate, so a
kernel is a change in log rate (Δ log rate): a value of 0.7 is a rate two times
as high. Colors are the same in all panels: visual (blue), task (orange),
motion energy (green), spike history (red), and model prediction (purple).
**a**, The design: each term over one schematic trial, with the time in which
it is on (bars) or the flashes it models (ticks), and its number of columns.
Before the center entry no event term is on; each event kernel is relative to
that time, at the mean of motion energy, spike history, the session drift, and
the previous choice, which are on there too. Panels b
to d are the example unit of {subject}. **b**, One held-out trial of the
example unit (unit {unit}, {unit_rate:.0f} spikes/s), against time from the
first flash. From the top: the events (the flashes; the pre-stimulus time, from
the center-port entry to the first flash, in a light color; the fixation, from
the first flash to the center exit; and the exit); the first motion-energy PC
of the video; the term that each group of regressors adds to the log rate; the
term of spike history, in its own row; and the observed spikes. **c**, The
kernels of the example unit. Top, from the left: the flash kernel for a flash
in the fixation that follows another within 50 ms (40 ms gap) and for every
other flash (80+ ms gap); the flash kernel for a flash in the fixation (in port)
and after the exit (dashed), at a gap of 80 ms or more; and the fixation
kernel, which starts at the first flash and ends at the exit. Bottom: the term
after the exit for a right and a left turn (response period plus and minus
turn direction) and the spike-history kernel. **d**, Observed and predicted
firing rate of the example unit over the {test_trials} held-out test trials, in
{bin_ms}-ms bins. Each trial is predicted by a model that was not fit on it. A
time bin is shown while at least half of the trials have not reached the
response port. **e**, Cross-validated D² (deviance explained on held-out
trials, relative to a constant rate) against bits/spike for each of the {units}
units of the two mice. Small points are units, in three groups of equal size by
firing rate (a darker gray is a higher rate). In each group, the units are in
bins of equal size by bits/spike; a large marker is the mean of a bin and the
band is ±1 SEM. The histograms are the distributions over units (median D²
{d2}; median bits/spike {bits}). **f**, What each regressor explains (in the
order of {subject}): the unique ΔD² (filled; the D² that the model loses when
that regressor is shuffled: inside each trial, or over trials for a variable
of the trial) and the maximal ΔD² (open; the D² of that regressor alone).
Markers are the median over the pooled units with its hierarchical-bootstrap
95% interval (the mice, then the units of each). The top row is the D² of the
full model. The axis is logarithmic and starts at {floor:g}; a smaller value is
drawn there. The numbers at the right are the fraction of the units of both
mice with a unique ΔD² above zero (one-sided t-test over the 10 folds, p <
{alpha}; a regressor with no effect gives about {alpha:.0%}). A regressor is in
the model when it passes two tests over the units of each mouse, each with a
Holm correction for the {regressors} regressors: the unique ΔD² is above zero
(one-sided Wilcoxon signed-rank test over units), and more units need the
regressor than chance gives (one-sided binomial test against {alpha:.0%};
{minimum} units or more). A dagger marks a regressor that fails in one mouse,
and n.s. one that fails in both. {verdict}
"""


def _plot_trial(axes, data) -> None:
    """One trial: events, motion-energy PC 1, the fitted terms, and the spikes."""
    events, motion, terms, history, spikes = axes
    times = 1000 * data["model_times"]
    left, right = times[0], times[-1]
    entry, leave = 1000 * data["event__hold"]  # first flash to center exit
    events.axvspan(
        max(entry, left), leave, ymin=0.38, ymax=0.62, color=TASK, alpha=0.4, lw=0
    )
    # The initiation (center entry to first flash), lighter, in the same row.
    start = 1000 * data["event__fixation"][0]
    events.axvspan(
        max(start, left), entry, ymin=0.38, ymax=0.62, color=TASK, alpha=0.18, lw=0
    )
    events.vlines(1000 * data["event__visual_flash"], 2.6, 3.4, color=VISUAL, lw=1)
    events.vlines(1000 * data["event__withdrawal"], 0.6, 1.4, color=TASK, lw=1.2)
    events.set_ylim(0.3, 3.7)
    events.set_yticks([3, 2, 1], ["flashes", "pre-stimulus, fixation", "center exit"])
    for tick, color in zip(events.get_yticklabels(), (VISUAL, TASK, TASK), strict=True):
        tick.set_color(color)
    shown = (1000 * data["motion_times"] >= left) & (
        1000 * data["motion_times"] <= right
    )
    motion.plot(
        1000 * data["motion_times"][shown],
        data["motion_scores"][shown, 0],
        color=MOTION,
    )
    motion.set_ylabel("motion\nenergy PC 1", color=MOTION)
    # Each group of regressors adds its term to the log rate.
    # Spike history has its own row: for a unit with a high rate it hides the rest.
    for key, color in (("video", MOTION), ("task", TASK), ("visual", VISUAL)):
        terms.plot(times, data[f"contrib__{key}"], color=color, lw=0.9)
    history.plot(times, data["contrib__history"], color=HISTORY, lw=0.6)
    for axis in (terms, history):
        axis.axhline(0, color=REFERENCE_COLOR, linestyle="--", linewidth=0.8)
        axis.set_ylabel(r"$\Delta$ log rate")
    spikes.vlines(
        np.repeat(times, data["model_counts"].astype(int)), 0, 1, color="black", lw=0.7
    )
    spikes.set_yticks([0.5], ["spikes"])
    spikes.set_ylim(-0.2, 1.2)
    spikes.set_xlabel("time from first flash (ms)")
    for axis in axes:
        axis.set_xlim(left, right)
    for axis in (events, motion, terms, history):  # one time axis, under the spikes
        axis.tick_params(bottom=False, labelbottom=False)


# One schematic trial for the design panel (s from the first flash): the center
# entry, the center exit, and the response entry at about the median times of
# the two mice, and a flash train in slots of `dmat.FLASH_SLOT_S`.
SCHEMATIC = {"entry": -0.05, "exit": 0.55, "response": 1.15}
SCHEMATIC_FLASHES = (
    np.array([0, 2, 3, 5, 8, 9, 11, 13, 15, 16, 19, 21, 22, 24]) * dmat.FLASH_SLOT_S
)


def design_rows(components: int) -> list[tuple]:
    """Return (key, columns, kind, start, stop, note) for each term of the design.

    `kind` is "flash" (ticks at the flashes it models; `start` names them) or
    "span" (a bar). The rows follow the switches of `dmat`, so the panel shows
    the design in use.
    """
    entry, leave, response = (SCHEMATIC[key] for key in ("entry", "exit", "response"))
    start = -dmat.PRE_S
    rows = []
    if dmat.FLASH_JUST_BEFORE_TERM and not dmat.FLASH_GAP_EDGES_S:
        rows += [
            ("visual_flash", dmat.FLASH_BASES, "flash", "all", None, "every flash"),
            (
                "flash_just_before",
                dmat.FLASH_BASES,
                "flash",
                "short",
                None,
                f"a flash ≤ {1000 * dmat.FLASH_JUST_BEFORE_S:.0f} ms after another",
            ),
        ]
    if dmat.FLASH_GAP_EDGES_S:
        rows += [
            (
                "flash_gap_1",
                dmat.FLASH_BASES,
                "flash",
                "short",
                None,
                "a flash ≤ 50 ms after another",
            ),
            (
                "flash_gap_2",
                dmat.FLASH_BASES,
                "flash",
                "long",
                None,
                "every other flash",
            ),
        ]
    if dmat.FLASH_AFTER_EXIT_TERM:
        rows.append(
            (
                "flash_after_exit",
                dmat.FLASH_BASES,
                "flash",
                "after",
                None,
                "flashes after the exit",
            )
        )
    if dmat.FIXATION_COLUMN:
        rows.append(("fixation", 1, "span", entry, 0.0, "entry to first flash"))
    if dmat.HOLD_KERNEL:
        rows.append(
            (
                "hold",
                dmat.HOLD_BASES,
                "span",
                0.0,
                leave,
                f"to {dmat.HOLD_WINDOW_S:g} s, then held",
            )
        )
    if dmat.WITHDRAWAL_KERNEL:
        rows.append(
            (
                "withdrawal",
                dmat.WITHDRAWAL_BASES,
                "span",
                leave,
                leave + dmat.WITHDRAWAL_WINDOW_S,
                "",
            )
        )
    if dmat.RESPONSE_PERIOD_TERM:
        rows.append(
            (
                "response_period",
                dmat.TURN_DIRECTION_BASES,
                "span",
                leave,
                response,
                f"to {dmat.TURN_DIRECTION_S:g} s, then held",
            )
        )
    if dmat.TURN_DIRECTION_TERM:
        rows.append(
            (
                "turn_direction",
                dmat.TURN_DIRECTION_BASES,
                "span",
                leave,
                response,
                "± side of the response",
            )
        )
    if dmat.PREVIOUS_SIDE_COLUMN:
        rows.append(
            ("previous_side", 1, "span", start, leave, "side of the trial before")
        )
    if dmat.SESSION_DRIFT:
        rows.append(
            (
                "session_drift",
                dmat.DRIFT_BASES - 1,
                "span",
                start,
                response,
                "constant inside a trial",
            )
        )
    rows.append(("video", components, "span", start, response, "PCs of the video"))
    rows.append(
        (
            "history",
            dmat.HISTORY_COLUMNS,
            "span",
            start,
            response,
            "the unit's own spikes",
        )
    )
    return rows


def _plot_design(axis, components: int) -> None:
    """Draw panel a: each term of the design over one schematic trial."""
    rows = design_rows(components)
    entry, leave, response = (SCHEMATIC[key] for key in ("entry", "exit", "response"))
    start = -dmat.PRE_S
    flashes = SCHEMATIC_FLASHES
    gaps = np.diff(flashes, prepend=-np.inf)
    edge = dmat.short_gap_s()
    picked = {
        "all": flashes,
        "short": flashes[gaps <= edge],
        "long": flashes[gaps > edge],
        "after": flashes[flashes >= leave],
    }
    # Before the entry no event term is on: the baseline of the event kernels.
    axis.axvspan(start, entry, color=REFERENCE_COLOR, alpha=0.15, lw=0)
    for time in (entry, 0.0, leave, response):
        axis.axvline(time, color=REFERENCE_COLOR, lw=0.6, ls=":")
    for row, (key, columns, kind, first, last, note) in enumerate(rows):
        label, color = REGRESSORS[key]
        y = len(rows) - row
        if kind == "flash":
            times = picked[first]
            axis.vlines(times, y - 0.32, y + 0.32, color=color, lw=1.2)
            note_x = times[-1] + 0.03
        else:
            axis.barh(
                y, last - first, left=first, height=0.55, color=color, alpha=0.55, lw=0
            )
            note_x = last + 0.03
        axis.text(note_x, y, note, va="center", fontsize=6, color=color)
        axis.text(
            start - 0.03, y, label, va="center", ha="right", fontsize=7, color=color
        )
        axis.text(
            1.0,
            y,
            str(columns),
            transform=axis.get_yaxis_transform(),
            va="center",
            ha="right",
            fontsize=7,
        )
    top = len(rows) + 1.0
    axis.text(
        1.0,
        top,
        f"columns ({sum(row[1] for row in rows)})",
        transform=axis.get_yaxis_transform(),
        ha="right",
        va="center",
        fontsize=6.5,
    )
    for time, label, align in (
        (entry, "center entry", "right"),
        (0.0, "first flash", "left"),
        (leave, "center exit", "center"),
        (response, "response entry", "center"),
    ):
        axis.text(time, top, label, ha=align, va="center", fontsize=6.5)
    axis.text(
        (start + entry) / 2,
        0.2,
        "no event term (baseline)",
        ha="center",
        fontsize=6,
        color="0.4",
    )
    axis.set_xlim(start, response + 0.55)
    axis.set_ylim(-0.2, top + 0.6)
    axis.set_yticks([])
    axis.set_xticks([start, 0, 0.5, 1.0], [f"{start:g}", "0", "0.5", "1"])
    axis.set_xlabel("time from first flash (s), one schematic trial")
    axis.spines["left"].set_visible(False)


def _plot_kernels(axes, data) -> None:
    by_gap, by_period, hold, after, history = axes
    lags = data["flash_lags"]
    # The flash kernel by the gap from the flash before: darker for a longer gap.
    for gap, label, shade, height in zip(
        FLASH_GAPS_MS, ("40 ms gap", "80+ ms gap"), (0.5, 1.0), (1.0, 1.13), strict=True
    ):
        by_gap.plot(lags, data[f"flashcond__gap_{gap}"], color=VISUAL, alpha=shade)
        by_gap.text(
            0,
            height,
            label,
            color=VISUAL,
            alpha=shade,
            fontsize=6.5,
            transform=by_gap.transAxes,
        )
    by_gap.set_xlabel("time from\nflash (ms)")
    for key, label, style, height in (
        ("fixation", "in port", "-", 1.13),
        ("after_exit", "after exit (dashed)", "--", 1.0),
    ):
        by_period.plot(lags, data[f"flashcond__{key}"], color=VISUAL, ls=style)
        by_period.text(
            0, height, label, color=VISUAL, fontsize=6.5, transform=by_period.transAxes
        )
    by_period.set_xlabel("time from\nflash (ms)")
    hold.plot(data["kern_t__hold"], data["kern_v__hold"], color=TASK)
    hold.set_xlabel("time from\nfirst flash (ms)")
    # After the exit: the response period plus or minus turn direction, the
    # term of a right and of a left turn.
    times = data["kern_t__response_period"]
    for sign, label, style, height in (
        (1, "right turn", "-", 1.13),
        (-1, "left turn (dashed)", "--", 1.0),
    ):
        after.plot(
            times,
            data["kern_v__response_period"] + sign * data["kern_v__turn_direction"],
            color=TASK,
            ls=style,
        )
        after.text(
            0, height, label, color=TASK, fontsize=6.5, transform=after.transAxes
        )
    after.set_xlabel("time from\ncenter exit (ms)")
    history.plot(data["kern_t__history"], data["kern_v__history"], color=HISTORY)
    history.set_xlabel("time since\nspike (ms)")
    for axis in axes:
        axis.axhline(0, color=REFERENCE_COLOR, linestyle="--", linewidth=0.8)
    for axis in (by_gap, after):  # one y label for each row
        axis.set_ylabel(r"$\Delta$ log rate")


def _plot_prediction(axis, data) -> None:
    valid = data["test_valid"]
    width = RATE_BIN_MS
    bins = valid.shape[1] // width
    grid = lambda values: values[:, : bins * width].reshape(len(values), bins, width)  # noqa: E731
    exposure = grid(valid.astype(float)).sum((0, 2))
    keep = grid(valid.astype(float)).all(2).sum(0) >= MIN_TRIAL_FRACTION * len(valid)
    times = 1000 * grid(np.broadcast_to(data["relative_times"], valid.shape)).mean(
        (0, 2)
    )
    for values, color, label in (
        (data["prediction_observed"], "black", "observed"),
        (data["conditional"], MODEL_COLOR, "predicted"),
    ):
        rate = grid(np.where(valid, values, 0)).sum((0, 2)) / exposure / BINWIDTH_S
        axis.plot(times[keep], rate[keep], color=color)
        axis.text(
            0.98,
            0.95 - 0.1 * (label == "predicted"),
            label,
            color=color,
            transform=axis.transAxes,
            ha="right",
            va="top",
        )
    axis.set_ylim(bottom=0)
    axis.set_xlabel("time from first flash (ms)")
    axis.set_ylabel("firing rate (spikes/s)")


def _plot_population(axis, sessions: list[dict]) -> None:
    """D² against bits/spike by firing-rate group, units of all mice pooled."""
    bits, deviance, rate = (
        np.concatenate([s[key] for s in sessions])
        for key in ("bits", "deviance", "rate")
    )
    _plot_bits_deviance_by_rate(axis, bits, deviance, rate)
    axis.set_xlabel("bits/spike")
    axis.set_ylabel(r"cross-validated $D^2$")
    above = axis.inset_axes([0, 1.04, 1, 0.22], sharex=axis)
    side = axis.inset_axes([1.04, 0, 0.22, 1], sharey=axis)
    above.hist(bits, bins=24, color="black", alpha=0.75)
    side.hist(deviance, bins=24, color="black", alpha=0.75, orientation="horizontal")
    above.axvline(np.median(bits), color="black", linewidth=1.2)
    side.axhline(np.median(deviance), color="black", linewidth=1.2)
    above.text(
        0.45,
        1.0,
        f"{len(bits)} units, {len(sessions)} mice. Median {np.median(bits):.2f}"
        f" bits/spike, D² {np.median(deviance):.3f}",
        transform=above.transAxes,
        ha="left",
        va="bottom",
        fontsize=6,
    )
    for inset in (above, side):  # only the shape of each distribution
        inset.axis("off")


def inclusion_tests(records: list[dict]) -> dict[str, dict]:
    """Return the two tests of `inclusion.py` for each regressor of the figure."""
    return decide(
        {
            name: regressor_tests(
                [
                    np.asarray(item["groups"][name]["folds"]["unique"])
                    for item in records
                ]
            )
            for name in REGRESSORS
            if name in records[0]["groups"]
        }
    )


def _plot_unique(axis, sessions: list[dict], order: list[str]) -> None:
    """Unique (filled) and maximal (open) ΔD² of each regressor, units of all mice pooled.

    The marker is the pooled median with its hierarchical-bootstrap 95%
    interval (mice, then units).
    """

    def draw(y, groups, color, face):
        stats = hierarchical_interval(groups)
        low, high = np.maximum(stats["interval_95"], FLOOR)
        axis.plot([low, high], [y, y], color=color, linewidth=1.0)
        axis.scatter(
            max(stats["median"], FLOOR),
            y,
            s=22,
            zorder=3,
            facecolor=face,
            edgecolor=color,
        )

    for position, name in enumerate(order):
        color = REGRESSORS[name][1]
        for key, offset, face in ((UNIQUE, -0.17, color), (MAXIMAL, 0.17, "white")):
            draw(
                position + offset,
                [
                    [item["groups"][name][key] for item in s["records"]]
                    for s in sessions
                ],
                color,
                face,
            )
        # The fraction of the units of all mice that need it; "†" when it fails
        # the rule in one mouse, " n.s." in every mouse.
        fraction = sum(s["tests"][name]["units_that_need_it"] for s in sessions) / sum(
            len(s["records"]) for s in sessions
        )
        failed = sum(not s["tests"][name]["passes_both_tests"] for s in sessions)
        mark = "" if not failed else (" n.s." if failed == len(sessions) else "†")
        axis.text(
            1.03,
            position,
            f"{fraction:.0%}{mark}",
            color=color,
            va="center",
            fontsize=6.5,
            transform=axis.get_yaxis_transform(),
        )
    draw(
        len(order),
        [
            [item["complete_test_deviance_explained"] for item in s["records"]]
            for s in sessions
        ],
        "black",
        "black",
    )
    axis.text(
        1.03,
        len(order) + 0.5,
        "units with unique\n" + r"$\Delta D^2$ > 0",
        transform=axis.get_yaxis_transform(),
        fontsize=6,
    )
    axis.set_xscale("log")
    axis.set_xlim(FLOOR, 3e-1)
    axis.set_yticks(
        range(len(order) + 1),
        [*(REGRESSORS[name][0] for name in order), r"full model ($D^2$)"],
    )
    for tick, name in zip(axis.get_yticklabels(), order, strict=False):
        tick.set_color(REGRESSORS[name][1])
    axis.set_xlabel(r"$\Delta D^2$ (filled: unique; open: maximal)")


def session_population(root: Path, unit_set: str) -> dict:
    """Return the population measures and the inclusion tests of one session."""
    fit_dir = cli.PoissonGLM(root=root).fit_dir(unit_set)
    final = load_unit_records(fit_dir / "checkpoints", "unit_*_final.json")
    selections = load_unit_records(fit_dir / "checkpoints", "unit_*_validation.json")
    records = load_unit_records(fit_dir / "checkpoints" / "unique", "unit_*.json")
    rate, deviance = training_rate_and_deviance(final, selections)
    tests = inclusion_tests(records)
    return {
        "subject": root.name.split("_")[0],
        "units": len(final),
        "rate": rate,
        "deviance": deviance,
        "bits": np.asarray([item["cross_validated_bits_per_spike"] for item in final]),
        "records": records,
        "tests": tests,
        "failed": [
            REGRESSORS[name][0]
            for name in tests
            if not tests[name]["passes_both_tests"]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("results/glm"))
    parser.add_argument("--units", type=cli._unit_set, default="all")
    args = parser.parse_args()
    # The example unit is from the first session; the population panels are
    # from each session.
    fit_dir = cli.PoissonGLM(root=args.out / SESSIONS[0]).fit_dir(args.units)
    with np.load(fit_dir / "figure_data.npz", allow_pickle=False) as saved:
        data = {key: saved[key] for key in saved.files}
    sessions = [session_population(args.out / name, args.units) for name in SESSIONS]
    # The regressors in the order of their median unique ΔD².
    order = sorted(
        sessions[0]["tests"],
        key=lambda name: np.median(
            [item["groups"][name][UNIQUE] for item in sessions[0]["records"]]
        ),
    )

    with plt.rc_context(FIGURE_STYLE):
        figure = plt.figure(figsize=(7.4, 12.6))
        (design_axis,) = (
            figure.add_subplot(cell)
            for cell in figure.add_gridspec(
                1, 1, left=0.2, right=0.96, top=0.975, bottom=0.79
            )
        )
        top = figure.add_gridspec(
            1,
            2,
            left=0.14,
            right=0.985,
            top=0.735,
            bottom=0.48,
            wspace=0.3,
            width_ratios=(1, 1.5),
        )
        trial = top[0].subgridspec(
            5, 1, height_ratios=(1.1, 0.8, 1.1, 0.8, 0.45), hspace=0.18
        )
        trial_axes = [figure.add_subplot(cell) for cell in trial]
        kernel_axes = [
            figure.add_subplot(cell)
            for cell in list(top[1].subgridspec(2, 3, hspace=0.85, wspace=0.55))[:5]
        ]
        (prediction,) = (
            figure.add_subplot(cell)
            for cell in figure.add_gridspec(
                1, 1, left=0.1, right=0.98, top=0.415, bottom=0.295
            )
        )
        population, unique_axis = (
            figure.add_subplot(cell)
            for cell in figure.add_gridspec(
                1,
                2,
                left=0.1,
                right=0.86,
                top=0.19,
                bottom=0.05,
                wspace=0.95,
                width_ratios=(1, 1.05),
            )
        )
        components = int(
            json.loads((fit_dir / "validation_summary.json").read_text())[
                "selected_video_components"
            ]
        )
        _plot_design(design_axis, components)
        _plot_trial(trial_axes, data)
        _plot_kernels(kernel_axes, data)
        _plot_prediction(prediction, data)
        _plot_population(population, sessions)
        _plot_unique(unique_axis, sessions, order)
        for axis in (*trial_axes, *kernel_axes, unique_axis):
            separate_axes(axis)
        for axis in trial_axes[:4]:
            axis.spines["bottom"].set_visible(False)

        figure.canvas.draw()
        to_figure = figure.transFigure.inverted()
        firsts = (
            design_axis,
            trial_axes[0],
            kernel_axes[0],
            prediction,
            population,
            unique_axis,
        )
        for letter, axis in zip("abcdef", firsts, strict=True):
            left = (
                0.02
                if axis is design_axis
                else axis.yaxis.get_tightbbox().transformed(to_figure).x0
            )
            top_edge = axis.get_position().y1 + (0.06 if axis is population else 0.02)
            figure.text(left, top_edge, letter, fontweight="bold", fontsize=10)
        output = save_figure(figure, args.out / "main_figure")
    (args.out / "main_figure_stats.json").write_text(
        json.dumps(
            {
                name: {
                    "units": len(session["records"]),
                    "units_above_chance_for_these_regressors": minimum_units(
                        len(session["records"]), len(session["tests"])
                    ),
                    "inclusion_tests": session["tests"],
                }
                for name, session in zip(SESSIONS, sessions, strict=True)
            },
            indent=1,
        )
    )
    scalars = json.loads((fit_dir / "figure_data.json").read_text())
    each = lambda text: " and ".join(  # noqa: E731
        text(session) + f" ({session['subject']})" for session in sessions
    )
    output.with_name("main_figure_caption.md").write_text(
        CAPTION.format(
            subject=sessions[0]["subject"],
            other=sessions[1]["subject"],
            unit=scalars["example_unit_id"],
            unit_rate=scalars["example_unit_training_rate_spikes_s"],
            test_trials=scalars["test_trials"],
            bin_ms=RATE_BIN_MS,
            units=sum(s["units"] for s in sessions),
            d2=f"{np.median(np.concatenate([s['deviance'] for s in sessions])):.3f}",
            bits=f"{np.median(np.concatenate([s['bits'] for s in sessions])):.2f}",
            floor=FLOOR,
            alpha=RELIABLE_ALPHA,
            regressors=len(sessions[0]["tests"]),
            minimum=each(
                lambda s: (
                    f"{minimum_units(len(s['records']), len(s['tests']))} of {s['units']}"
                )
            ),
            verdict=" ".join(
                f"{s['subject']}: "
                + (
                    "every regressor passes both tests."
                    if not s["failed"]
                    else "not passing both tests (n.s.): "
                    + ", ".join(s["failed"])
                    + "."
                )
                for s in sessions
            ),
        )
    )
    print(output)


if __name__ == "__main__":
    main()
