"""Draw the main figure of the V1 GLM: what predicts V1 spikes in the task.

Question: which variables of the task predict the spikes of V1 units, and how
much does each one explain?

Comparison: no new fits. The script reads one fit set (`--units`, `all` by
default) of each session of `SESSIONS`, after `glm fit` and `glm unique`.
Panels a to c are the example unit of the first session (GRB006); panels d to
g are each session, side by side. The tests are within each animal; the units
of the two animals are not pooled.

- Panel a (the task and the model): one held-out trial of the example unit,
  against time from the first flash. Flashes, the fixation (first flash to center
  exit; the initiation, from the center entry to the first flash, is the
  lighter part), and the withdrawal; the first motion-energy PC; the term that each group of
  regressors adds to the log rate in the final refit (visual, task, and motion
  energy in one row, spike history in its own row, in their colors); and the
  observed spikes.
- Panel b (what the model learns): the kernels of the example unit from the
  final refit, as a change in log rate. Top: the flash kernel for a flash in
  the fixation in the two gap classes, and for a flash in the fixation and
  after the center exit in the class of the longer gaps
  (`results.flash_condition_kernels`: the flash kernel of the gap class plus
  the flash × move kernel at the value of that flash); the fixation
  kernel. Bottom: the withdrawal kernel and the spike-history kernel. The
  initiation, the two side columns, and the session drift have no kernel over
  time and are not drawn.
- Panel c (prediction): observed and predicted firing rate of the example unit
  over the held-out test trials, in `RATE_BIN_MS` bins. The prediction of a
  trial comes from the cross-validation fold that did not train on it. A time
  bin is shown while at least `MIN_TRIAL_FRACTION` of the test trials have not
  reached the response port.
- Panels d and e (population, one for each animal): cross-validated D² of each unit against its
  bits/spike (the log-likelihood gain over the constant-rate model for each
  spike), with the units in three groups of equal size by firing rate
  (`figures._plot_bits_deviance_by_rate`). For the same bits/spike, a unit with
  a higher rate has a higher D². The histograms above and at the right are the
  distributions of the two measures over units, with their medians.
- Panels f and g (what explains the spikes, one for each animal; the
  order of the regressors is that of the first session): for each regressor, the unique ΔD²
  (filled; the D² that the model loses when that regressor is shuffled in time
  within each trial) and the maximal ΔD² (open; the D² of that regressor alone
  minus the D² with every regressor shuffled); median and interquartile range
  over units. The top row is the D² of the full model. The axis is logarithmic and starts at `FLOOR`; a smaller value is
  drawn at the floor. The number at the right is the fraction of units whose
  unique ΔD² is above zero (one-sided t-test of the fold values,
  p < `RELIABLE_ALPHA`; a regressor with no effect gives about 5%). The two
  tests of `inclusion.py` for each regressor go to `main_figure_stats.json`; a
  regressor that does not pass both has "n.s." after its number.

Units, trials, alignment (first flash), baseline (constant rate), and
exclusions (bins after response entry) are those of `fit.py`. The example unit
is the one that `results.example_unit_id` selects (unit 197 when it was fit).
The sampling unit is the unit in panels d to g. The script writes
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

from thesis.ephys.analyses.glm import cli
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
    "flash_gap_1": ("flash, 40 ms gap", VISUAL),
    "flash_gap_2": ("flash, 80+ ms gap", VISUAL),
    "flash_gap_3": ("flash, 160+ ms gap", VISUAL),
    "flash_after_exit": ("flash × move", VISUAL),
    "flash_move_gap": ("flash × move × gap", VISUAL),
    "fixation": ("initiation", TASK),
    "hold": ("fixation", TASK),
    "withdrawal": ("withdrawal", TASK),
    "side_after_exit": ("choice", TASK),
    "previous_side": ("previous choice", TASK),
    "video": ("motion energy", MOTION),
    "session_drift": ("session drift", DRIFT_COLOR),
}

CAPTION = """\
**A Poisson GLM predicts V1 spikes from the visual stimulus, the task, movement,
and spike history.** The model predicts the spike count of each unit in 1-ms
bins. Each group of regressors adds a term to the log of the firing rate, so a
kernel is a change in log rate (Δ log rate): a value of 0.7 is a rate two times
as high. Colors are the same in all panels: visual (blue), task (orange),
motion energy (green), spike history (red), and model prediction (purple).
Panels a to c are the example unit of {subject}. **a**, One held-out trial of
the example unit (unit {unit}, {unit_rate:.0f} spikes/s), against time from the first flash. From the top: the events (the
flashes; the initiation, from the center-port entry to the first flash, in a
light color; the fixation, from the first flash to the withdrawal from the
center port; and the withdrawal); the first motion-energy PC of the video; the
term that each group of regressors adds to the log rate; the term of spike
history, in its own row; and the observed spikes. **b**, The kernels of the
example unit. Top, from the left: the flash kernel for a flash in the fixation that
follows another within 50 ms (40 ms gap) and for every other flash (80+ ms
gap); the flash kernel for a flash in the fixation (still) and after the
withdrawal (move, dashed), at a gap of 80 ms or more; and the fixation kernel, which starts at the first
flash and ends at the withdrawal. Bottom: the withdrawal kernel and the
spike-history kernel. **c**, Observed and
predicted firing rate of the example unit over the {test_trials} held-out test
trials, in {bin_ms}-ms bins. Each trial is predicted by a model that was not
fit on it. A time bin is shown while at least half of the trials have not
reached the response port. **d, e**, Cross-validated D² (deviance explained on
held-out trials, relative to a constant rate) against bits/spike for each of
the {units} units, one panel for each animal. Small points are units, in three groups of equal size by
firing rate (a darker gray is a higher rate). In each group, the units are in bins of equal size by bits/spike;
a large marker is the mean of a bin and the band is ±1 SEM. For the same
bits/spike, a unit with a higher rate has a higher D². The histograms are the
distributions over units (median D² {d2}; median bits/spike {bits}).
**f, g**, What each regressor explains, in each animal (the order of the
regressors is that of {subject}): the unique ΔD² (filled; the D² that the
model loses when that regressor is shuffled in time within each trial) and the
maximal ΔD² (open; the D² of that regressor alone). Points are the median and
lines the interquartile range over units; the top row is the D² of the full
model. The axis is logarithmic and starts at {floor:g}; a smaller value is
drawn there. The number at the right is the fraction of units with a unique ΔD²
above zero (one-sided t-test over the 10 folds, p < {alpha}; a regressor with
no effect gives about {alpha:.0%}). A regressor is in the model when it passes
two tests over all units, each with a Holm correction for the {regressors}
regressors: the unique ΔD² is above zero (one-sided Wilcoxon signed-rank test
over units), and more units need the regressor than chance gives (one-sided
binomial test against {alpha:.0%}; {minimum} units or more).
The tests are within each animal; the units of the two animals are not pooled.
{verdict}
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
    events.set_yticks([3, 2, 1], ["flashes", "initiation, fixation", "withdrawal"])
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


def _plot_kernels(axes, data) -> None:
    by_gap, by_period, hold, withdrawal, history = axes
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
        ("fixation", "still", "-", 1.13),
        ("after_exit", "move (dashed)", "--", 1.0),
    ):
        by_period.plot(lags, data[f"flashcond__{key}"], color=VISUAL, ls=style)
        by_period.text(
            0, height, label, color=VISUAL, fontsize=6.5, transform=by_period.transAxes
        )
    by_period.set_xlabel("time from\nflash (ms)")
    hold.plot(data["kern_t__hold"], data["kern_v__hold"], color=TASK)
    hold.set_xlabel("time in fixation,\nfrom first flash (ms)")
    withdrawal.plot(data["kern_t__withdrawal"], data["kern_v__withdrawal"], color=TASK)
    withdrawal.set_xlabel("time from\nwithdrawal (ms)")
    history.plot(data["kern_t__history"], data["kern_v__history"], color=HISTORY)
    history.set_xlabel("time since\nspike (ms)")
    for axis in axes:
        axis.axhline(0, color=REFERENCE_COLOR, linestyle="--", linewidth=0.8)
    for axis in (by_gap, withdrawal):  # one y label for each row
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


def _plot_population(axis, bits, deviance, rate) -> None:
    """D² against bits/spike by firing-rate group, with the two distributions."""
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
        np.median(bits),
        1.02,
        f"median {np.median(bits):.2f}",
        transform=above.get_xaxis_transform(),
        va="bottom",
        fontsize=6.5,
    )
    side.text(
        1.0,
        np.median(deviance),
        f" median\n {np.median(deviance):.3f}",
        transform=side.get_yaxis_transform(),
        va="center",
        fontsize=6.5,
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


def unique_order(records: list[dict], tests: dict[str, dict]) -> list[str]:
    """Return the regressors in the order of their median unique ΔD²."""
    return sorted(
        tests,
        key=lambda name: np.median([item["groups"][name][UNIQUE] for item in records]),
    )


def _plot_unique(
    axis, records: list[dict], tests: dict[str, dict], order: list[str], labels: bool
) -> None:
    """Unique (filled) and maximal (open) ΔD² of each regressor, over units."""
    values = lambda name, key: np.asarray(  # noqa: E731
        [item["groups"][name][key] for item in records]
    )
    for position, name in enumerate(order):
        color = REGRESSORS[name][1]
        for key, offset, face in ((MAXIMAL, 0.17, "white"), (UNIQUE, -0.17, color)):
            low, median, high = np.maximum(
                np.percentile(values(name, key), [25, 50, 75]), FLOOR
            )
            axis.plot([low, high], [position + offset] * 2, color=color, linewidth=1.2)
            axis.scatter(
                median,
                position + offset,
                s=22,
                zorder=3,
                facecolor=face,
                edgecolor=color,
            )
        axis.text(
            1.04,
            position,
            f"{tests[name]['fraction_units_that_need_it']:.0%}"
            + ("" if tests[name]["passes_both_tests"] else " n.s."),
            color=color,
            va="center",
            transform=axis.get_yaxis_transform(),
        )
    # The D² of the full model, on the same axis, above the regressors.
    full = np.asarray([item["complete_test_deviance_explained"] for item in records])
    low, median, high = np.percentile(full, [25, 50, 75])
    axis.plot([low, high], [len(order)] * 2, color="black", linewidth=1.2)
    axis.scatter(median, len(order), s=22, zorder=3, color="black")
    axis.text(
        1.04,
        len(order) + 0.65,
        "units with\nunique " + r"$\Delta D^2$ > 0",
        transform=axis.get_yaxis_transform(),
        fontsize=6.5,
    )
    axis.set_xscale("log")
    axis.set_xlim(FLOOR, 3e-1)
    axis.set_yticks(
        range(len(order) + 1),
        [*(REGRESSORS[name][0] for name in order), r"full model ($D^2$)"]
        if labels
        else [],
    )
    for tick, name in zip(axis.get_yticklabels(), order, strict=False):
        tick.set_color(REGRESSORS[name][1])
    axis.set_xlabel(r"$\Delta D^2$" + "\n(filled: unique; open: maximal)")


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
    parser.add_argument("--out", type=Path, default=Path("figures/glm"))
    parser.add_argument("--units", type=cli._unit_set, default="all")
    args = parser.parse_args()
    # The example unit is from the first session; the population panels are
    # from each session.
    fit_dir = cli.PoissonGLM(root=args.out / SESSIONS[0]).fit_dir(args.units)
    with np.load(fit_dir / "figure_data.npz", allow_pickle=False) as saved:
        data = {key: saved[key] for key in saved.files}
    sessions = [session_population(args.out / name, args.units) for name in SESSIONS]
    order = unique_order(sessions[0]["records"], sessions[0]["tests"])

    with plt.rc_context(FIGURE_STYLE):
        figure = plt.figure(figsize=(7.4, 12.6))
        top = figure.add_gridspec(
            1,
            2,
            left=0.14,
            right=0.985,
            top=0.97,
            bottom=0.713,
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
                1, 1, left=0.1, right=0.98, top=0.642, bottom=0.526
            )
        )
        populations = [
            figure.add_subplot(cell)
            for cell in figure.add_gridspec(
                1, 2, left=0.1, right=0.88, top=0.425, bottom=0.297, wspace=0.85
            )
        ]
        unique_axes = [
            figure.add_subplot(cell)
            for cell in figure.add_gridspec(
                1, 2, left=0.22, right=0.92, top=0.222, bottom=0.05, wspace=0.3
            )
        ]

        _plot_trial(trial_axes, data)
        _plot_kernels(kernel_axes, data)
        _plot_prediction(prediction, data)
        for k, (session, population, unique_axis) in enumerate(
            zip(sessions, populations, unique_axes, strict=True)
        ):
            _plot_population(
                population, session["bits"], session["deviance"], session["rate"]
            )
            _plot_unique(
                unique_axis, session["records"], session["tests"], order, labels=k == 0
            )
            for axis in (population, unique_axis):
                axis.text(
                    0.5 if axis is unique_axis else 0.0,
                    1.02 if axis is unique_axis else 1.45,
                    f"{session['subject']} ({session['units']} units)",
                    transform=axis.transAxes,
                    ha="center" if axis is unique_axis else "left",
                    fontsize=7.5,
                )
        for axis in (*trial_axes, *kernel_axes, *unique_axes):
            separate_axes(axis)
        for axis in trial_axes[:4]:
            axis.spines["bottom"].set_visible(False)

        figure.canvas.draw()
        to_figure = figure.transFigure.inverted()
        firsts = (trial_axes[0], kernel_axes[0], prediction, *populations, *unique_axes)
        for letter, axis in zip("abcdefg", firsts, strict=True):
            box = axis.yaxis.get_tightbbox()
            left = (
                box.transformed(to_figure).x0
                if box and axis is not unique_axes[1]
                else axis.get_position().x0 - 0.03
            )
            top_edge = axis.get_position().y1 + (0.056 if axis in populations else 0.02)
            figure.text(left, top_edge, letter, fontweight="bold", fontsize=10)
        output = save_figure(figure, args.out / "main_figure")
    output.with_name("main_figure_stats.json").write_text(
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
            unit=scalars["example_unit_id"],
            unit_rate=scalars["example_unit_training_rate_spikes_s"],
            test_trials=scalars["test_trials"],
            bin_ms=RATE_BIN_MS,
            units=each(lambda s: str(s["units"])),
            d2=each(lambda s: f"{np.median(s['deviance']):.3f}"),
            bits=each(lambda s: f"{np.median(s['bits']):.2f}"),
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
