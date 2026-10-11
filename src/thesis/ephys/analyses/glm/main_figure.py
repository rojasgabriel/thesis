"""Draw the main figure of the V1 GLM: what predicts V1 spikes in the task.

Question: which variables of the task predict the spikes of V1 units, and how
much does each one explain?

Comparison: no new fits. The script reads one fit set (`--units`, `all` by
default) of one session, after `glm fit` and `glm unique`.

- Panel a (the task and the model inputs): one held-out trial of the example
  unit. Flashes, the time in the center port, the withdrawal, the first
  motion-energy PC, and the observed spikes, against time from the first flash.
- Panel b (what the model learns): the kernels of the example unit from the
  final refit, as a change in log rate. Flash kernel for a flash with no flash
  in the last 50 ms and for a flash that follows one; withdrawal kernel; the
  step for the time in the center port; spike-history kernel.
- Panel c (prediction): observed and predicted firing rate of the example unit
  over the held-out test trials, in `RATE_BIN_MS` bins. The prediction of a
  trial comes from the cross-validation fold that did not train on it. A time
  bin is shown while at least `MIN_TRIAL_FRACTION` of the test trials have not
  reached the response port.
- Panel d (population): cross-validated D² of each unit; the line is the
  median.
- Panel e (what explains the spikes): unique ΔD² of each regressor, the D² that
  the model loses when that regressor is shuffled in time within each trial;
  median and interquartile range over units. The axis is logarithmic and starts
  at `FLOOR`; a smaller quartile is drawn at the floor.

Units, trials, alignment (first flash), baseline (constant rate), and
exclusions (bins after response entry) are those of `fit.py`. The example unit
is the one that `results.example_unit_id` selects (unit 197 when it was fit).
The sampling unit is the unit in panels d and e.

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
from thesis.ephys.analyses.glm.fit import load_unit_records
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
REGRESSORS = {
    "history": ("spike history", HISTORY),
    "visual_flash": ("visual flash", VISUAL),
    "flash_adaptation": ("flash adaptation", VISUAL),
    "in_port": ("in center port", TASK),
    "withdrawal": ("withdrawal", TASK),
    "video": ("motion energy", MOTION),
}


def _plot_trial(axes, data) -> None:
    """One trial: the events, the first motion-energy PC, and the spikes."""
    events, motion, spikes = axes
    times = 1000 * data["model_times"]
    left, right = times[0], times[-1]
    entry, leave = 1000 * data["event__in_port"]
    events.axvspan(
        max(entry, left), leave, ymin=0.38, ymax=0.62, color=TASK, alpha=0.4, lw=0
    )
    events.vlines(1000 * data["event__visual_flash"], 2.6, 3.4, color=VISUAL, lw=1)
    events.vlines(1000 * data["event__withdrawal"], 0.6, 1.4, color=TASK, lw=1.2)
    events.set_ylim(0.3, 3.7)
    events.set_yticks([3, 2, 1], ["flashes", "in center port", "withdrawal"])
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
    spikes.vlines(
        np.repeat(times, data["model_counts"].astype(int)), 0, 1, color="black", lw=0.7
    )
    spikes.set_yticks([0.5], ["spikes"])
    spikes.set_ylim(-0.2, 1.2)
    spikes.set_xlabel("time from first flash (ms)")
    for axis in axes:
        axis.set_xlim(left, right)
    for axis in (events, motion):  # one time axis, under the spikes
        axis.tick_params(bottom=False, labelbottom=False)


def _plot_kernels(axes, data) -> None:
    flash, withdrawal, port, history = axes
    for key, label, style, height in (
        ("flash_isolated", "no flash in the last 50 ms", "-", 1.12),
        ("flash_repeat", "after a flash (dashed)", "--", 1.0),
    ):
        flash.plot(
            data[f"kern_t__{key}"], data[f"kern_v__{key}"], color=VISUAL, ls=style
        )
        flash.text(
            0, height, label, color=VISUAL, fontsize=6.5, transform=flash.transAxes
        )
    flash.set_xlabel("time from flash (ms)")
    withdrawal.plot(data["kern_t__withdrawal"], data["kern_v__withdrawal"], color=TASK)
    withdrawal.set_xlabel("time from withdrawal (ms)")
    weight = float(data["kern_v__in_port"][0])
    port.step([-0.5, 0, 1, 1.5], [0, weight, 0, 0], where="post", color=TASK)
    port.set_xticks([0, 1], ["entry", "exit"])
    port.set_xlabel("in center port")
    history.plot(data["kern_t__history"], data["kern_v__history"], color=HISTORY)
    history.set_xlabel("time since spike (ms)")
    for axis in axes:
        axis.axhline(0, color=REFERENCE_COLOR, linestyle="--", linewidth=0.8)
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


def _plot_population(axis, final: list[dict]) -> None:
    scores = np.asarray([item["cross_validated_deviance_explained"] for item in final])
    median = float(np.median(scores))
    axis.hist(scores, bins=24, color="black", alpha=0.75)
    axis.axvline(median, color="black", linewidth=1.5)
    axis.text(
        median,
        1.01,
        f"median = {median:.3f}",
        transform=axis.get_xaxis_transform(),
        ha="center",
        va="bottom",
    )
    axis.set_xlabel(r"cross-validated $D^2$")
    axis.set_ylabel("units")


def _plot_unique(axis, groups: dict) -> None:
    order = sorted(REGRESSORS, key=lambda name: groups[name]["median"])
    for position, name in enumerate(order):
        color = REGRESSORS[name][1]
        low, median, high = (
            max(groups[name][key], FLOOR) for key in ("q25", "median", "q75")
        )
        axis.plot([low, high], [position] * 2, color=color, linewidth=1.2)
        axis.scatter(median, position, color=color, s=22, zorder=3)
    axis.set_xscale("log")
    axis.set_xlim(FLOOR, 1e-1)
    axis.set_yticks(range(len(order)), [REGRESSORS[name][0] for name in order])
    for tick, name in zip(axis.get_yticklabels(), order, strict=True):
        tick.set_color(REGRESSORS[name][1])
    axis.set_xlabel(r"unique $\Delta D^2$")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=cli.PoissonGLM().root)
    parser.add_argument("--units", type=cli._unit_set, default="all")
    args = parser.parse_args()
    fit_dir = cli.PoissonGLM(root=args.root).fit_dir(args.units)
    with np.load(fit_dir / "figure_data.npz", allow_pickle=False) as saved:
        data = {key: saved[key] for key in saved.files}
    final = load_unit_records(fit_dir / "checkpoints", "unit_*_final.json")
    unique = json.loads((fit_dir / "unique_deviance" / "summary.json").read_text())

    with plt.rc_context(FIGURE_STYLE):
        figure = plt.figure(figsize=(7.4, 6.6))
        top = figure.add_gridspec(
            1,
            2,
            left=0.16,
            right=0.98,
            top=0.95,
            bottom=0.5,
            wspace=0.42,
            width_ratios=(1, 1.15),
        )
        trial = top[0].subgridspec(3, 1, height_ratios=(1.3, 1, 0.6), hspace=0.15)
        trial_axes = [figure.add_subplot(cell) for cell in trial]
        kernel_axes = [
            figure.add_subplot(cell)
            for cell in top[1].subgridspec(2, 2, hspace=0.75, wspace=0.6)
        ]
        bottom = figure.add_gridspec(
            1,
            3,
            left=0.1,
            right=0.98,
            top=0.37,
            bottom=0.09,
            wspace=0.55,
            width_ratios=(2.0, 0.8, 1.0),
        )
        prediction, population, unique_axis = (
            figure.add_subplot(cell) for cell in bottom
        )
        box = unique_axis.get_position()  # room for the regressor names
        unique_axis.set_position([box.x0 + 0.09, box.y0, box.width - 0.09, box.height])

        _plot_trial(trial_axes, data)
        _plot_kernels(kernel_axes, data)
        _plot_prediction(prediction, data)
        _plot_population(population, final)
        _plot_unique(unique_axis, unique["groups"])
        for axis in (*trial_axes, *kernel_axes, population, unique_axis):
            separate_axes(axis)
        for axis in trial_axes[:2]:
            axis.spines["bottom"].set_visible(False)

        figure.canvas.draw()
        to_figure = figure.transFigure.inverted()
        firsts = (trial_axes[0], kernel_axes[0], prediction, population, unique_axis)
        for letter, axis in zip("abcde", firsts, strict=True):
            left = axis.yaxis.get_tightbbox().transformed(to_figure).x0
            top_edge = axis.get_position().y1 + 0.02
            figure.text(left, top_edge, letter, fontweight="bold", fontsize=10)
        print(save_figure(figure, fit_dir / "main_figure"))


if __name__ == "__main__":
    main()
