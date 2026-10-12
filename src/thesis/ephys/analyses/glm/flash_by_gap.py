"""Show how the flash response depends on the time from the flash before.

Question: how do the size and the latency of the response of a V1 unit to a
flash change with the gap from the flash before, and does the model predict
the same change?

Comparison: the observed firing rate and the firing rate that the model
predicts, around the same flashes, at different gaps. Both rates have the
response to the flash before in them, so they are measured in the same way.
The comparison over gaps is paired: each unit has a value at each gap.

- Flashes: every flash that is not the first of its trial and whose response
  window ends before the center exit (`LAGS_MS` after the flash). So all
  flashes are in the fixation, and the period of the trial is not a factor.
  The gaps are multiples of 40 ms. Each gap up to `LONG_GAP_MS` is a group
  (40, 80, 120, and 160 ms), and all longer gaps are one group.
- Observed rate: the spike count of the unit in the 1-ms bins of the model
  around each flash, as a rate.
- Predicted rate: the rate of the final refit of the unit in the same bins
  (all its terms, with the observed spike history). The final refit is fit on
  all trials, so this is not a held-out prediction.
- Panel a (observation and model): the rate of each gap group in its own row,
  from `LAGS_MS` before the flash, so that the response to the flash before
  is in the row. The rate of a unit is divided by its mean observed rate
  around these flashes; lines are the mean over units, smoothed (Gaussian,
  `SMOOTH_MS`). The solid vertical line is the flash and the dashed line is
  the flash before it (for the longest group, the nearest one). A time bin is
  drawn when `MIN_FLASHES` flashes or more have it inside the trial window.
- Response measures: the mean over the flashes of a group, smoothed. Peak: its
  maximum from `PEAK_MS`; the latency is the time of that maximum. Rate before
  the flash: its mean from `BASELINE_MS`. The two are given apart, and not as
  a ratio: the rate before a flash is not a neutral level, because it is a
  part of the response to the flash before (on its peak at a gap of 80 ms, in
  the dip after it at 160 ms). The maximum of a noisy trace is higher with
  fewer flashes, so every group uses the same number of flashes: `SUBSAMPLES`
  random subsets with the size of the smallest group, the same subsets for the
  observed and the predicted rate, and the mean over them.
- Panel b: the peak after the flash (solid) and the rate before the flash
  (dashed) against the gap, observed and predicted. Panel c: the latency of
  the peak. Median and interquartile range over units. Two-sided Wilcoxon
  signed-rank test of the longest against the shortest gap, for each measure.
- Units: the units of the fit (`--units`) with a response in the data: the
  peak of the response to all these flashes is `MIN_RESPONSE_RATIO` times the
  rate before the flash or more, and the mean rate is `MIN_RATE` spikes/s or
  more. The numbers are in the output.

Trials, alignment, and exclusions are those of `fit.py`. The sampling unit is
the unit. The script writes `flash_by_gap.svg` and `flash_by_gap.json` in the
session directory (`--root`).

    uv run python -m thesis.ephys.analyses.glm.flash_by_gap
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
from scipy.stats import wilcoxon

from thesis.ephys.analyses.glm import cli, dmat
from thesis.ephys.analyses.glm.fit import (
    analysis_unit_indices,
    build_unit_counts,
    load_unit_records,
)
from thesis.ephys.units import fetch_unit_table
from thesis.plotting import (
    CONCEPT_COLORS,
    FIGURE_STYLE,
    MODEL_COLOR,
    OBSERVED_COLOR,
    REFERENCE_COLOR,
    save_figure,
)

# The gaps are multiples of 40 ms. Each gap up to `LONG_GAP_MS` is a group, and
# all longer gaps are one group.
LONG_GAP_MS = 200
# The window around a flash: long before it, to show the flash before.
LAGS_MS = (-300, 150)
MIN_FLASHES = 20
SUBSAMPLES = 50
SEED = 0
BASELINE_MS = (-20, 20)
PEAK_MS = (20, 130)
SMOOTH_MS = 5
MIN_RESPONSE_RATIO = 1.2
MIN_RATE = 2.0
VISUAL = CONCEPT_COLORS["visual"]
NAMES = ("observed", "model")
COLORS = (OBSERVED_COLOR, MODEL_COLOR)


def fixation_flashes(trials) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the trial row, the time, and the gap (ms) of each flash that is used."""
    rows, times, gaps = [], [], []
    for row, (flashes, entry, leave, response) in enumerate(
        zip(
            trials["stim_pulse_times_s"],
            trials["center_entry_s"],
            trials["center_exit_s"],
            trials["response_port_entry_s"],
            strict=True,
        )
    ):
        flashes = np.asarray(flashes, dtype=float)
        flashes = flashes[(flashes >= entry) & (flashes <= response)]
        keep = flashes[1:] + LAGS_MS[1] / 1000 <= leave
        times.append(flashes[1:][keep])
        gaps.append(1000 * np.diff(flashes)[keep])
        rows.append(np.full(int(keep.sum()), row))
    return np.concatenate(rows), np.concatenate(times), np.concatenate(gaps)


def _measures(response: np.ndarray, lags: np.ndarray) -> np.ndarray:
    """Return the peak rate, the rate before the flash, and the latency (ms)."""
    smooth = gaussian_filter1d(response, SMOOTH_MS, axis=-1)
    searched = (lags >= PEAK_MS[0]) & (lags < PEAK_MS[1])
    before = (lags >= BASELINE_MS[0]) & (lags < BASELINE_MS[1])
    part = smooth[..., searched]
    return np.stack(
        (
            part.max(-1),
            smooth[..., before].mean(-1),
            lags[searched][part.argmax(-1)].astype(float),
        )
    )


def _summary(values: np.ndarray) -> dict:
    """Median and quartiles over units for each gap, and the longest against the shortest."""
    values = np.where(np.isfinite(values), values, np.nan)
    low, median, high = np.nanpercentile(values, [25, 50, 75], axis=0)
    change = values[:, -1] - values[:, 0]
    change = change[np.isfinite(change)]
    return {
        "median": median.tolist(),
        "quartiles": [low.tolist(), high.tolist()],
        "longest_minus_shortest_median": float(np.median(change)),
        "units_higher_at_longest": int(np.sum(change > 0)),
        "units": len(change),
        "wilcoxon_p": float(wilcoxon(change).pvalue),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=cli.PoissonGLM().root)
    parser.add_argument("--units", type=cli._unit_set, default="all")
    args = parser.parse_args()
    model = cli.PoissonGLM(root=args.root)
    prepared = dmat._load_windows(model.windows)
    trials = dmat._load_selected_trials(prepared)
    alignments, relative = prepared["alignments"], prepared["relative_centers"]
    grid = (len(alignments), len(relative))
    valid = dmat._valid_bin_mask(prepared)
    rows, times, gaps = fixation_flashes(trials)
    slot = 1000 * dmat.FLASH_SLOT_S
    gaps = slot * np.rint(gaps / slot)
    group = (np.minimum(gaps, LONG_GAP_MS) / slot).astype(int) - 1
    groups = int(group.max()) + 1
    lags = np.arange(*LAGS_MS)
    # The bins of the model grid around each flash. A bin before the window of
    # the trial is not used; the bins of the response measures are all inside.
    first = np.rint((times - alignments[rows] - relative[0]) / dmat.BINWIDTH_S)
    columns = first.astype(int)[:, None] + lags[None, :]
    inside = (columns >= 0) & (columns < grid[1])
    columns = np.clip(columns, 0, grid[1] - 1)
    measured_lags = lags >= min(BASELINE_MS[0], -2 * SMOOTH_MS * 3)
    if not inside[:, measured_lags].all():
        raise ValueError("A response window is outside the model grid.")
    # weights[g]: the mean over the flashes of group g, for each lag.
    members = inside[None] & (group[None, :, None] == np.arange(groups)[:, None, None])
    flashes_in_bin = members.sum(1)
    weights = members / np.maximum(flashes_in_bin, 1)[:, None, :]
    rng = np.random.default_rng(SEED)
    size = int(np.bincount(group, minlength=groups).min())
    subsets = [
        np.stack(
            [
                rng.choice(np.flatnonzero(group == g), size, replace=False)
                for _ in range(SUBSAMPLES)
            ]
        )
        for g in range(groups)
    ]

    units = fetch_unit_table(
        prepared["metadata"]["subject_name"],
        prepared["metadata"]["session_name"],
        unit_criteria_id=1,
        stability_param_id=0,
        include_metrics=False,
    )
    units = units.iloc[analysis_unit_indices(units, args.units)]
    metadata = json.loads(model.design.with_suffix(".json").read_text())
    common = np.load(model.design, mmap_mode="r", allow_pickle=False)
    final = {
        int(item["unit_id"]): item
        for item in load_unit_records(
            model.fit_dir(args.units) / "checkpoints", "unit_*_final.json"
        )
    }
    response_lags = lags[measured_lags]
    before = (response_lags >= BASELINE_MS[0]) & (response_lags < BASELINE_MS[1])
    searched = (response_lags >= PEAK_MS[0]) & (response_lags < PEAK_MS[1])
    out = {key: [] for key in ("rate", "measures")}
    for unit_id, spikes in zip(units["unit_id"], units["spike_times_s"], strict=True):
        spikes = np.asarray(spikes, dtype=float)
        counts = build_unit_counts(alignments, spikes)
        observed = counts.reshape(grid)[rows[:, None], columns] / dmat.BINWIDTH_S
        everything = gaussian_filter1d(observed[:, measured_lags].mean(0), SMOOTH_MS)
        if (
            everything.mean() < MIN_RATE
            or everything[searched].max()
            < MIN_RESPONSE_RATIO * everything[before].mean()
        ):
            continue
        # The rate of the final refit in every bin, with the observed history.
        record = final[int(unit_id)]
        width = int(metadata["base_columns"]) + int(record["components"])
        coefficients = np.asarray(record["coefficients"], dtype=float)
        history, _, _ = dmat.training_zscore(
            dmat.build_unit_history(alignments, spikes), valid
        )
        log_rate = (
            float(record["intercept"])
            + np.asarray(common[:, :width]) @ coefficients[:width]
            + history @ coefficients[width:]
        )
        predicted = (
            np.exp(log_rate).reshape(grid)[rows[:, None], columns] / dmat.BINWIDTH_S
        )
        pair = np.stack((observed, predicted))  # (observed or model, flash, lag)
        scale = observed[:, measured_lags].mean()  # the mean observed rate
        # (group, observed or model, lag), over the unit mean
        out["rate"].append(np.einsum("gfl,mfl->gml", weights, pair) / scale)
        response = pair[:, :, measured_lags]
        measured = np.stack(
            [
                _measures(response[:, subset].mean(2), response_lags).mean(-1)
                for subset in subsets
            ]
        )  # (group, measure, observed or model)
        measured[:, :2] /= scale
        out["measures"].append(measured)
    # rate: (observed or model, unit, group, lag)
    rate = np.moveaxis(np.asarray(out["rate"]), 2, 0)
    # (measure, observed or model, unit, group)
    peak, before_flash, latency = np.moveaxis(
        np.asarray(out["measures"]), (2, 3), (0, 1)
    )
    group_names = [
        f"{gaps[group == g].min():.0f}" + ("" if np.ptp(gaps[group == g]) == 0 else "+")
        for g in range(groups)
    ]
    stats = {
        "units": int(len(units)),
        "units_with_a_response": int(rate.shape[1]),
        "flashes": len(times),
        "gap_groups_ms": group_names,
        "flashes_in_each_gap_group": np.bincount(group, minlength=groups).tolist(),
        "flashes_in_each_subset": size,
        "mean_gap_of_each_group_ms": [
            float(gaps[group == g].mean()) for g in range(groups)
        ],
    }
    for index, name in enumerate(NAMES):
        stats[f"{name} peak rate (over the unit mean)"] = _summary(peak[index])
        stats[f"{name} rate before the flash (over the unit mean)"] = _summary(
            before_flash[index]
        )
        stats[f"{name} latency (ms)"] = _summary(latency[index])

    traces = gaussian_filter1d(rate.mean(1), SMOOTH_MS, axis=-1)  # (name, group, lag)
    with plt.rc_context(FIGURE_STYLE):
        figure = plt.figure(figsize=(7.2, 5.0))
        outer = figure.add_gridspec(1, 2, width_ratios=(1.25, 1), wspace=0.38)
        left = outer[0].subgridspec(groups, 1, hspace=0.18)
        right = outer[1].subgridspec(2, 1, hspace=0.5)
        row_axes = [figure.add_subplot(left[0])]
        row_axes += [
            figure.add_subplot(left[g], sharex=row_axes[0], sharey=row_axes[0])
            for g in range(1, groups)
        ]
        for g, axis in enumerate(row_axes):
            drawn = flashes_in_bin[g] >= MIN_FLASHES
            for index, color in enumerate(COLORS):
                axis.plot(lags[drawn], traces[index, g, drawn], color=color, lw=1.1)
            axis.axvline(0, color=VISUAL, linewidth=1.1)
            axis.axvline(
                -gaps[group == g].min(), color=VISUAL, linestyle="--", linewidth=0.9
            )
            axis.text(
                0.01,
                0.92,
                f"{group_names[g]} ms gap",
                transform=axis.transAxes,
                va="top",
                fontsize=6.5,
            )
            if g < groups - 1:
                axis.tick_params(labelbottom=False)
        for index, (name, color) in enumerate(zip(NAMES, COLORS, strict=True)):
            row_axes[0].text(
                0.84 + 0.16 * index,
                1.04,
                name,
                color=color,
                transform=row_axes[0].transAxes,
                va="bottom",
                ha="right",
            )
        row_axes[0].text(
            0,
            1.04,
            "flash (solid), the flash before (dashed)",
            color=VISUAL,
            transform=row_axes[0].transAxes,
            va="bottom",
            fontsize=6,
        )
        row_axes[-1].set_xlabel("time from flash (ms)")
        row_axes[groups // 2].set_ylabel("normalized firing rate")

        size_axis, latency_axis = (figure.add_subplot(cell) for cell in right)

        def draw(axis, values, color, style, face, shift):
            values = np.where(np.isfinite(values), values, np.nan)
            low, median, high = np.nanpercentile(values, [25, 50, 75], axis=0)
            axis.errorbar(
                np.arange(groups) + shift,
                median,
                yerr=(median - low, high - median),
                color=color,
                linestyle=style,
                marker="o",
                markerfacecolor=face or color,
                markersize=3,
                linewidth=1.1,
                elinewidth=0.5,
                capsize=1.5,
            )

        for index, color in enumerate(COLORS):
            shift = 0.08 * (2 * index - 1)  # observed and model side by side
            draw(size_axis, peak[index], color, "-", None, shift)
            draw(size_axis, before_flash[index], color, "--", "white", shift)
            draw(latency_axis, latency[index], color, "-", None, shift)
        size_axis.text(
            0,
            1.03,
            "peak after the flash (solid)\nrate before the flash (dashed)",
            transform=size_axis.transAxes,
            va="bottom",
            fontsize=6.5,
        )
        size_axis.axhline(1, color=REFERENCE_COLOR, linestyle=":", linewidth=0.7)
        size_axis.set_ylabel("normalized firing rate")
        latency_axis.set_ylabel("peak latency (ms)")
        for axis in (size_axis, latency_axis):
            axis.set_xticks(range(groups), group_names)
        latency_axis.set_xlabel("gap from the flash before (ms)")
        figure.canvas.draw()
        to_figure = figure.transFigure.inverted()
        for letter, axis in zip(
            "abc", (row_axes[0], size_axis, latency_axis), strict=True
        ):
            edge = axis.yaxis.get_tightbbox().transformed(to_figure).x0
            figure.text(
                edge - 0.035,
                axis.get_position().y1 + 0.03,
                letter,
                fontweight="bold",
                fontsize=10,
            )
        print(save_figure(figure, args.root / "flash_by_gap"))
    output = args.root / "flash_by_gap.json"
    output.write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
