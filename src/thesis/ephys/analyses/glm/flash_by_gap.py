"""Show how the flash response depends on the time from the flash before.

Question: how do the size and the latency of the response of a V1 unit to a
flash change with the gap from the flash before, and does the model predict
the same change?

Comparison: the observed firing rate and the firing rate that the model
predicts, around the same flashes, at different gaps. Both rates have the
response to the flash before in them, so they are measured in the same way.
The comparison over gaps is paired: each unit has a value at each gap.

- Flashes: every flash that is not the first of its trial and whose window
  ends before the center exit (`LAGS_MS` after the flash). So all flashes are
  in the fixation, and the period of the trial is not a factor. The gaps are
  multiples of 40 ms. Each gap up to `LONG_GAP_MS` is a group (40, 80, 120,
  and 160 ms), and all longer gaps are one group. The window starts
  `LAGS_MS` before the flash, so the response to the flash before is in it.
- Observed rate: the spike count of the unit in the 1-ms bins of the model
  around each flash, as a rate.
- Predicted rate: the rate of the final refit of the unit in the same bins
  (all its terms, with the observed spike history). The final refit is fit on
  all trials, so this is not a held-out prediction.
- Response: the mean over the flashes of a group, smoothed (Gaussian,
  `SMOOTH_MS`). Peak: its maximum from `PEAK_MS`; the latency is the time of
  that maximum. Rate before the flash: its mean from `BASELINE_MS`. The two
  are given apart, and not as a ratio: the rate before a flash is not a
  neutral level, because it is a part of the response to the flash before (on
  its peak at a gap of 80 ms, in the dip after it at 160 ms). The maximum of a
  noisy trace is higher with fewer flashes, so every group uses the same
  number of flashes: `SUBSAMPLES` random subsets with the size of the smallest
  group, the same subsets for the observed and the predicted rate, and the
  mean over them.
- Units: the units of the fit (`--units`) with a response in the data: the
  peak of the response to all these flashes is `MIN_RESPONSE_RATIO` times the
  baseline or more, and the mean rate is `MIN_RATE` spikes/s or more. The
  numbers are in the output.
- Panel a (observation) and panel b (the model): the response of each gap
  group, with the rate of a unit divided by its mean observed rate over the
  groups; mean over units.
- Panel c: the peak after the flash (solid) and the rate before the flash
  (dashed) against the gap, observed and predicted, with the rate of a unit
  divided by its mean observed rate. Panel d: the latency of the peak. Median
  and interquartile range over units. Two-sided Wilcoxon signed-rank test of
  the longest against the shortest gap, for each measure.

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
    label_panels,
    save_figure,
)

# The gaps are multiples of 40 ms. Each gap up to `LONG_GAP_MS` is a group, and
# all longer gaps are one group.
LONG_GAP_MS = 200
# The window around a flash: long before it, to show the flash before.
LAGS_MS = (-120, 150)
SUBSAMPLES = 50
SEED = 0
BASELINE_MS = (-20, 20)
PEAK_MS = (20, 130)
SMOOTH_MS = 5
MIN_RESPONSE_RATIO = 1.2
MIN_RATE = 2.0
VISUAL = CONCEPT_COLORS["visual"]


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
    # The gaps are multiples of the flash slot (40 ms), so a limit is a gap value:
    # a group is all the flashes with a gap up to its limit.
    slot = 1000 * dmat.FLASH_SLOT_S
    gaps = slot * np.rint(gaps / slot)
    group = (np.minimum(gaps, LONG_GAP_MS) / slot).astype(int) - 1
    groups = int(group.max()) + 1
    group_gap = np.asarray([gaps[group == g].mean() for g in range(groups)])
    lags = np.arange(*LAGS_MS)
    # The bins of the model grid around each flash.
    first = np.rint((times - alignments[rows] - relative[0]) / dmat.BINWIDTH_S)
    columns = first.astype(int)[:, None] + lags[None, :]
    inside = (columns.min(1) >= 0) & (columns.max(1) < grid[1])
    rows, times, gaps, group, columns = (
        values[inside] for values in (rows, times, gaps, group, columns)
    )
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
    before = (lags >= BASELINE_MS[0]) & (lags < BASELINE_MS[1])
    searched = (lags >= PEAK_MS[0]) & (lags < PEAK_MS[1])
    out = {key: [] for key in ("rate", "measures")}
    for unit_id, spikes in zip(units["unit_id"], units["spike_times_s"], strict=True):
        spikes = np.asarray(spikes, dtype=float)
        counts = build_unit_counts(alignments, spikes)
        observed = counts.reshape(grid)[rows[:, None], columns] / dmat.BINWIDTH_S
        everything = gaussian_filter1d(observed.mean(0), SMOOTH_MS)
        if (
            everything.mean() < MIN_RATE
            or everything[searched].max()
            < MIN_RESPONSE_RATIO * everything[before].mean()
        ):
            continue
        # The rate of the final refit in every bin, with the observed history.
        record = final[int(unit_id)]
        width = int(metadata["base_columns"]) + int(record["components"])
        weights = np.asarray(record["coefficients"], dtype=float)
        history, _, _ = dmat.training_zscore(
            dmat.build_unit_history(alignments, spikes), valid
        )
        log_rate = (
            float(record["intercept"])
            + np.asarray(common[:, :width]) @ weights[:width]
            + history @ weights[width:]
        )
        predicted = (
            np.exp(log_rate).reshape(grid)[rows[:, None], columns] / dmat.BINWIDTH_S
        )
        pair = np.stack((observed, predicted))  # (observed or model, flash, lag)
        out["rate"].append([pair[:, group == g].mean(1) for g in range(groups)])
        # For each group: (measure, observed or model), the mean over subsets.
        out["measures"].append(
            [_measures(pair[:, subset].mean(2), lags).mean(-1) for subset in subsets]
        )
    # rate: (observed or model, unit, group, lag)
    rate = np.moveaxis(np.asarray(out["rate"]), 2, 0)
    scale = rate[0].mean((1, 2), keepdims=True)  # the mean observed rate of a unit
    # measures: (measure, observed or model, unit, group); rates over the unit mean
    measures = np.moveaxis(np.asarray(out["measures"]), (2, 3), (0, 1))
    peak, before_flash, latency = measures
    peak, before_flash = peak / scale[:, :, 0], before_flash / scale[:, :, 0]
    names = ("observed", "model")
    stats = {
        "units": int(len(units)),
        "units_with_a_response": int(rate.shape[1]),
        "flashes": len(times),
        "flashes_in_each_gap_group": np.bincount(group, minlength=groups).tolist(),
        "flashes_in_each_subset": size,
        "mean_gap_of_each_group_ms": group_gap.tolist(),
    }
    for index, name in enumerate(names):
        stats[f"{name} peak rate (over the unit mean)"] = _summary(peak[index])
        stats[f"{name} rate before the flash (over the unit mean)"] = _summary(
            before_flash[index]
        )
        stats[f"{name} latency (ms)"] = _summary(latency[index])

    shades = np.linspace(0.3, 1.0, groups)
    group_names = [
        f"{gaps[group == g].min():.0f} ms"
        if np.ptp(gaps[group == g]) == 0
        else f"{gaps[group == g].min():.0f}+ ms"
        for g in range(groups)
    ]
    with plt.rc_context(FIGURE_STYLE):
        figure, axes = plt.subplots(1, 4, figsize=(7.4, 2.5))
        axes[1].sharey(axes[0])
        for axis, values, label in zip(
            axes[:2], rate, ("observed", "predicted by the model"), strict=True
        ):
            smooth = gaussian_filter1d(values / scale, SMOOTH_MS, axis=-1)
            for g, shade in enumerate(shades):
                axis.plot(lags, smooth[:, g].mean(0), color=VISUAL, alpha=shade)
            axis.axvline(0, color=REFERENCE_COLOR, linewidth=0.8)
            axis.set_xlabel("time from flash (ms)")
            axis.set_ylabel(f"normalized firing rate,\n{label}")
        for g in range(groups):
            axes[0].text(
                (0.0, 0.15, 0.3, 0.48, 0.67)[g] if groups == 5 else g / groups,
                1.02,
                group_names[g].replace(" ms", "")
                + (" ms gap" if g == groups - 1 else ""),
                color=VISUAL,
                alpha=shades[g],
                transform=axes[0].transAxes,
                va="bottom",
                fontsize=6,
            )

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

        # Panel c: the peak (solid) and the rate before the flash (dashed).
        for index, color in enumerate((OBSERVED_COLOR, MODEL_COLOR)):
            shift = 0.08 * (2 * index - 1)  # observed and model side by side
            draw(axes[2], peak[index], color, "-", None, shift)
            draw(axes[2], before_flash[index], color, "--", "white", shift)
            draw(axes[3], latency[index], color, "-", None, shift)
            axes[3].text(
                0.6 * index,
                1.02,
                names[index],
                color=color,
                transform=axes[3].transAxes,
                va="bottom",
            )
        axes[2].text(
            0,
            1.02,
            "peak after the flash (solid)\nrate before the flash (dashed)",
            transform=axes[2].transAxes,
            va="bottom",
            fontsize=6,
        )
        axes[2].set_ylabel("normalized firing rate")
        axes[3].set_ylabel("peak latency (ms)")
        for axis in axes[2:]:
            axis.set_xticks(
                range(groups), [name.replace(" ms", "") for name in group_names]
            )
            axis.set_xlabel("gap from the\nflash before (ms)")
            axis.tick_params(axis="x", labelsize=6.5)
        figure.tight_layout()
        label_panels(axes, x=-0.45)
        print(save_figure(figure, args.root / "flash_by_gap"))
    output = args.root / "flash_by_gap.json"
    output.write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
