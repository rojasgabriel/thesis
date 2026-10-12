"""Show how the flash response depends on the time from the flash before.

Question: how do the size and the latency of the response of a V1 unit to a
flash change with the gap from the flash before, and does the model have the
same dependence?

Comparison: the observed response and the flash kernel of the model, for
flashes in the fixation at different gaps. The comparison over gaps is paired:
each unit has a value at each gap.

- Flashes: every flash that is not the first of its trial and whose whole
  window (`WINDOW_MS`) ends before the center exit. So all flashes are in the
  fixation, and the period of the trial is not a factor. The flashes are put
  in the three gap groups of `GAP_GROUPS` (40 ms; 80 and 120 ms, called
  100 ms; 200 ms or more, called 250 ms).
- Observed response: the mean spike count around the flashes of a group, as a
  rate, smoothed (Gaussian, `SMOOTH_MS`). Its peak is the maximum from
  `PEAK_MS`; the latency is the time of that maximum. The maximum of a noisy
  trace is higher with fewer flashes, so the peak and the latency of every
  group come from the same number of flashes: `SUBSAMPLES` random subsets with
  the size of the smallest group, and the mean over the subsets.
- Model response: the flash kernel of the unit at the gap of the group
  (`results.flash_kernel_at`, at 40, 100, and 250 ms: the common flash kernel
  plus the kernels of fast adaptation and flash recovery at that gap, for a
  flash in the fixation),
  from the final refit. Its peak and latency are found in the same way.
- Size: the peak relative to the shortest gap, as a change in log (observed:
  the log of the peak rate over the peak rate at the shortest gap; model: the
  peak of the kernel minus the peak at the shortest gap). The observed rate
  has the response to the flash before in it at short gaps; the kernel does
  not.
- Units: the units of the fit (`--units`) with a response in the data: the
  peak of the response to all these flashes is `MIN_RESPONSE_RATIO` times the
  rate before the flash (`BASELINE_MS`) or more, and the mean rate is
  `MIN_RATE` spikes/s or more. The numbers are in the output.
- Panel a (observation): the observed response of each gap group, with the
  rate of a unit divided by its mean over the groups; mean over units.
- Panel b (the model): the median flash kernel of each gap group over units.
- Panel c (size) and panel d (latency): observed and model against the gap;
  median and interquartile range over units. Two-sided Wilcoxon signed-rank
  test of the longest against the shortest gap, for the data and the model.

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
from thesis.ephys.analyses.glm.fit import analysis_unit_indices, load_unit_records
from thesis.ephys.analyses.glm.matched_flashes import WINDOW_MS, flash_counts
from thesis.ephys.analyses.glm.results import (
    FLASH_ADAPT,
    FLASH_COMMON,
    flash_kernel_at,
    flash_kernel_levels,
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

# The gap of each group (the gap of the model kernel) and the gaps of the
# flashes in it. Gaps are multiples of 40 ms: the groups are 40 ms, 80 and
# 120 ms, and 200 ms or more. Flashes with a gap of 160 ms are in no group.
GAP_GROUPS = {40: (0, 60), 100: (60, 140), 250: (190, np.inf)}
SUBSAMPLES = 50
SEED = 0
BASELINE_MS = (-20, 20)
PEAK_MS = (20, 130)
SMOOTH_MS = 5
MIN_RESPONSE_RATIO = 1.2
MIN_RATE = 2.0
VISUAL = CONCEPT_COLORS["visual"]


def fixation_flashes(trials) -> tuple[np.ndarray, np.ndarray]:
    """Return the time and the gap (ms) of each flash that is used."""
    times, gaps = [], []
    for flashes, entry, leave, response in zip(
        trials["stim_pulse_times_s"],
        trials["center_entry_s"],
        trials["center_exit_s"],
        trials["response_port_entry_s"],
        strict=True,
    ):
        flashes = np.asarray(flashes, dtype=float)
        flashes = flashes[(flashes >= entry) & (flashes <= response)]
        keep = flashes[1:] + WINDOW_MS[1] / 1000 <= leave
        times.append(flashes[1:][keep])
        gaps.append(1000 * np.diff(flashes)[keep])
    return np.concatenate(times), np.concatenate(gaps)


def _peak(values: np.ndarray, lags: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the maximum from `PEAK_MS` and its time, along the last axis."""
    searched = (lags >= PEAK_MS[0]) & (lags < PEAK_MS[1])
    part = values[..., searched]
    return part.max(-1), lags[searched][part.argmax(-1)]


def _summary(values: np.ndarray) -> dict:
    """Median and quartiles over units for each gap, and the longest against the shortest."""
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
    times, gaps = fixation_flashes(trials)
    groups = len(GAP_GROUPS)
    group = np.full(len(gaps), -1)
    for index, (low, high) in enumerate(GAP_GROUPS.values()):
        group[(gaps >= low) & (gaps < high)] = index
    times, gaps, group = times[group >= 0], gaps[group >= 0], group[group >= 0]
    group_gap = np.asarray(list(GAP_GROUPS), dtype=float)
    lags = np.arange(*WINDOW_MS)

    units = fetch_unit_table(
        prepared["metadata"]["subject_name"],
        prepared["metadata"]["session_name"],
        unit_criteria_id=1,
        stability_param_id=0,
        include_metrics=False,
    )
    units = units.iloc[analysis_unit_indices(units, args.units)]
    # The peak of a noisy trace is higher with fewer flashes. So the peak and
    # the latency of every group come from the same number of flashes: random
    # subsets with the size of the smallest group, and the mean over subsets.
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
    observed, all_flashes, observed_peak, observed_latency = [], [], [], []
    for spikes in units["spike_times_s"]:
        counts = flash_counts(np.asarray(spikes, dtype=float), times) / dmat.BINWIDTH_S
        observed.append([counts[group == g].mean(0) for g in range(groups)])
        all_flashes.append(counts.mean(0))
        peaks = [
            _peak(gaussian_filter1d(counts[subset].mean(1), SMOOTH_MS, axis=-1), lags)
            for subset in subsets
        ]
        observed_peak.append([peak.mean() for peak, _ in peaks])
        observed_latency.append([latency.mean() for _, latency in peaks])
    observed = gaussian_filter1d(np.asarray(observed), SMOOTH_MS, axis=-1)
    all_flashes = gaussian_filter1d(np.asarray(all_flashes), SMOOTH_MS, axis=-1)
    observed_peak, observed_latency = (
        np.asarray(observed_peak),
        np.asarray(observed_latency),
    )
    before = (lags >= BASELINE_MS[0]) & (lags < BASELINE_MS[1])
    responsive = (
        _peak(all_flashes, lags)[0]
        >= MIN_RESPONSE_RATIO * all_flashes[:, before].mean(1)
    ) & (all_flashes.mean(1) >= MIN_RATE)

    # The flash kernel of each unit at the gap of each group, in the fixation.
    metadata = json.loads(model.design.with_suffix(".json").read_text())
    names = metadata["column_names"]
    scale = np.asarray(metadata["training_scale"])
    basis = dmat.task_temporal_bases()[FLASH_COMMON]
    adapt_levels = next(
        item["adapt_levels_centered"]
        for item in metadata["task_manifest"]
        if item["name"] == FLASH_ADAPT
    )
    levels = flash_kernel_levels(trials)
    flash_terms = [
        item["name"]
        for item in metadata["task_manifest"]
        if item["name"].startswith(("visual_flash", "flash_"))
    ]
    final = {
        int(item["unit_id"]): np.asarray(item["coefficients"])
        for item in load_unit_records(
            model.fit_dir(args.units) / "checkpoints", "unit_*_final.json"
        )
    }
    kernel_lags = 1000 * basis.basis_time
    modeled = []
    for unit_id in units["unit_id"].to_numpy(dtype=int):
        kernels = {}
        for name in flash_terms:
            columns = [
                i
                for i, column in enumerate(names)
                if column.startswith(f"{name}_basis")
            ]
            kernels[name] = (
                kernel_lags,
                (final[unit_id][columns] / scale[columns]) @ basis.basis.T,
            )
        modeled.append(
            [
                flash_kernel_at(kernels, adapt_levels, levels, gap, False)
                for gap in group_gap
            ]
        )
    modeled = np.asarray(modeled)

    observed, modeled = observed[responsive], modeled[responsive]
    observed_peak = observed_peak[responsive]
    observed_latency = observed_latency[responsive]
    model_peak, model_latency = _peak(modeled, kernel_lags)
    with np.errstate(divide="ignore", invalid="ignore"):
        observed_size = np.log(observed_peak / observed_peak[:, :1])
    observed_size[~np.isfinite(observed_size)] = np.nan
    model_size = model_peak - model_peak[:, :1]
    stats = {
        "units": int(len(units)),
        "units_with_a_response": int(responsive.sum()),
        "flashes": len(times),
        "flashes_in_each_gap_group": np.bincount(group, minlength=groups).tolist(),
        "flashes_in_each_subset": size,
        "gap_of_each_group_ms": group_gap.tolist(),
        "median_gap_of_the_flashes_in_each_group_ms": [
            float(np.median(gaps[group == g])) for g in range(groups)
        ],
        "flash_terms_of_the_model": flash_terms,
        "observed size (log of the peak over the peak at the shortest gap)": _summary(
            observed_size
        ),
        "model size (peak of the kernel minus the peak at the shortest gap)": _summary(
            model_size
        ),
        "observed latency (ms)": _summary(observed_latency.astype(float)),
        "model latency (ms)": _summary(model_latency.astype(float)),
    }

    shades = np.linspace(0.3, 1.0, groups)
    with plt.rc_context(FIGURE_STYLE):
        figure, axes = plt.subplots(1, 4, figsize=(7.4, 2.3))
        data_axis, model_axis, size_axis, latency_axis = axes
        normalized = observed / observed.mean((1, 2), keepdims=True)
        for g, shade in enumerate(shades):
            data_axis.plot(lags, normalized[:, g].mean(0), color=VISUAL, alpha=shade)
            model_axis.plot(
                kernel_lags, np.median(modeled[:, g], 0), color=VISUAL, alpha=shade
            )
        for g in range(groups):
            data_axis.text(
                0.03 + 0.33 * g,
                1.02,
                f"{group_gap[g]:.0f} ms",
                color=VISUAL,
                alpha=shades[g],
                transform=data_axis.transAxes,
                va="bottom",
                fontsize=6.5,
            )
        data_axis.axvline(0, color=REFERENCE_COLOR, linewidth=0.8)
        data_axis.set_xlabel("time from flash (ms)")
        data_axis.set_ylabel("normalized firing rate")
        model_axis.axhline(0, color=REFERENCE_COLOR, linestyle="--", linewidth=0.8)
        model_axis.set_xlabel("time from flash (ms)")
        model_axis.set_ylabel(r"flash kernel ($\Delta$ log rate)")
        for axis, pair, label in (
            (
                size_axis,
                (observed_size, model_size),
                r"peak, $\Delta$ log from" + "\nthe shortest gap",
            ),
            (latency_axis, (observed_latency, model_latency), "peak latency (ms)"),
        ):
            for values, color, name, offset in zip(
                pair,
                (OBSERVED_COLOR, MODEL_COLOR),
                ("observed", "model"),
                (0.97, 1.03),
                strict=True,
            ):
                low, median, high = np.nanpercentile(values, [25, 50, 75], axis=0)
                axis.errorbar(
                    group_gap * offset,
                    median,
                    yerr=(median - low, high - median),
                    color=color,
                    marker="o",
                    markersize=3,
                    linewidth=1.1,
                    elinewidth=0.6,
                    capsize=1.5,
                )
                if axis is size_axis:
                    axis.text(
                        0.03,
                        0.98 - 0.11 * (name == "model"),
                        name,
                        color=color,
                        transform=axis.transAxes,
                        va="top",
                    )
            axis.set_xscale("log")
            axis.set_xticks(group_gap, [f"{gap:.0f}" for gap in group_gap])
            axis.minorticks_off()
            axis.set_xlabel("gap from the flash\nbefore (ms)")
            axis.set_ylabel(label)
        size_axis.axhline(0, color=REFERENCE_COLOR, linestyle="--", linewidth=0.8)
        figure.tight_layout()
        label_panels(axes, x=-0.42)
        print(save_figure(figure, args.root / "flash_by_gap"))
    output = args.root / "flash_by_gap.json"
    output.write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
