"""Compare observed and fitted flash peaks across the center-port exit.

Question: is the apparent sensory enhancement independent of locomotion?
The measured comparison is stationary versus movement task periods; video
covariates do not isolate locomotion itself. No new model fits are made.

Observation: one pair of 15-ms flashes per trial and preceding-gap class,
<=50 ms (nominal 40 ms) or >50 ms (nominal 80+ ms), exactly the model split:
the last in-port flash of that class and the first after-exit flash of that
class at least `MIN_PAIR_SEPARATION_S` after it, so the two 30..120-ms peak
windows do not overlap. (Before 2026-10-25 the pair was the last in-port and
the first after-exit flash; in the 40-ms class these were always consecutive
flashes 40 ms apart, with overlapping windows and often equal peaks.) The gap
is to the immediately preceding measured pulse of any width; the first pulse
of a trial has none and is not used. A trial can give a pair to each class.
Report all exclusion counts and selected trial IDs. Align to measured flash onset, use
10-ms bins, and subtract the stationary -40..0-ms baseline within each gap
class from both peaks (30..120 ms). Restrict all sources to the GLM's selected
completed trials and full paired windows inside its valid native 1-ms grid.
Round flash anchors to the nearest model bin. Other flashes can enter the window.
PSTHs show each gap class on its own, above the scatter of that class: the
population mean and SEM across unit trial means. Saved per-unit peak uncertainty is SD across paired-trial
bootstrap draws, stratified by gap class and conditional on saved models.

Primary model comparison: observed counts, saved whole-trial held-out rate
predictions, and those same predictions with only the raw flash-by-exit
contribution removed. Reconstruct each fold with history scaled on its own
fitting rows; keep observed history and all other contributions fixed. Apply
the identical stationary-baseline peak calculation in spikes/s to all three
sources. The same paired trial draws give saved SD on both axes for every source.
This removes a fitted contribution without refitting or simulating history;
it is not a test of unique necessity, a causal effect, or locomotion independence.

Supporting model view: reconstruct raw flash kernels from saved final coefficients and
column scales. Compare the peak of each 10-ms-averaged kernel at 30..120 ms
in port versus after exit, separately at 40-ms and 80+-ms preceding gaps.
Only flash, short-gap and flash-by-exit terms enter; nuisance terms cancel
when held at the same reference. Peaks are changes in log rate, not spikes/s.
Saved per-unit fold SD describes spread across overlapping training sets;
it is not an independent confidence interval. Displayed group bars are SEM
across units. These peaks are not leak
corrected and differ from the earlier matched-rate peak contribution.

Selection: both sessions in flash_kernels.SESSIONS, quality 1/stability 0,
and available main-GLM fits after the existing fit exclusions. Scatters show
only the mean of all eligible units (open markers) and the mean of the saved
StimulusResponse excited group (filled markers), with SEM across units on
both axes. The mouse is the marker shape (`mouse_marker`); the color is the
source (observed black, model wine). PSTHs and the existing report
summaries use only that excited group. The same units enter every rate column. The
sampling unit is the neural unit; tests are paired within mouse. Trial
selection differs from standardized model-kernel conditions. All finite
responsive-group values enter summary statistics. All-unit values are saved.
Rate scatters use linear axes and show peaks divided by each unit's mean
observed rate over the full valid model interval of the selected paired trials,
pooling counts and exposure across gap classes. The same positive denominator
scales every source and condition. Each bootstrap draw recomputes that pooled
observed denominator using the shared paired draws within each gap class.
All units enter their group means; complete mean SEM bars set display limits.
Absolute peaks and statistics remain in the report. Supporting kernel
scatters stay linear because their values are already changes in log rate.

Writes flash_observation.svg, its two gap classes as slides of one size and the
same limits (flash_observation_80ms.svg, flash_observation_40ms.svg),
flash_comparison.svg, flash_model.svg (supporting), PNG previews, and
flash_movement.json under results/glm. Older working figures remain intact.

    THESIS_CACHE_DIR=figures/cache nice -n 10 uv run python -m \
        thesis.ephys.analyses.glm.flash_movement
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy import stats

from thesis.plotting import (
    FIGURE_STYLE,
    MODEL_COLOR,
    OBSERVED_COLOR,
    label_panels,
    mouse_marker,
    save_figure,
)
from thesis.plotting import PERIOD_COLORS as PERIOD_COLOR_OF

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from thesis.ephys.analyses.glm import dmat
from thesis.ephys.analyses.glm.fit import (
    build_unit_counts,
    cross_validation_partitions,
    load_unit_records,
)
from thesis.ephys.analyses.glm.flash_kernels import SESSIONS
from thesis.ephys.analyses.glm.peaks import LAGS_MS
from thesis.ephys.analyses.glm.raw_video_kernels import task_kernel
from thesis.ephys.analyses.glm.results import flash_kernel_at, raw_contribution
from thesis.ephys.units import fetch_unit_table

OUTPUT = Path("results/glm")
GAPS_MS = (100, 40)
GAP_LABELS = ("preceding gap ≥80 ms", "preceding gap 40 ms")
PERIOD_COLORS = (PERIOD_COLOR_OF["in port"], PERIOD_COLOR_OF["after exit"])
MIN_PAIR_SEPARATION_S = 0.09  # the 30..120-ms peak windows do not overlap
RATE_SOURCES = ("observed", "full model", "without flash × exit")


def paired_flashes_by_gap(trials):
    """Pair, in each gap class, the last in-port and a later after-exit flash."""
    groups = [
        {"stationary": [], "movement": [], "trial_ids": [], "gaps_ms": []}
        for _ in GAPS_MS
    ]
    counts = {
        "trials_with_flashes_in_both_periods": 0,
        "no_pair_in_class": dict.fromkeys(GAP_LABELS, 0),
    }
    for _, trial in trials.iterrows():
        entry, leave, response = (
            trial["center_entry_s"],
            trial["center_exit_s"],
            trial["response_port_entry_s"],
        )
        if not np.isfinite([entry, leave, response]).all():
            continue
        times = np.asarray(trial["stim_pulse_times_s"], dtype=float)
        widths = np.asarray(trial["stim_pulse_widths_ms"], dtype=float)
        if len(times) != len(widths) or not np.isfinite(times).all():
            raise ValueError(
                "Pulse times and widths must match and times must be finite."
            )
        if np.any(np.diff(times) <= 0):
            raise ValueError("Measured pulse times must be strictly increasing.")
        flash = widths == 15
        stationary = flash & (times >= entry) & (times < leave)
        movement = flash & (times >= leave) & (times <= response)
        if not stationary.any() or not movement.any():
            continue
        counts["trials_with_flashes_in_both_periods"] += 1
        gaps = np.diff(times, prepend=np.nan)  # the first pulse has no gap
        short = gaps <= dmat.FLASH_JUST_BEFORE_S
        for index, label in enumerate(GAP_LABELS):
            in_class = np.isfinite(gaps) & (short == (GAPS_MS[index] <= 50))
            before = np.flatnonzero(stationary & in_class)
            after = (
                np.flatnonzero(
                    movement
                    & in_class
                    & (times >= times[before[-1]] + MIN_PAIR_SEPARATION_S)
                )
                if len(before)
                else []
            )
            if not len(after):
                counts["no_pair_in_class"][label] += 1
                continue
            pair = np.array([before[-1], after[0]])
            group = groups[index]
            group["stationary"].append(float(times[pair[0]]))
            group["movement"].append(float(times[pair[1]]))
            group["trial_ids"].append(int(trial["trial_num"]))
            group["gaps_ms"].append((1000 * gaps[pair]).tolist())
    return groups, counts


def observed_peaks(rate, centers):
    """Peak of (..., condition, time), using one stationary baseline."""
    rate = np.asarray(rate)
    baseline = rate[..., 0, (centers >= -0.04) & (centers < 0)].mean(-1)
    response = rate[..., (centers >= 0.03) & (centers < 0.12)].max(-1)
    return response - baseline[..., None]


def kernel_peaks(coefficients, metadata, basis, lags):
    """Return (gap, period) peaks with other raw contributions held fixed."""
    names = (
        dmat.FLASH_COMMON,
        dmat.FLASH_JUST_BEFORE,
        dmat.FLASH_AFTER_EXIT,
        dmat.FLASH_RECOVERY,
    )
    present = {c.split("_basis_")[0] for c in metadata["column_names"]}
    terms = {
        name: (lags, task_kernel(coefficients, metadata, name, basis))
        for name in names
        if name in present
    }
    center = metadata.get("flash_recovery_center")
    # The basis includes its 150-ms endpoint; 10-ms bins are half-open.
    inside = (np.round(lags, 6) >= 0) & (np.round(lags, 6) < 150)
    centers = lags[inside].reshape(-1, 10).mean(-1) + 0.5
    window = (centers >= 30) & (centers < 120)
    return np.asarray(
        [
            [
                flash_kernel_at(terms, gap, after, center)[inside]
                .reshape(-1, 10)
                .mean(-1)[window]
                .max()
                for after in (False, True)
            ]
            for gap in GAPS_MS
        ]
    )


def valid_pair_windows(groups, prepared):
    """Keep entire paired -40..120-ms windows on the model's valid grid."""
    lookup = {
        int(number): row
        for row, number in enumerate(prepared["selected_trial_numbers"])
    }
    n_bins = len(prepared["relative_centers"])
    valid = dmat._valid_bin_mask(prepared).reshape(-1, n_bins)
    retained, windows = [], []
    excluded = 0
    for group in groups:
        rows = np.asarray([lookup[number] for number in group["trial_ids"]], dtype=int)
        times = np.asarray([group["stationary"], group["movement"]])
        anchors = np.rint(
            (times - prepared["alignments"][rows] - prepared["relative_centers"][0])
            / dmat.BINWIDTH_S
        ).astype(int)
        columns = anchors[:, :, None] + LAGS_MS
        bounded = np.clip(columns, 0, n_bins - 1)
        keep = ((columns >= 0) & (columns < n_bins)).all(axis=(0, 2)) & valid[
            rows[None, :, None], bounded
        ].all(axis=(0, 2))
        excluded += int((~keep).sum())
        retained.append(
            {
                key: [value[i] for i in np.flatnonzero(keep)]
                for key, value in group.items()
            }
        )
        windows.append((rows[None, :, None] * n_bins + columns)[:, keep])
    return retained, windows, excluded


def held_out_rates(
    counts, raw_history, common, metadata, record, partitions, requested
):
    """Reconstruct saved fold predictions and remove only the raw interaction."""
    width = int(metadata["base_columns"]) + int(record["components"])
    removed = [
        i
        for i, name in enumerate(metadata["column_names"])
        if name.startswith(f"{dmat.FLASH_AFTER_EXIT}_basis")
    ]
    if not removed:
        raise ValueError("The saved design has no flash × exit block.")
    predicted = np.full((2, len(counts)), np.nan)
    folds = iter(record["folds"])
    for part in partitions:
        spikes = counts[part["test"]].sum()
        if spikes <= 0:
            continue
        fold = next(folds, None)
        if fold is None or not np.isclose(spikes, fold["test"]["observed_spikes"]):
            raise ValueError("Saved fold and observed test spikes differ.")
        weights = np.asarray(fold["coefficients"], dtype=float)
        if len(weights) != width + dmat.HISTORY_COLUMNS:
            raise ValueError("Saved fold and design dimensions differ.")
        rows = part["test"] & requested
        if not rows.any():
            continue
        history, _, _ = dmat.training_zscore(raw_history, part["fit"])
        design = np.asarray(common[rows, :width])
        log_rate = (
            float(fold["intercept"])
            + design @ weights[:width]
            + history[rows] @ weights[width:]
        )
        contribution = raw_contribution(design, weights, metadata, removed)
        predicted[0, rows] = np.exp(log_rate)
        predicted[1, rows] = np.exp(log_rate - contribution)
    if next(folds, None) is not None:
        raise ValueError("Too many saved folds for the reconstructed trial split.")
    if not np.isfinite(predicted[:, requested]).all():
        raise ValueError("A selected flash window has no finite held-out prediction.")
    return predicted


def matched_rate_peaks(rates):
    """Apply the same 10-ms-bin peak measure to every source and bootstrap."""
    rates = np.asarray(rates)
    binned = rates.reshape(*rates.shape[:-1], -1, 10).mean(-1)
    centers = (LAGS_MS.reshape(-1, 10).mean(-1) + 0.5) / 1000
    return observed_peaks(binned, centers)


def normalize_peaks(peaks, peak_draws, scale, scale_draws):
    """Share the observed scale across sources, periods and gaps in each draw."""
    scale_draws = np.asarray(scale_draws)
    if (
        not np.isfinite(scale)
        or scale <= 0
        or not np.all(np.isfinite(scale_draws) & (scale_draws > 0))
    ):
        raise ValueError("Normalization requires positive observed mean rates.")
    normalized = np.asarray(peaks) / scale
    normalized_draws = np.asarray(peak_draws) / scale_draws[None, :, None, None]
    return normalized, normalized_draws.std(axis=1, ddof=1)


def trial_rate_scale(counts, exposure, group_rows, draws):
    """Pool full-trial spike counts and valid seconds, including in each draw."""
    selected = np.concatenate(group_rows)
    scale = counts[selected].sum() / exposure[selected].sum()
    sampled_counts = sum(
        len(rows) * (weights @ counts[rows])
        for rows, weights in zip(group_rows, draws, strict=True)
    )
    sampled_exposure = sum(
        len(rows) * (weights @ exposure[rows])
        for rows, weights in zip(group_rows, draws, strict=True)
    )
    return scale, sampled_counts / sampled_exposure


def session_data(session, excited):
    root = OUTPUT / session
    metadata = json.loads((root / "common_design_me.json").read_text())
    checkpoints = root / "all_fit_me/checkpoints"
    final = {
        int(item["unit_id"]): item
        for item in load_unit_records(checkpoints, "unit_*_final.json")
    }
    records = {
        int(item["unit_id"]): item
        for item in load_unit_records(checkpoints, "unit_*_folds.json")
    }
    folds = {unit_id: item["folds"] for unit_id, item in records.items()}
    ids = sorted(final)
    responsive = np.isin(ids, list(excited))
    if not responsive.any():
        raise ValueError(f"No paired excited units for {session}.")
    subject, stamp = session.split("_", 1)
    units = fetch_unit_table(subject, stamp, 1, 0, include_metrics=False).set_index(
        "unit_id"
    )
    prepared = dmat._load_windows(root / "stimulus_windows_me.npz")
    trials = dmat._load_selected_trials(prepared)
    groups, selection = paired_flashes_by_gap(trials)
    groups, windows, excluded = valid_pair_windows(groups, prepared)
    selection["incomplete_model_window_pairs"] = excluded
    # The gaps of the two flashes of each pair are not matched; the report
    # gives them and their recovery (r = 1 - exp(-gap / tau)) in each period.
    selection["pair_gaps"] = {}
    for group, label in zip(groups, GAP_LABELS, strict=True):
        gaps = np.asarray(group["gaps_ms"]) / 1000  # (pair, period)
        recovery = 1 - np.exp(-gaps / dmat.FLASH_RECOVERY_TAU_S)
        selection["pair_gaps"][label] = {
            "median_gap_ms_in_port_after_exit": (1000 * np.median(gaps, 0)).tolist(),
            "mean_recovery_in_port_after_exit": recovery.mean(0).tolist(),
        }
    for group, label in zip(groups, GAP_LABELS, strict=True):
        if len(group["trial_ids"]) < 2:
            raise ValueError(f"Too few complete paired windows for {session}, {label}.")
    common = np.load(root / "common_design_me.npy", mmap_mode="r", allow_pickle=False)
    valid = dmat._valid_bin_mask(prepared)
    partitions = cross_validation_partitions(
        len(trials), len(prepared["relative_centers"]), valid
    )
    requested = np.zeros(len(valid), dtype=bool)
    requested[np.concatenate([window.ravel() for window in windows])] = True
    # Reuse the same paired trial draws across units, periods and rate sources.
    rng = np.random.default_rng(20261020)
    draws = [
        rng.multinomial(
            len(g["trial_ids"]),
            np.full(len(g["trial_ids"]), 1 / len(g["trial_ids"])),
            size=500,
        )
        / len(g["trial_ids"])
        for g in groups
    ]
    trial_lookup = {
        int(number): row
        for row, number in enumerate(prepared["selected_trial_numbers"])
    }
    group_rows = [
        np.asarray([trial_lookup[number] for number in group["trial_ids"]])
        for group in groups
    ]
    exposure = valid.reshape(len(trials), -1).sum(1) * dmat.BINWIDTH_S
    rate_values, rate_spread, traces = [], [], []
    normalized_values, normalized_spread, normalization_rates = [], [], []
    contribution_draw_sd = []
    for position, unit_id in enumerate(ids):
        spikes = units.loc[unit_id, "spike_times_s"]
        counts = build_unit_counts(prepared["alignments"], spikes)
        history = dmat.build_unit_history(prepared["alignments"], spikes)
        predicted = held_out_rates(
            counts, history, common, metadata, records[unit_id], partitions, requested
        )
        sources = np.vstack([counts, predicted]) / dmat.BINWIDTH_S
        peaks, spreads, unit_traces, contribution_sd = [], [], [], []
        peak_draws = []
        for window, weights in zip(windows, draws, strict=True):
            # (source, period, trial, native 1-ms bin), identical across sources.
            rates = sources[:, window]
            peaks.append(matched_rate_peaks(rates.mean(2)))
            # Average 1-ms bins before bootstrapping to keep the calculation small.
            binned = rates.reshape(*rates.shape[:-1], -1, 10).mean(-1)
            repeated = np.einsum("bt,sptj->bspj", weights, binned)
            centers = (LAGS_MS.reshape(-1, 10).mean(-1) + 0.5) / 1000
            boot_peaks = observed_peaks(repeated, centers)
            spreads.append(boot_peaks.std(axis=0, ddof=1))
            peak_draws.append(boot_peaks)
            enhancement = boot_peaks[..., 1] - boot_peaks[..., 0]
            contribution_sd.append((enhancement[:, 1] - enhancement[:, 2]).std(ddof=1))
            unit_traces.append(binned[0])
        scale, scale_draws = trial_rate_scale(
            (counts * valid).reshape(len(trials), -1).sum(1),
            exposure,
            group_rows,
            draws,
        )
        normalized, normalized_sd = normalize_peaks(
            peaks, peak_draws, scale, scale_draws
        )
        normalized_values.append(normalized)
        normalized_spread.append(normalized_sd)
        normalization_rates.append(scale)
        rate_values.append(peaks)
        rate_spread.append(spreads)
        contribution_draw_sd.append(contribution_sd)
        traces.append(np.stack([trace.mean(1) for trace in unit_traces]))
        print(
            f"{session}: reconstructed {position + 1}/{len(ids)} held-out units",
            flush=True,
        )
    traces = np.asarray(traces)  # (unit, gap, period, bin), observed spikes/s
    rate_values = np.asarray(rate_values)  # (unit, gap, source, period)
    rate_spread = np.asarray(rate_spread)
    normalized_values = np.asarray(normalized_values)
    normalized_spread = np.asarray(normalized_spread)
    basis = dmat.task_temporal_bases()[dmat.FLASH_COMMON]
    lags = 1000 * basis.basis_time
    model, model_sd = [], []
    for unit_id in ids:
        model.append(
            kernel_peaks(final[unit_id]["coefficients"], metadata, basis.basis, lags)
        )
        repeated = [
            kernel_peaks(fold["coefficients"], metadata, basis.basis, lags)
            for fold in folds[unit_id]
        ]
        if len(repeated) < 2:
            raise ValueError(f"Too few folds for unit {unit_id} in {session}.")
        model_sd.append(np.std(repeated, axis=0, ddof=1))
    result = {
        "subject": subject,
        "unit_ids": ids,
        "responsive": responsive,
        "trials": [len(group["trial_ids"]) for group in groups],
        "trial_groups": dict(zip(GAP_LABELS, groups, strict=True)),
        "selection": selection,
        "centers": centers,
        "traces": traces,
        "observed": normalized_values[:, :, 0],
        "observed_sd": normalized_spread[:, :, 0],
        "full_model": normalized_values[:, :, 1],
        "full_model_sd": normalized_spread[:, :, 1],
        "without_flash_exit": normalized_values[:, :, 2],
        "without_flash_exit_sd": normalized_spread[:, :, 2],
        "normalized_rate_peaks": normalized_values,
        "normalized_rate_bootstrap_sd": normalized_spread,
        "normalization_rate": np.asarray(normalization_rates),
        "rate_peaks": rate_values,
        "rate_bootstrap_sd": rate_spread,
        "contribution_bootstrap_sd": np.asarray(contribution_draw_sd),
        "model": np.asarray(model),
        "model_sd": np.asarray(model_sd),
    }
    for name in (
        "traces",
        "rate_peaks",
        "rate_bootstrap_sd",
        "model",
        "model_sd",
        "normalized_rate_peaks",
        "normalized_rate_bootstrap_sd",
        "normalization_rate",
    ):
        if not np.isfinite(result[name]).all():
            raise ValueError(f"Non-finite {name} in {session}.")
    return result


def comparison_summary(values):
    delta = values[:, 1] - values[:, 0]
    return {
        "units": len(delta),
        "mean_stationary": float(values[:, 0].mean()),
        "mean_movement": float(values[:, 1].mean()),
        "mean_difference": float(delta.mean()),
        "median_difference": float(np.median(delta)),
        "units_higher_after_exit": int((delta > 0).sum()),
        "paired_wilcoxon_p_unadjusted": float(stats.wilcoxon(delta).pvalue)
        if np.any(delta != 0)
        else 1.0,
    }


def peak_scatter(axis, sessions, field, gap=None):
    extents = []
    color = OBSERVED_COLOR if field == "observed" else MODEL_COLOR
    for index, data in enumerate(sessions):
        marker = mouse_marker(data["subject"], index)
        values = data[field]
        if gap is not None:
            values = values[:, gap]
        responsive = np.asarray(data["responsive"], dtype=bool)
        for selected, filled in ((values, False), (values[responsive], True)):
            mean = selected.mean(0)
            spread = stats.sem(selected, axis=0)
            axis.errorbar(
                *mean,
                xerr=spread[0],
                yerr=spread[1],
                fmt=marker,
                markerfacecolor=color if filled else "none",
                color=color,
                ms=7,
                capsize=2,
                alpha=0.8,
                zorder=3 + int(filled),
            )
            extents.extend((mean, mean - spread, mean + spread))
        # Fixed corner labels keep mouse names clear of the point cloud.
        axis.plot(
            0.66,
            0.16 - index * 0.08,
            marker,
            color=OBSERVED_COLOR,
            mfc="none",
            transform=axis.transAxes,
            clip_on=False,
            ms=4,
        )
        axis.text(
            0.71,
            0.16 - index * 0.08,
            data["subject"],
            transform=axis.transAxes,
            va="center",
            color=OBSERVED_COLOR,
        )
    axis.text(
        0.66,
        0.25,
        "open: all units\nfilled: flash-excited",
        transform=axis.transAxes,
        va="bottom",
        fontsize=5.5,
        color="0.35",
    )
    low = min(0, min(values.min() for values in extents))
    high = max(0, max(values.max() for values in extents))
    padding = max((high - low) * 0.06, 0.01)
    limits = (low - padding, high + padding)
    axis.set(xlim=limits, ylim=limits, aspect="equal")


def match_scatter_limits(axes):
    """Use common limits across gap/source panels without changing statistics."""
    limits = (
        min(axis.get_xlim()[0] for axis in axes),
        max(axis.get_xlim()[1] for axis in axes),
    )
    for axis in axes:
        axis.set(xlim=limits, ylim=limits)
        axis.plot(limits, limits, color="0.65", linestyle="--", linewidth=0.7, zorder=0)


def psth_panel(axis, data, gap, labels):
    """Mean rate of the group units around the flashes of one gap class."""
    unit_means = data["traces"][data["responsive"], gap]
    for period, (name, color) in enumerate(
        zip(("stationary", "movement"), PERIOD_COLORS, strict=True)
    ):
        mean = unit_means[:, period].mean(0)
        spread = stats.sem(unit_means[:, period], axis=0)
        axis.plot(1000 * data["centers"], mean, color=color)
        axis.fill_between(
            1000 * data["centers"],
            mean - spread,
            mean + spread,
            color=color,
            alpha=0.15,
            linewidth=0,
        )
        if labels:
            axis.text(
                0.04,
                0.95 - 0.12 * period,
                name,
                color=color,
                transform=axis.transAxes,
                va="top",
            )
    axis.axvline(0, color="0.7", linestyle="--", linewidth=0.7)
    if GAPS_MS[gap] == 40:  # the flash before is inside the window
        axis.axvline(-40, color="0.7", linestyle="--", linewidth=0.7)
    axis.set_title(
        f"{data['subject']}\n{int(data['responsive'].sum())} units, "
        f"{data['trials'][gap]} pairs"
    )
    axis.set(
        xlim=(LAGS_MS[0] - 5, LAGS_MS[-1] + 1),
        xlabel="time from flash (ms)",
        ylabel="firing rate (spikes/s)",
    )


def write_figure(figure, name):
    print(save_figure(figure, OUTPUT / name))


def plot(sessions):
    with plt.rc_context(FIGURE_STYLE):
        # One column per gap class: the PSTH of each mouse above the scatter.
        figure = plt.figure(figsize=(7.6, 6.4), layout="constrained")
        grid = figure.add_gridspec(2, 4, height_ratios=(1.0, 1.9))
        panels, psths, scatters = [], [], []
        for gap, label in enumerate(GAP_LABELS):
            for mouse, data in enumerate(sessions):
                # One rate scale for both mice and both gaps.
                axis = figure.add_subplot(
                    grid[0, 2 * gap + mouse], sharey=psths[0] if psths else None
                )
                psth_panel(axis, data, gap, labels=mouse == 0)
                if mouse:
                    axis.set_ylabel("")
                psths.append(axis)
                panels.append(axis)
            axis = figure.add_subplot(grid[1, 2 * gap : 2 * gap + 2])
            peak_scatter(axis, sessions, "observed", gap)
            axis.text(
                0.98, 1.03, label, transform=axis.transAxes, ha="right", va="bottom"
            )
            axis.set(
                xlabel="stationary peak / mean observed rate",
                ylabel="movement peak / mean observed rate",
            )
            scatters.append(axis)
            panels.append(axis)
        match_scatter_limits(scatters)
        label_panels(panels, x=-0.12, y=1.08)
        write_figure(figure, "flash_observation")

        # One slide per gap class, with the same size and limits.
        rate_limits, scatter_limits = psths[0].get_ylim(), scatters[0].get_xlim()
        for gap, (label, name) in enumerate(
            zip(GAP_LABELS, ("80ms", "40ms"), strict=True)
        ):
            figure, axes = plt.subplots(
                1, 3, figsize=(9.0, 3.4), layout="constrained", width_ratios=(1, 1, 1.3)
            )
            for mouse, data in enumerate(sessions):
                psth_panel(axes[mouse], data, gap, labels=mouse == 0)
                axes[mouse].set_ylim(rate_limits)
            axes[1].set_ylabel("")
            peak_scatter(axes[2], sessions, "observed", gap)
            axes[2].text(
                0.98, 1.03, label, transform=axes[2].transAxes, ha="right", va="bottom"
            )
            axes[2].set(
                xlabel="stationary peak / mean observed rate",
                ylabel="movement peak / mean observed rate",
                xlim=scatter_limits,
            )
            match_scatter_limits(axes[2:])
            label_panels(axes, x=-0.12, y=1.08)
            write_figure(figure, f"flash_observation_{name}")

        figure, axes = plt.subplots(1, 2, figsize=(7.6, 3.5), layout="constrained")
        for gap, (axis, label) in enumerate(zip(axes, GAP_LABELS, strict=True)):
            peak_scatter(axis, sessions, "model", gap)
            axis.text(
                0.98, 1.03, label, transform=axis.transAxes, ha="right", va="bottom"
            )
            axis.set(
                xlabel="stationary kernel peak (Δ log rate)",
                ylabel="movement kernel peak (Δ log rate)",
            )
        match_scatter_limits(axes)
        label_panels(axes, x=-0.12, y=1.08)
        write_figure(figure, "flash_model")

        figure, axes = plt.subplots(2, 3, figsize=(10.8, 7.1), layout="constrained")
        fields = ("observed", "full_model", "without_flash_exit")
        for gap, row in enumerate(axes):
            for axis, field, source in zip(row, fields, RATE_SOURCES, strict=True):
                peak_scatter(axis, sessions, field, gap)
                axis.text(
                    0.5,
                    1.03,
                    f"{source}\n{GAP_LABELS[gap]}",
                    transform=axis.transAxes,
                    ha="center",
                    va="bottom",
                )
                axis.set(
                    xlabel="stationary peak / mean observed rate",
                    ylabel="movement peak / mean observed rate",
                )
        match_scatter_limits(axes.ravel())
        label_panels(axes.ravel(), x=-0.12, y=1.13)
        write_figure(figure, "flash_comparison")


def main():
    classes = json.loads((OUTPUT / "raw_video_trust_classification.json").read_text())
    sessions = [session_data(session, classes[session]) for session in SESSIONS]
    plot(sessions)
    report = {
        "kernel_peak_window_ms": [30, 120],
        "kernel_bin_ms": 10,
        "kernel_scale": "change in log rate, uncorrected for leak",
        "observed_error_bars": "paired-trial bootstrap SD (500 draws)",
        "gap_split_ms": 1000 * dmat.FLASH_JUST_BEFORE_S,
        "gap_rule": "both flashes in the trial pair must have the same preceding-gap class",
        "model_error_bars": "SD over saved trial folds",
        "rate_sources": RATE_SOURCES,
        "rate_peak_scale": "spikes/s; identical 10-ms bins, stationary baseline and 30..120-ms peak",
        "rate_prediction": "saved whole-trial held-out folds; observed history; raw flash-by-exit removal; no refit",
        "rate_error_bars": "paired-trial bootstrap SD (500 shared draws); fixed fitted models",
        "rate_scatter_axes": "linear; population and flash-excited means with complete SEM bars; no individual units",
        "rate_normalization": "one observed mean per unit over full valid model intervals of the selected paired trials; pooled spike counts / valid seconds; same denominator for all sources, periods and gaps; recomputed in each stratified paired bootstrap draw",
        "display_selection": "all eligible units mean: open markers; saved flash-excited mean: filled markers; all means alpha 0.8; mouse by marker shape, source by color",
        "summary_selection": "scatters show both nested group means with SEM across units; PSTHs and existing report summaries use the saved flash-excited group only; per-unit bootstrap and kernel-fold spreads are saved but not drawn",
        "sessions": {
            data["subject"]: {
                "unit_ids": data["unit_ids"],
                "responsive_unit_ids": np.asarray(data["unit_ids"])[
                    data["responsive"]
                ].tolist(),
                "paired_trials": data["trials"],
                "trial_selection": data["selection"],
                "trial_groups": data["trial_groups"],
                "observed_peaks": data["rate_peaks"][:, :, 0].tolist(),
                "observed_bootstrap_sd": data["rate_bootstrap_sd"][:, :, 0].tolist(),
                "rate_peaks": data["rate_peaks"].tolist(),
                "rate_bootstrap_sd": data["rate_bootstrap_sd"].tolist(),
                "normalization_rate_spikes_s": data["normalization_rate"].tolist(),
                "normalized_rate_peaks": data["normalized_rate_peaks"].tolist(),
                "normalized_rate_bootstrap_sd": data[
                    "normalized_rate_bootstrap_sd"
                ].tolist(),
                "normalized_rate_comparison": {
                    label: {
                        source: comparison_summary(
                            data["normalized_rate_peaks"][
                                data["responsive"], gap, source_index
                            ]
                        )
                        for source_index, source in enumerate(RATE_SOURCES)
                    }
                    for gap, label in enumerate(GAP_LABELS)
                },
                "contribution_bootstrap_sd": data["contribution_bootstrap_sd"].tolist(),
                "kernel_peaks": data["model"].tolist(),
                "kernel_fold_sd": data["model_sd"].tolist(),
                "observed": {
                    label: comparison_summary(
                        data["rate_peaks"][data["responsive"], gap, 0]
                    )
                    for gap, label in enumerate(GAP_LABELS)
                },
                "rate_comparison": {
                    label: {
                        source: comparison_summary(
                            data["rate_peaks"][data["responsive"], gap, source_index]
                        )
                        for source_index, source in enumerate(RATE_SOURCES)
                    }
                    for gap, label in enumerate(GAP_LABELS)
                },
                "flash_exit_contribution": {
                    label: {
                        "unit_values": (
                            np.diff(data["rate_peaks"][:, gap, 1], axis=-1)
                            - np.diff(data["rate_peaks"][:, gap, 2], axis=-1)
                        )
                        .ravel()
                        .tolist(),
                        "mean": float(
                            (
                                np.diff(
                                    data["rate_peaks"][data["responsive"], gap, 1],
                                    axis=-1,
                                )
                                - np.diff(
                                    data["rate_peaks"][data["responsive"], gap, 2],
                                    axis=-1,
                                )
                            ).mean()
                        ),
                    }
                    for gap, label in enumerate(GAP_LABELS)
                },
                "kernel": {
                    label: comparison_summary(data["model"][data["responsive"], gap])
                    for gap, label in enumerate(GAP_LABELS)
                },
            }
            for data in sessions
        },
    }
    (OUTPUT / "flash_movement.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
