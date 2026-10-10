"""Compare V1 flash responses before and after the mouse leaves the center port.

Comparison: in each trial, the script compares two 15 ms flashes. The first is
the last flash between the matched center entry and center exit (stationary).
The second is the first flash from center exit to response-port entry
(movement). "Movement" is a task epoch. The script does not measure locomotion.
The two flashes come from the same trial, and each unit gives one peak for each
condition. Sessions: GRB006 20240821_121447 and GRB064
20260916_132939 (one session for each subject). Unit selection: units pass
quality criterion 1 and stability parameter 0. They also have the stored
"excited" classification in StimulusResponse (stimulus-response parameter 0).
Event alignment: spikes align to flash onset on the visual-stimulus TTL
channel. The PETH uses 10 ms bins from -90 to 140 ms. Baseline: the stationary
trial-mean rate from -40 to 0 ms. The script subtracts this one baseline from
both conditions. Peak: the maximum baseline-corrected trial-mean rate from 30
to 120 ms. The script finds the peak separately for each condition. Exclusions:
the script removes trials with no matched center entry, center exit, and
response-port entry sequence. It also removes trials that do not have one
stationary flash and one movement flash. It uses only 15 ms flashes, so it
ignores 30 ms flashes and flashes of unknown width. It does not remove other
flashes inside the PETH window. Sampling unit: the unit. Panel a shows each
unit and the subject mean ± SEM across units on both axes. Panel b shows
movement minus stationary peak for each unit and the subject mean ± SEM.
Display limit: panel a stops at 60 sp/s. Units above this limit stay in the
means, and the script prints their count.

The script writes locomotion/locomotion_peaks.svg under THESIS_FIGURE_ROOT. The
default root is "figures".
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats
from spks.event_aligned import population_peth

# Keep this import above `labdata_plugin`: after the plugin loads, the name
# `thesis` points to the plugin, and a first import of `thesis.plotting` fails.
from thesis.plotting import FIGURE_STYLE, label_panels, save_figure  # isort: skip
from labdata_plugin.schema import StimulusResponse
from thesis.ephys.trials import build_trial_table
from thesis.ephys.units import fetch_unit_table

FIGURE_ROOT = Path(os.environ.get("THESIS_FIGURE_ROOT", "figures"))
FIGURE_DIR = FIGURE_ROOT / "locomotion"

SUBJECT_SESSIONS = [
    ("GRB006", "20240821_121447"),
    ("GRB064", "20260916_132939"),
]

UNIT_CRITERIA_ID = 1
STABILITY_PARAM_ID = 0
STIM_RESPONSE_PARAM_ID = 0
PETH_BINWIDTH_MS = 10
SCATTER_MAX_RATE = 60.0
BACKGROUND_DOT_ALPHA = 0.2
SUBJECT_COLORS = ("#E41A1C", "#377EB8", "#4DAF4A")


def load_trial_classification(subject: str, session: str) -> pd.DataFrame:
    """Classify 15 ms pulses as stationary or movement for each trial."""
    classification_records = []
    for _, trial in build_trial_table(subject, session).iterrows():
        cp_entry = trial["center_entry_s"]
        cp_exit = trial["center_exit_s"]
        rp_entry = trial["response_port_entry_s"]
        if not all(np.isfinite([cp_entry, cp_exit, rp_entry])):
            continue
        stim_times = np.asarray(
            [
                timestamp
                for timestamp, width_ms in zip(
                    trial["stim_pulse_times_s"],
                    trial["stim_pulse_widths_ms"],
                    strict=True,
                )
                if width_ms == 15
            ],
            dtype=float,
        )
        stationary_stims = stim_times[
            (stim_times >= cp_entry) & (stim_times < cp_exit)
        ].tolist()
        movement_stims = stim_times[
            (stim_times >= cp_exit) & (stim_times <= rp_entry)
        ].tolist()
        if stationary_stims and movement_stims:
            classification_records.append(
                {"stationary_stims": stationary_stims, "movement_stims": movement_stims}
            )

    return pd.DataFrame(classification_records)


def extract_paired_stim_anchors(
    trial_classification: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Return last-stationary and first-movement stimulus times."""
    stationary = [stims[-1] for stims in trial_classification["stationary_stims"]]
    movement = [stims[0] for stims in trial_classification["movement_stims"]]
    return np.asarray(stationary, dtype=float), np.asarray(movement, dtype=float)


def compute_locomotion_peaks(subject: str, session: str) -> pd.DataFrame:
    """Return stationary and movement peaks for stable stimulus-excited units."""
    stationary_events, movement_events = extract_paired_stim_anchors(
        load_trial_classification(subject, session)
    )
    if stationary_events.size == 0 or movement_events.size == 0:
        raise RuntimeError(f"No paired locomotion trials for {subject} {session}.")

    units = fetch_unit_table(subject, session, UNIT_CRITERIA_ID, STABILITY_PARAM_ID)
    response_query = {
        "subject_name": subject,
        "session_name": session,
        "unit_criteria_id": UNIT_CRITERIA_ID,
        "stim_response_param_id": STIM_RESPONSE_PARAM_ID,
        "response_type": "excited",
    }
    excited_unit_ids = set((StimulusResponse.Unit & response_query).fetch("unit_id"))
    units = units[units["unit_id"].isin(excited_unit_ids)].sort_values("unit_id")
    if units.empty:
        raise RuntimeError(
            f"No stable stimulus-excited units found for {subject} {session}."
        )
    unit_ids = units["unit_id"].astype(int).tolist()
    spike_times = units["spike_times_s"].tolist()

    trial_means = []
    for events in (stationary_events, movement_events):
        peth, bin_edges, _ = population_peth(
            all_spike_times=spike_times,
            alignment_times=events,
            pre_seconds=0.1,
            post_seconds=0.15,
            binwidth_ms=PETH_BINWIDTH_MS,
        )
        trial_means.append((peth / (PETH_BINWIDTH_MS / 1000)).mean(axis=1))
    stationary_mean, movement_mean = trial_means
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    baseline_mask = (bin_centers >= -0.04) & (bin_centers < 0.0)
    response_mask = (bin_centers >= 0.03) & (bin_centers < 0.12)
    # Hold the baseline fixed so both conditions are measured from the
    # stationary pre-stimulus state.
    baseline = stationary_mean[:, baseline_mask].mean(axis=1)
    stationary_response = stationary_mean[:, response_mask] - baseline[:, None]
    movement_response = movement_mean[:, response_mask] - baseline[:, None]
    stationary_peak_idx = np.argmax(stationary_response, axis=1)
    movement_peak_idx = np.argmax(movement_response, axis=1)
    unit_index = np.arange(len(unit_ids))
    response_times = bin_centers[response_mask]

    return pd.DataFrame(
        {
            "unit_id": unit_ids,
            "stat_peak": stationary_response[unit_index, stationary_peak_idx],
            "stat_latency": response_times[stationary_peak_idx],
            "move_peak": movement_response[unit_index, movement_peak_idx],
            "move_latency": response_times[movement_peak_idx],
        }
    )


def main() -> None:
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    peak_results: list[tuple[str, pd.DataFrame]] = []

    for subject, session in SUBJECT_SESSIONS:
        print(f"\nComputing locomotion peaks: {subject} {session}")
        peak_table = compute_locomotion_peaks(subject, session)
        for column in ["stat_peak", "move_peak", "stat_latency", "move_latency"]:
            values = peak_table[column].to_numpy(dtype=float)
            if not np.all(np.isfinite(values)):
                raise RuntimeError(
                    f"Non-finite {column} values for {subject} {session}."
                )

        print(f"  Units: {len(peak_table)}")
        print(
            "  Median latency (stat, move): "
            f"{peak_table['stat_latency'].median():.3f}s, "
            f"{peak_table['move_latency'].median():.3f}s"
        )
        peak_results.append((subject, peak_table))

    plt.rcParams.update(FIGURE_STYLE)
    fig, (ax, delta_ax) = plt.subplots(
        1, 2, figsize=(7.2, 3.3), gridspec_kw={"width_ratios": [1, 0.65]}
    )

    peaks = pd.concat([table[["stat_peak", "move_peak"]] for _, table in peak_results])
    lower_limit = min(0.0, float(peaks.to_numpy(dtype=float).min()))
    padding = 0.05 * (SCATTER_MAX_RATE - lower_limit)
    lower_limit -= padding

    rng = np.random.default_rng(0)
    for subject_index, (subject, peak_table) in enumerate(peak_results):
        color = SUBJECT_COLORS[subject_index]
        stat_peak = peak_table["stat_peak"].to_numpy(dtype=float)
        move_peak = peak_table["move_peak"].to_numpy(dtype=float)
        outside_view = np.count_nonzero(
            (stat_peak > SCATTER_MAX_RATE) | (move_peak > SCATTER_MAX_RATE)
        )
        if outside_view:
            print(
                f"  {subject}: {outside_view} units exceed the {SCATTER_MAX_RATE:g} "
                "sp/s scatter limit; retained in summaries"
            )
        ax.scatter(
            stat_peak,
            move_peak,
            s=18,
            alpha=BACKGROUND_DOT_ALPHA,
            color=color,
            linewidths=0,
            zorder=2,
        )
        mean_x, sem_x = stat_peak.mean(), stats.sem(stat_peak)
        mean_y, sem_y = move_peak.mean(), stats.sem(move_peak)
        ax.errorbar(
            mean_x,
            mean_y,
            xerr=sem_x,
            yerr=sem_y,
            fmt="o",
            ms=9,
            color=color,
            mfc=color,
            mec="white",
            mew=0.8,
            elinewidth=1.2,
            ecolor=color,
            capsize=2.5,
            alpha=0.95,
            zorder=5,
        )
        ax.annotate(
            subject,
            (mean_x, mean_y),
            xytext=(4, 4),
            textcoords="offset points",
            color=color,
            fontsize=7,
        )

        delta = move_peak - stat_peak
        x = subject_index + rng.uniform(-0.08, 0.08, delta.size)
        delta_ax.scatter(
            x,
            delta,
            s=18,
            alpha=BACKGROUND_DOT_ALPHA,
            color=color,
            linewidths=0,
        )
        delta_ax.errorbar(
            subject_index,
            delta.mean(),
            yerr=stats.sem(delta),
            fmt="o",
            ms=9,
            color=color,
            mfc=color,
            mec="white",
            mew=0.8,
            elinewidth=1.2,
            capsize=2.5,
            zorder=5,
        )

    ax.plot(
        [lower_limit, SCATTER_MAX_RATE],
        [lower_limit, SCATTER_MAX_RATE],
        "k--",
        alpha=0.4,
        lw=0.8,
    )
    ax.set_xlim(lower_limit, SCATTER_MAX_RATE)
    ax.set_ylim(lower_limit, SCATTER_MAX_RATE)
    ax.set_aspect("equal")
    ax.set_xlabel("stationary peak (baseline-corrected spikes/s)")
    ax.set_ylabel("movement peak (baseline-corrected spikes/s)")
    delta_ax.axhline(0, color="0.6", linewidth=0.7)
    delta_ax.set_xticks(
        range(len(peak_results)), [subject for subject, _ in peak_results]
    )
    for label, color in zip(delta_ax.get_xticklabels(), SUBJECT_COLORS):
        label.set_color(color)
    delta_ax.set_ylabel("movement − stationary peak (spikes/s)")
    fig.subplots_adjust(left=0.1, right=0.98, bottom=0.18, top=0.9, wspace=0.45)
    label_panels((ax, delta_ax), -0.2)
    save_path = save_figure(fig, FIGURE_DIR / "locomotion_peaks")
    print(f"\nSaved -> {save_path}")


if __name__ == "__main__":
    main()
