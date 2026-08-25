"""Run and inspect the locomotion peak analysis.

This script computes peak responses, writes the canonical PDF, and can
optionally leave an interactive matplotlib window open for inspection.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats
from spks.event_aligned import population_peth

from labdata_plugin.schema import StimulusResponsiveness
from thesis.ephys.trials import build_trial_table
from thesis.ephys.units import fetch_unit_table

FIGURE_ROOT = Path(os.environ.get("THESIS_FIGURE_ROOT", "figures"))
FIGURE_DIR = FIGURE_ROOT / "locomotion"

SUBJECT_SESSIONS = [
    ("GRB006", "20240821_121447"),
    ("GRB058", "20260224_152424"),
]

UNIT_CRITERIA_ID = 1
STABILITY_PARAM_ID = 0
RESPONSIVENESS_PARAM_ID = 0
PETH_BINWIDTH_MS = 10
SCATTER_MAX_RATE = 60.0
BACKGROUND_DOT_ALPHA = 0.2
SUBJECT_COLORS = ("#E41A1C", "#377EB8")


def load_trial_classification(subject: str, session: str) -> pd.DataFrame:
    """Load the conditioned-stim classification for one session."""
    return build_trial_stim_classification(
        build_trial_table(subject, session)
    ).reset_index(drop=True)


def build_trial_stim_classification(trial_df: pd.DataFrame) -> pd.DataFrame:
    """Classify 15 ms pulses as stationary or movement for each trial."""
    rows = []
    for trial_index, trial in trial_df.iterrows():
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
            rows.append(
                {
                    "trial_idx": trial_index,
                    "cp_entry": cp_entry,
                    "cp_exit": cp_exit,
                    "rp_entry": rp_entry,
                    "stationary_stims": stationary_stims,
                    "movement_stims": movement_stims,
                    "n_cp_entries": sum(
                        timestamp < cp_exit
                        for timestamp in trial["center_port_entry_times_s"]
                    ),
                }
            )

    return pd.DataFrame(rows)


def extract_paired_stim_anchors(
    trial_ts: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Return last-stationary and first-movement stimulus times."""
    paired = trial_ts[
        trial_ts["stationary_stims"].str.len().gt(0)
        & trial_ts["movement_stims"].str.len().gt(0)
    ]
    return (
        np.asarray([stims[-1] for stims in paired["stationary_stims"]], dtype=float),
        np.asarray([stims[0] for stims in paired["movement_stims"]], dtype=float),
    )


def compute_locomotion_peaks(
    subject: str,
    session: str,
    unit_criteria_id: int = 1,
    stability_param_id: int | None = 0,
    responsiveness_param_id: int = 0,
) -> pd.DataFrame:
    """Return stationary and movement peaks for stable stimulus-excited units."""
    stationary_events, movement_events = extract_paired_stim_anchors(
        load_trial_classification(subject, session)
    )
    if stationary_events.size == 0 or movement_events.size == 0:
        raise RuntimeError(f"No paired locomotion trials for {subject} {session}.")

    units = fetch_unit_table(subject, session, unit_criteria_id, stability_param_id)
    excited_unit_ids = StimulusResponsiveness.fetch_excited_unit_ids(
        subject,
        session,
        unit_criteria_id,
        responsiveness_param_id,
        stability_param_id,
    )
    units = units[units["unit_id"].isin(excited_unit_ids)].sort_values("unit_id")
    if units.empty:
        raise RuntimeError(
            f"No stable stimulus-excited units found for {subject} {session}."
        )
    unit_ids = units["unit_id"].astype(int).tolist()
    spike_times = units["spike_times_s"].tolist()

    stationary_peth, bin_edges, _ = population_peth(
        all_spike_times=spike_times,
        alignment_times=stationary_events,
        pre_seconds=0.1,
        post_seconds=0.15,
        binwidth_ms=PETH_BINWIDTH_MS,
    )
    stationary_peth = stationary_peth / (PETH_BINWIDTH_MS / 1000)
    movement_peth, _, _ = population_peth(
        all_spike_times=spike_times,
        alignment_times=movement_events,
        pre_seconds=0.1,
        post_seconds=0.15,
        binwidth_ms=PETH_BINWIDTH_MS,
    )
    movement_peth = movement_peth / (PETH_BINWIDTH_MS / 1000)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    stationary_mean = stationary_peth.mean(axis=1)
    movement_mean = movement_peth.mean(axis=1)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open an interactive matplotlib window after saving the figure.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Build the figure without writing the PDF. Usually paired with --show.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.show:
        matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    peak_results: list[tuple[str, pd.DataFrame]] = []

    for subject, session in SUBJECT_SESSIONS:
        print(f"\nComputing locomotion peaks: {subject} {session}")
        peak_table = compute_locomotion_peaks(
            subject,
            session,
            unit_criteria_id=UNIT_CRITERIA_ID,
            stability_param_id=STABILITY_PARAM_ID,
            responsiveness_param_id=RESPONSIVENESS_PARAM_ID,
        )
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

    fig, (ax, delta_ax) = plt.subplots(
        1, 2, figsize=(7.2, 3.3), gridspec_kw={"width_ratios": [1, 0.65]}
    )

    all_peak_values = np.concatenate(
        [
            values
            for _, result in peak_results
            for values in (
                result["stat_peak"].to_numpy(dtype=float),
                result["move_peak"].to_numpy(dtype=float),
            )
        ]
    )
    lower_limit = min(0.0, float(all_peak_values.min()))
    padding = 0.05 * (SCATTER_MAX_RATE - lower_limit)
    lower_limit -= padding

    rng = np.random.default_rng(0)
    for subject_index, (subject, result) in enumerate(peak_results):
        color = SUBJECT_COLORS[subject_index]
        stat_peak = result["stat_peak"].to_numpy(dtype=float)
        move_peak = result["move_peak"].to_numpy(dtype=float)
        ax.scatter(
            stat_peak,
            move_peak,
            s=18,
            alpha=BACKGROUND_DOT_ALPHA,
            color=color,
            linewidths=0,
            zorder=2,
        )
        summaries = []
        for values in (stat_peak, move_peak):
            mean = float(values.mean())
            summaries.append((mean, float(stats.sem(values))))
        (mean_x, sem_x), (mean_y, sem_y) = summaries
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
        delta_mean = float(delta.mean())
        delta_ax.errorbar(
            subject_index,
            delta_mean,
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
    ax.set_xlabel("Stationary peak (baseline-corrected sp/s)")
    ax.set_ylabel("Movement peak (baseline-corrected sp/s)")
    delta_ax.axhline(0, color="0.6", linewidth=0.7)
    delta_ax.set_xticks(
        range(len(peak_results)), [subject for subject, _ in peak_results]
    )
    for label, color in zip(delta_ax.get_xticklabels(), SUBJECT_COLORS):
        label.set_color(color)
    delta_ax.set_ylabel("Movement − stationary peak (sp/s)")
    for axis in (ax, delta_ax):
        axis.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.1, right=0.98, bottom=0.18, top=0.9, wspace=0.45)
    for label, axis in zip("ab", (ax, delta_ax)):
        position = axis.get_position()
        fig.text(
            position.x0 - 0.025,
            position.y1 + 0.025,
            label,
            fontweight="bold",
            fontsize=9,
            ha="right",
            va="bottom",
        )

    if not args.no_save:
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        save_path = FIGURE_DIR / "locomotion_peaks.pdf"
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"\nSaved -> {save_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
