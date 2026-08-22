"""Run and inspect the locomotion peak analysis.

This script computes peak responses, writes the canonical PDF, and can
optionally leave an interactive matplotlib window open for inspection.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Literal, TypedDict

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from spks.event_aligned import population_peth

from thesis.ephys.io_session_units import (
    fetch_good_unit_metrics_table,
    fetch_good_units,
)
from thesis.ephys.locomotion import (
    extract_paired_stim_anchors,
    load_trial_classification,
)

FIGURE_ROOT = Path(os.environ.get("THESIS_FIGURE_ROOT", "figures"))
FIGURE_DIR = FIGURE_ROOT / "locomotion"

SUBJECT_SESSIONS = [
    ("GRB006", "20240821_121447"),
    ("GRB058", "20260312_134952"),
]

UNIT_CRITERIA_ID = 1
FS_RS_BOUNDARY_MS = 0.4
BACKGROUND_DOT_ALPHA = 0.2
MEAN_CI_LEVEL = 0.95


def compute_locomotion_peaks(
    subject: str, session: str, unit_criteria_id: int = 1
) -> pd.DataFrame:
    """Return stationary and movement peaks for all good units in one session."""
    stationary_events, movement_events = extract_paired_stim_anchors(
        load_trial_classification(subject, session)
    )
    if stationary_events.size == 0 or movement_events.size == 0:
        raise RuntimeError(f"No paired locomotion trials for {subject} {session}.")

    spike_times_by_unit = fetch_good_units(subject, session, unit_criteria_id)
    unit_ids = sorted(spike_times_by_unit)
    if not unit_ids:
        raise RuntimeError(f"No good units found for {subject} {session}.")
    spike_times = [spike_times_by_unit[unit_id] for unit_id in unit_ids]

    stationary_peth, bin_edges, _ = population_peth(
        all_spike_times=spike_times,
        alignment_times=stationary_events,
        pre_seconds=0.1,
        post_seconds=0.15,
        binwidth_ms=10,
    )
    stationary_peth = stationary_peth / 0.01
    movement_peth, _, _ = population_peth(
        all_spike_times=spike_times,
        alignment_times=movement_events,
        pre_seconds=0.1,
        post_seconds=0.15,
        binwidth_ms=10,
    )
    movement_peth = movement_peth / 0.01

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    stationary_mean = stationary_peth.mean(axis=1)
    movement_mean = movement_peth.mean(axis=1)
    baseline_mask = (bin_centers >= -0.04) & (bin_centers < 0.0)
    response_mask = (bin_centers >= 0.03) & (bin_centers < 0.12)
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


class PeakResult(TypedDict):
    subject: str
    stat_peak: np.ndarray
    move_peak: np.ndarray
    fast_spiking_mask: np.ndarray
    regular_spiking_mask: np.ndarray


MaskKey = Literal["all_units", "regular_spiking_mask", "fast_spiking_mask"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--linear-scale",
        action="store_true",
        help="Save a linear-scale scatter version instead of the default log-scale output.",
    )
    parser.add_argument(
        "--split-by-waveform",
        action="store_true",
        help="Split unit clouds and means by putative FS/RS waveform class.",
    )
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

    log_scale = not args.linear_scale
    peak_results: list[PeakResult] = []

    for subject, session in SUBJECT_SESSIONS:
        print(f"\nComputing locomotion peaks: {subject} {session}")
        peak_table = compute_locomotion_peaks(
            subject, session, unit_criteria_id=UNIT_CRITERIA_ID
        )
        if peak_table.empty:
            raise RuntimeError(
                f"No locomotion peaks found for {subject} {session} "
                f"with unit_criteria_id={UNIT_CRITERIA_ID} and passes=1."
            )
        for column in ["stat_peak", "move_peak", "stat_latency", "move_latency"]:
            values = peak_table[column].to_numpy(dtype=float)
            if not np.all(np.isfinite(values)):
                raise RuntimeError(
                    f"Non-finite {column} values for {subject} {session}."
                )

        unit_ids = peak_table["unit_id"].astype(int).tolist()
        unit_metrics = fetch_good_unit_metrics_table(subject, session, UNIT_CRITERIA_ID)
        waveform_duration_ms = (
            unit_metrics.set_index("unit_id")
            .reindex(unit_ids)["spike_duration_ms"]
            .to_numpy()
        )
        if np.any(~np.isfinite(waveform_duration_ms) | (waveform_duration_ms <= 0)):
            raise RuntimeError(f"Invalid waveform duration for {subject} {session}.")
        fast_spiking_mask = waveform_duration_ms <= FS_RS_BOUNDARY_MS
        regular_spiking_mask = waveform_duration_ms > FS_RS_BOUNDARY_MS
        if not np.all(fast_spiking_mask | regular_spiking_mask):
            raise RuntimeError(f"Waveform class assignment failed for {subject}.")

        print(f"  Units: {len(peak_table)}")
        print(
            "  Median latency (stat, move): "
            f"{peak_table['stat_latency'].median():.3f}s, "
            f"{peak_table['move_latency'].median():.3f}s"
        )
        peak_results.append(
            {
                "subject": subject,
                "stat_peak": peak_table["stat_peak"].to_numpy(dtype=float),
                "move_peak": peak_table["move_peak"].to_numpy(dtype=float),
                "fast_spiking_mask": fast_spiking_mask,
                "regular_spiking_mask": regular_spiking_mask,
            }
        )

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 6.0))
    subject_colors = sns.color_palette("Set1")

    all_peak_values = np.concatenate(
        [
            values
            for result in peak_results
            for values in (result["stat_peak"], result["move_peak"])
        ]
    )
    if log_scale:
        all_peak_values = np.maximum(all_peak_values, 0) + 0.1
        lower_limit = 0.1
        upper_limit = max(1.0, float(np.percentile(all_peak_values, 99) * 1.05))
    else:
        lower_limit = 0.0
        upper_limit = max(5.0, float(np.percentile(all_peak_values, 99) * 1.05))

    marker_specs: list[tuple[MaskKey, str, str]] = (
        [
            ("regular_spiking_mask", "o", "RS"),
            ("fast_spiking_mask", "^", "FS"),
        ]
        if args.split_by_waveform
        else [("all_units", "o", "All units")]
    )

    for subject_index, result in enumerate(peak_results):
        color = subject_colors[subject_index]
        stat_peak = result["stat_peak"]
        move_peak = result["move_peak"]
        all_units = np.ones_like(stat_peak, dtype=bool)
        if log_scale:
            plotted_stat = np.maximum(stat_peak, 0) + 0.1
            plotted_move = np.maximum(move_peak, 0) + 0.1
        else:
            plotted_stat = stat_peak
            plotted_move = move_peak

        for mask_key, marker, cell_class in marker_specs:
            unit_mask = all_units if mask_key == "all_units" else result[mask_key]
            if not np.any(unit_mask):
                continue
            ax.scatter(
                plotted_stat[unit_mask],
                plotted_move[unit_mask],
                s=18,
                alpha=BACKGROUND_DOT_ALPHA,
                color=color,
                marker=marker,
                linewidths=0,
                label="_nolegend_",
                zorder=2,
            )
            summaries = []
            for values in (plotted_stat[unit_mask], plotted_move[unit_mask]):
                scale_values = np.log(values) if log_scale else values
                mean = float(scale_values.mean())
                lower, upper = (
                    (mean, mean)
                    if values.size == 1
                    else stats.t.interval(
                        MEAN_CI_LEVEL,
                        df=values.size - 1,
                        loc=mean,
                        scale=stats.sem(scale_values),
                    )
                )
                summaries.append(
                    np.exp([mean, lower, upper])
                    if log_scale
                    else np.array([mean, lower, upper])
                )
            (mean_x, lower_x, upper_x), (mean_y, lower_y, upper_y) = summaries
            ax.errorbar(
                mean_x,
                mean_y,
                xerr=np.array([[mean_x - lower_x], [upper_x - mean_x]]),
                yerr=np.array([[mean_y - lower_y], [upper_y - mean_y]]),
                fmt=marker,
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
                label=(
                    f"{result['subject']} {cell_class}"
                    if args.split_by_waveform
                    else result["subject"]
                ),
            )

    ax.plot(
        [lower_limit, upper_limit], [lower_limit, upper_limit], "k--", alpha=0.4, lw=0.8
    )
    ax.set_xlim(lower_limit, upper_limit)
    ax.set_ylim(lower_limit, upper_limit)
    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_aspect("equal")
    ax.set_xlabel("Stationary peak (baseline-corrected sp/s)")
    ax.set_ylabel("Movement peak (baseline-corrected sp/s)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.12, top=0.98)

    if not args.no_save:
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        suffix_parts = [
            "condition_peak_from_locomotion_peaks",
            "paired_last_stat_first_move",
            "shared_stat_baseline",
        ]
        if not args.split_by_waveform:
            suffix_parts.append("no_waveform_split")
        if log_scale:
            suffix_parts.append("log")
        save_path = FIGURE_DIR / ("_".join(suffix_parts) + ".pdf")
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"\nSaved -> {save_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
