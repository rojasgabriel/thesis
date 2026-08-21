"""Run and inspect the table-backed locomotion peak analysis.

This script reads peak responses from `LocomotionPeaks`, writes the canonical
PDF, and can optionally leave an interactive matplotlib window open for
inspection.
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

from thesis.ephys.utils.analysis_stats import mean_and_t_ci
from thesis.ephys.utils.unit_metrics import fetch_waveform_durations_ms

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

    from labdata_plugin.schema import LocomotionPeaks

    log_scale = not args.linear_scale
    peak_results: list[PeakResult] = []

    for subject, session in SUBJECT_SESSIONS:
        print(f"\nLoading LocomotionPeaks rows: {subject} {session}")
        relation = (
            LocomotionPeaks() * LocomotionPeaks.key_source
            & f'subject_name = "{subject}"'
            & f'session_name = "{session}"'
            & f"unit_criteria_id = {UNIT_CRITERIA_ID}"
            & "passes = 1"
        )
        rows = relation.fetch(
            "unit_id",
            "stat_peak",
            "move_peak",
            "stat_latency",
            "move_latency",
            as_dict=True,
        )
        peak_table = pd.DataFrame(rows).sort_values("unit_id").reset_index(drop=True)
        if peak_table.empty:
            raise RuntimeError(
                f"No LocomotionPeaks rows found for {subject} {session} "
                f"with unit_criteria_id={UNIT_CRITERIA_ID} and passes=1."
            )
        for column in ["stat_peak", "move_peak", "stat_latency", "move_latency"]:
            values = peak_table[column].to_numpy(dtype=float)
            if not np.all(np.isfinite(values)):
                raise RuntimeError(
                    f"Non-finite {column} values for {subject} {session}."
                )

        unit_ids = peak_table["unit_id"].astype(int).tolist()
        waveform_duration_ms = fetch_waveform_durations_ms(
            subject, session, unit_ids, strict=True, unit_criteria_id=UNIT_CRITERIA_ID
        )
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

    if log_scale:
        all_peak_values = np.concatenate(
            [
                np.maximum(result[key], 0) + 0.1
                for result in peak_results
                for key in ["stat_peak", "move_peak"]
            ]
        )
        lower_limit = 0.1
        upper_limit = max(1.0, float(np.percentile(all_peak_values, 99) * 1.05))
    else:
        all_peak_values = np.concatenate(
            [
                result[key]
                for result in peak_results
                for key in ["stat_peak", "move_peak"]
            ]
        )
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
            mean_x, lower_x, upper_x = mean_and_t_ci(
                plotted_stat[unit_mask],
                log_scale=log_scale,
                ci_level=MEAN_CI_LEVEL,
                drop_nonfinite=False,
            )
            mean_y, lower_y, upper_y = mean_and_t_ci(
                plotted_move[unit_mask],
                log_scale=log_scale,
                ci_level=MEAN_CI_LEVEL,
                drop_nonfinite=False,
            )
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
