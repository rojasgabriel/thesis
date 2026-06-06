"""Matched stationary-vs-movement locomotion scatters across training sessions."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import types

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[2]
if "ephys" not in sys.modules:
    package = types.ModuleType("ephys")
    package.__path__ = [str(REPO_ROOT)]
    sys.modules["ephys"] = package
sys.path.insert(0, str(REPO_ROOT))

from ephys.src.config.locomotion import (  # noqa: E402
    BASELINE_WINDOW,
    PETH_KWARGS,
    RESP_WINDOW,
)
from ephys.src.utils.analysis_conditioned_stim import (  # noqa: E402
    build_trial_stim_classification,
)
from ephys.src.utils.analysis_peth import compute_population_peth  # noqa: E402
from ephys.src.utils.io_chipmunk_trials import fetch_trial_metadata  # noqa: E402
from ephys.src.utils.io_digital_events import fetch_session_events  # noqa: E402
from ephys.src.utils.io_session_units import fetch_good_units  # noqa: E402
from ephys.src.utils.trial_alignment import enrich_chipmunk_trial_table  # noqa: E402

FIGURE_ROOT = Path(
    os.environ.get("EPHYS_FIGURE_ROOT", "/Users/gabriel/lib/ephys/figures")
)
FIGURE_DIR = FIGURE_ROOT / "locomotion"
OUTPUT_PDF = FIGURE_DIR / "training_progression_matched_stationary_scatter.pdf"
OUTPUT_CSV = FIGURE_DIR / "training_progression_matched_stationary_summary.csv"

EXPERT_SESSION = ("GRB006", "20240821_121447")
TRAINING_SESSIONS = [
    ("GRB058", "20260224_152424"),
    ("GRB058", "20260319_131303"),
    ("GRB058", "20260421_160125"),
    ("GRB058", "20260526_124438"),
    ("GRB059", "20260225_154153"),
    ("GRB059", "20260319_142123"),
    ("GRB059", "20260421_134936"),
    ("GRB059", "20260526_140834"),
]
MATCH_TOLERANCE_S = 0.100
UNIT_CRITERIA_ID = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=FIGURE_DIR)
    return parser.parse_args()


def first_flash_by_trial(
    aligned_events: dict[str, np.ndarray],
    trial_df: pd.DataFrame,
) -> dict[int, float]:
    first_stims = np.asarray(aligned_events["first_stim_ev_15ms"], dtype=float)
    trial_starts = trial_df["trial_start_ts"].to_numpy(dtype=float)
    result = {}
    for position, trial_idx in enumerate(trial_df.index):
        start = trial_starts[position]
        stop = (
            trial_starts[position + 1] if position + 1 < len(trial_starts) else np.inf
        )
        matches = first_stims[(first_stims >= start) & (first_stims < stop)]
        if matches.size:
            result[int(trial_idx)] = float(matches[0])
    return result


def load_classified_session(subject: str, session: str) -> tuple[pd.DataFrame, dict]:
    aligned_events = fetch_session_events(subject, session)
    trial_df = fetch_trial_metadata(subject, session, aligned_events)
    if trial_df is None:
        raise RuntimeError(f"Could not load Chipmunk trials for {subject} {session}.")
    trial_df = enrich_chipmunk_trial_table(trial_df)
    classified = build_trial_stim_classification(aligned_events, trial_df)
    if classified.empty:
        raise RuntimeError(f"No paired locomotion trials for {subject} {session}.")
    first_flash = first_flash_by_trial(aligned_events, trial_df)
    classified["first_flash"] = classified["trial_idx"].map(first_flash)
    classified = classified[np.isfinite(classified["first_flash"])].copy()
    if classified.empty:
        raise RuntimeError(f"No 15 ms first-flash trials for {subject} {session}.")
    return classified, aligned_events


def training_last_stationary_latency(classified: pd.DataFrame) -> np.ndarray:
    latencies = []
    for row in classified.itertuples(index=False):
        latencies.append(float(row.stationary_stims[-1]) - float(row.first_flash))
    return np.asarray(latencies, dtype=float)


def training_anchor_times(classified: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    stat_times = [float(stims[-1]) for stims in classified["stationary_stims"]]
    move_times = [float(stims[0]) for stims in classified["movement_stims"]]
    return np.asarray(stat_times, dtype=float), np.asarray(move_times, dtype=float)


def expert_anchor_times(
    classified: pd.DataFrame,
    target_latency_s: float,
    tolerance_s: float = MATCH_TOLERANCE_S,
) -> tuple[np.ndarray, np.ndarray]:
    stat_times = []
    move_times = []
    for row in classified.itertuples(index=False):
        target_time = float(row.first_flash) + target_latency_s
        stationary = np.asarray(row.stationary_stims, dtype=float)
        if stationary.size == 0:
            continue
        nearest_idx = int(np.argmin(np.abs(stationary - target_time)))
        if abs(stationary[nearest_idx] - target_time) > tolerance_s:
            continue
        stat_times.append(float(stationary[nearest_idx]))
        move_times.append(float(row.movement_stims[0]))
    return np.asarray(stat_times, dtype=float), np.asarray(move_times, dtype=float)


def peak_table_for_session(
    subject: str,
    session: str,
    stat_times: np.ndarray,
    move_times: np.ndarray,
) -> pd.DataFrame:
    spikes_by_unit = fetch_good_units(
        subject,
        session,
        unit_criteria_id=UNIT_CRITERIA_ID,
    )
    if not spikes_by_unit:
        raise RuntimeError(f"No good units for {subject} {session}.")
    unit_ids = np.asarray(list(spikes_by_unit.keys()), dtype=int)
    spike_times = [spikes_by_unit[int(unit_id)] for unit_id in unit_ids]
    stat_peth, _, bin_centers = compute_population_peth(
        spike_times,
        stat_times,
        **PETH_KWARGS,
    )
    move_peth, _, _ = compute_population_peth(
        spike_times,
        move_times,
        **PETH_KWARGS,
    )
    stat_mean = stat_peth.mean(axis=1)
    move_mean = move_peth.mean(axis=1)
    baseline_mask = (bin_centers >= BASELINE_WINDOW[0]) & (
        bin_centers < BASELINE_WINDOW[1]
    )
    response_mask = (bin_centers >= RESP_WINDOW[0]) & (bin_centers < RESP_WINDOW[1])
    response_times = bin_centers[response_mask]
    baseline = stat_mean[:, baseline_mask].mean(axis=1)
    stat_response = stat_mean[:, response_mask] - baseline[:, np.newaxis]
    move_response = move_mean[:, response_mask] - baseline[:, np.newaxis]
    stat_peak_idx = np.argmax(stat_response, axis=1)
    move_peak_idx = np.argmax(move_response, axis=1)
    rows = []
    for unit_index, unit_id in enumerate(unit_ids):
        stat_peak = float(stat_response[unit_index, stat_peak_idx[unit_index]])
        move_peak = float(move_response[unit_index, move_peak_idx[unit_index]])
        rows.append(
            {
                "subject": subject,
                "session": session,
                "unit_id": int(unit_id),
                "stat_peak": stat_peak,
                "move_peak": move_peak,
                "stat_latency": float(response_times[stat_peak_idx[unit_index]]),
                "move_latency": float(response_times[move_peak_idx[unit_index]]),
                "delta_move_stat": move_peak - stat_peak,
            }
        )
    return pd.DataFrame(rows)


def session_summary(
    unit_table: pd.DataFrame,
    *,
    n_trials: int,
    anchor_latency_s: float,
    target_latency_s: float,
) -> dict:
    delta = unit_table["delta_move_stat"]
    return {
        "subject": unit_table["subject"].iloc[0],
        "session": unit_table["session"].iloc[0],
        "n_units": int(unit_table["unit_id"].nunique()),
        "n_trials": int(n_trials),
        "anchor_latency_s": float(anchor_latency_s),
        "target_training_latency_s": float(target_latency_s),
        "median_delta_sp_s": float(delta.median()),
        "mean_delta_sp_s": float(delta.mean()),
        "percent_move_gt_stat": float((delta > 0).mean() * 100),
    }


def build_outputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    loaded = {}
    training_latencies = []
    for subject, session in TRAINING_SESSIONS:
        try:
            classified, _aligned_events = load_classified_session(subject, session)
        except Exception as exc:
            print(f"Skipping target-latency source {subject} {session}: {exc}")
            continue
        loaded[(subject, session)] = classified
        training_latencies.extend(training_last_stationary_latency(classified))
    if not training_latencies:
        raise RuntimeError("Could not compute a training-mouse stationary latency.")
    target_latency_s = float(np.nanmean(training_latencies))
    print(f"Training last-stationary target latency: {target_latency_s:.3f}s")

    unit_tables = []
    summary_rows = []
    for subject, session in [EXPERT_SESSION, *TRAINING_SESSIONS]:
        try:
            if (subject, session) in loaded:
                classified = loaded[(subject, session)]
            else:
                classified, _aligned_events = load_classified_session(subject, session)
            if (subject, session) == EXPERT_SESSION:
                stat_times, move_times = expert_anchor_times(
                    classified,
                    target_latency_s,
                )
            else:
                stat_times, move_times = training_anchor_times(classified)
            if stat_times.size == 0 or move_times.size == 0:
                raise RuntimeError("No matched stationary/movement anchors.")
            unit_table = peak_table_for_session(
                subject, session, stat_times, move_times
            )
        except Exception as exc:
            print(f"Skipping {subject} {session}: {exc}")
            continue
        anchor_latency_s = float(np.nanmedian(stat_times - move_times + 0.0))
        if (subject, session) == EXPERT_SESSION:
            anchor_latency_s = target_latency_s
        else:
            anchor_latency_s = float(
                np.nanmedian(training_last_stationary_latency(classified))
            )
        unit_tables.append(unit_table)
        summary_rows.append(
            session_summary(
                unit_table,
                n_trials=len(stat_times),
                anchor_latency_s=anchor_latency_s,
                target_latency_s=target_latency_s,
            )
        )
        print(
            f"Loaded {subject} {session}: {len(unit_table)} units, "
            f"{len(stat_times)} paired trials"
        )
    if not unit_tables:
        raise RuntimeError("No sessions produced locomotion progression outputs.")
    return pd.concat(unit_tables, ignore_index=True), pd.DataFrame(summary_rows)


def plot_scatter(unit_table: pd.DataFrame, summary_table: pd.DataFrame):
    from matplotlib import pyplot as plt

    sessions = summary_table[["subject", "session"]].itertuples(index=False, name=None)
    sessions = list(sessions)
    ncols = 3
    nrows = int(np.ceil(len(sessions) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10.5, 3.4 * nrows), squeeze=False)
    palette = dict(
        zip(sorted(unit_table["subject"].unique()), sns.color_palette("Set1"))
    )
    plotted = (
        np.maximum(
            unit_table[["stat_peak", "move_peak"]].to_numpy(dtype=float),
            0,
        )
        + 0.1
    )
    upper_limit = max(1.0, float(np.nanpercentile(plotted, 99) * 1.05))
    for ax, (subject, session) in zip(axes.flat, sessions, strict=False):
        plot_df = unit_table[
            (unit_table["subject"] == subject) & (unit_table["session"] == session)
        ].copy()
        summary = summary_table[
            (summary_table["subject"] == subject)
            & (summary_table["session"] == session)
        ].iloc[0]
        x = np.maximum(plot_df["stat_peak"].to_numpy(dtype=float), 0) + 0.1
        y = np.maximum(plot_df["move_peak"].to_numpy(dtype=float), 0) + 0.1
        ax.scatter(x, y, s=12, alpha=0.22, color=palette[subject], linewidths=0)
        ax.plot([0.1, upper_limit], [0.1, upper_limit], "k--", alpha=0.4, lw=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.1, upper_limit)
        ax.set_ylim(0.1, upper_limit)
        ax.set_aspect("equal")
        ax.set_title(f"{subject} {session}", fontsize=9)
        ax.text(
            0.04,
            0.96,
            (
                f"n={int(summary.n_units)} units, {int(summary.n_trials)} trials\n"
                f"{summary.percent_move_gt_stat:.0f}% move>stat\n"
                f"median d={summary.median_delta_sp_s:+.2f} sp/s"
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
        )
        ax.grid(False)
    for ax in axes.flat[len(sessions) :]:
        ax.axis("off")
    for ax in axes[-1, :]:
        ax.set_xlabel("Stationary peak (baseline-corrected sp/s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Movement peak (baseline-corrected sp/s)")
    fig.tight_layout()
    return fig


def main() -> None:
    args = parse_args()
    matplotlib.use("Agg")
    sns.set_theme(style="white", context="paper")
    unit_table, summary_table = build_outputs()
    if args.no_save:
        print("\nBuilt locomotion training progression outputs without writing files.")
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / OUTPUT_CSV.name
    pdf_path = args.output_dir / OUTPUT_PDF.name
    summary_table.to_csv(summary_path, index=False)
    fig = plot_scatter(unit_table, summary_table)
    fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
    print(f"\nSaved -> {summary_path}")
    print(f"Saved -> {pdf_path}")


if __name__ == "__main__":
    main()
