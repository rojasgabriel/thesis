"""Time-resolved task rate encoding."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import pandas as pd
import seaborn as sns

from task_rate_tuning_shared import (
    FIGURE_DIR,
    TIMECOURSE_BIN_EDGES_S,
    load_all_sessions,
)
from ephys.src.utils.analysis_rate_tuning_timecourse import (
    compute_timecourse_responses,
    summarize_timecourse_encoding,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=FIGURE_DIR)
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def build_timecourse_tables(session_payloads):
    timecourse_responses = []
    for subject, session, _align_ev, spike_times_by_unit, windows in session_payloads:
        table = compute_timecourse_responses(
            windows,
            spike_times_by_unit,
            TIMECOURSE_BIN_EDGES_S,
        )
        table.insert(0, "session", session)
        table.insert(0, "subject", subject)
        timecourse_responses.append(table)
    timecourse_responses_df = pd.concat(timecourse_responses, ignore_index=True)
    summary, coefficients = summarize_timecourse_encoding(timecourse_responses_df)
    subject = timecourse_responses_df["subject"].iloc[0]
    session = timecourse_responses_df["session"].iloc[0]
    for table in (summary, coefficients):
        table.insert(0, "session", session)
        table.insert(0, "subject", subject)
    combined = pd.concat(
        [
            summary.assign(summary_level="population_timecourse"),
            coefficients.assign(summary_level="unit_timecourse"),
        ],
        ignore_index=True,
        sort=False,
    )
    return summary, coefficients, combined


def plot_timecourse_encoding(timecourse_summary, timecourse_coefficients):
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.2))
    bin_centers = (
        timecourse_summary["bin_start_s"] + timecourse_summary["bin_end_s"]
    ) / 2
    axes[0].plot(
        bin_centers,
        timecourse_summary["rate_correlation"],
        color="black",
        marker="o",
    )
    axes[0].axhline(0, color="0.4", linewidth=0.8, linestyle="--")
    axes[0].axvline(0.5, color="red", linewidth=1.0, linestyle="--")
    axes[0].set(xlabel="time from first flash (s)", ylabel="rate correlation")

    axes[1].plot(
        bin_centers,
        timecourse_summary["category_accuracy"],
        color="black",
        marker="o",
    )
    axes[1].axhline(0.5, color="0.4", linewidth=0.8, linestyle="--")
    axes[1].axvline(0.5, color="red", linewidth=1.0, linestyle="--")
    axes[1].set(xlabel="time from first flash (s)", ylabel="category accuracy")

    heatmap_values = timecourse_coefficients.pivot_table(
        index="unit_id",
        columns="bin_start_s",
        values="signed_rate_coefficient",
    )
    sort_index = heatmap_values.mean(axis=1).sort_values().index
    sns.heatmap(
        heatmap_values.loc[sort_index],
        cmap="vlag",
        center=0,
        yticklabels=False,
        xticklabels=[f"{value:.1f}" for value in heatmap_values.columns],
        cbar_kws={"label": "signed rate coefficient"},
        ax=axes[2],
    )
    axes[2].set(xlabel="bin start (s)", ylabel="units")
    for ax in axes[:2]:
        ax.grid(False)
    fig.tight_layout()
    return fig


def write_pdf(output_path: Path, figure) -> None:
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib import pyplot as plt

    with PdfPages(output_path) as pdf:
        pdf.savefig(figure, dpi=300)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    matplotlib.use("Agg")
    sns.set_theme(style="whitegrid", context="paper")
    *_, session_payloads = load_all_sessions()
    summary, coefficients, combined = build_timecourse_tables(session_payloads)
    if args.no_save:
        print("\nBuilt timecourse outputs without writing files.")
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output_dir / "timecourse_encoding_summary.csv", index=False)
    write_pdf(
        args.output_dir / "rate_tuning_timecourse.pdf",
        plot_timecourse_encoding(summary, coefficients),
    )
    print(f"\nSaved outputs -> {args.output_dir}")


if __name__ == "__main__":
    main()
