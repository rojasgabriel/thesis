"""Shuffle-null controls for task rate-tuning FSI."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

from task_rate_tuning_shared import FIGURE_DIR, RANDOM_SEED, load_all_sessions
from ephys.src.utils.analysis_rate_tuning_shuffle import compute_shuffle_fsi_values

N_SHUFFLES = 1000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=FIGURE_DIR)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--n-shuffles", type=int, default=N_SHUFFLES)
    return parser.parse_args()


def build_shuffle_summary(trial_responses, unit_summary, n_shuffles: int):
    population_null, unit_shuffle_values = compute_shuffle_fsi_values(
        trial_responses,
        n_shuffles=n_shuffles,
        seed=RANDOM_SEED,
    )
    unit_p95 = (
        unit_shuffle_values.groupby("unit_id")["frequency_selectivity_index"]
        .quantile(0.95)
        .rename("shuffle_fsi_p95")
        .reset_index()
    )
    unit_thresholds = unit_summary[["unit_id", "frequency_selectivity_index"]].rename(
        columns={"frequency_selectivity_index": "observed_fsi"}
    )
    unit_thresholds = unit_thresholds.merge(unit_p95, on="unit_id", how="left")
    unit_thresholds["exceeds_shuffle_p95"] = (
        unit_thresholds["observed_fsi"] > unit_thresholds["shuffle_fsi_p95"]
    )
    observed_median = unit_summary["frequency_selectivity_index"].median()
    observed_mean = unit_summary["frequency_selectivity_index"].mean()
    population_null["observed_median_fsi"] = observed_median
    population_null["observed_mean_fsi"] = observed_mean
    unit_thresholds["observed_population_median_fsi"] = observed_median
    unit_thresholds["observed_population_mean_fsi"] = observed_mean
    return (
        population_null.assign(summary_level="population_shuffle"),
        unit_thresholds.assign(summary_level="unit_shuffle_threshold"),
        unit_shuffle_values.assign(summary_level="unit_shuffle_values"),
    )


def plot_shuffle_null(shuffle_summary):
    from matplotlib import pyplot as plt

    population_null = shuffle_summary[
        shuffle_summary["summary_level"] == "population_shuffle"
    ]
    unit_null = shuffle_summary[
        shuffle_summary["summary_level"] == "unit_shuffle_threshold"
    ]
    observed_median = float(population_null["observed_median_fsi"].iloc[0])

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4))
    sns.histplot(
        population_null["median_fsi"],
        bins=30,
        color="0.65",
        edgecolor="white",
        ax=axes[0],
    )
    axes[0].axvline(observed_median, color="red", linewidth=2)
    axes[0].set(xlabel="median FSI", ylabel="shuffle count")

    axes[1].scatter(
        unit_null["shuffle_fsi_p95"],
        unit_null["observed_fsi"],
        s=16,
        alpha=0.7,
        color="black",
        linewidths=0,
    )
    lim_max = float(
        np.nanmax([unit_null["shuffle_fsi_p95"].max(), unit_null["observed_fsi"].max()])
    )
    axes[1].plot([0, lim_max], [0, lim_max], color="red", linewidth=1.5, linestyle="--")
    axes[1].set(
        xlabel="shuffle 95th percentile FSI",
        ylabel="observed FSI",
        xlim=(0, lim_max * 1.03),
        ylim=(0, lim_max * 1.03),
    )
    for ax in axes:
        ax.grid(False)
    fig.tight_layout()
    return fig


def plot_observed_vs_shuffle_box(shuffle_summary):
    from matplotlib import pyplot as plt

    unit_null = shuffle_summary[
        shuffle_summary["summary_level"] == "unit_shuffle_threshold"
    ]
    unit_shuffle = shuffle_summary[
        shuffle_summary["summary_level"] == "unit_shuffle_values"
    ]
    plot_df = pd.concat(
        [
            unit_null[["observed_fsi"]]
            .rename(columns={"observed_fsi": "frequency_selectivity_index"})
            .assign(distribution="observed"),
            unit_shuffle[["frequency_selectivity_index"]].assign(
                distribution="shuffle"
            ),
        ],
        ignore_index=True,
    )

    fig, ax = plt.subplots(figsize=(3.2, 3.6))
    sns.boxplot(
        data=plot_df,
        x="distribution",
        y="frequency_selectivity_index",
        color="0.8",
        width=0.45,
        fliersize=1.0,
        ax=ax,
    )
    sns.stripplot(
        data=plot_df[plot_df["distribution"] == "observed"],
        x="distribution",
        y="frequency_selectivity_index",
        color="black",
        alpha=0.45,
        size=2.2,
        jitter=0.22,
        ax=ax,
    )
    ax.set(xlabel="", ylabel="FSI", ylim=(-0.02, 1.02))
    ax.grid(False)
    fig.tight_layout()
    return fig


def write_pdf(output_path: Path, figures) -> None:
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib import pyplot as plt

    with PdfPages(output_path) as pdf:
        for figure in figures:
            pdf.savefig(figure, dpi=300)
            plt.close(figure)


def main() -> None:
    args = parse_args()
    matplotlib.use("Agg")
    sns.set_theme(style="whitegrid", context="paper")
    _, trial_responses, _, unit_summary, _, _ = load_all_sessions()
    population_null, unit_null, unit_shuffle_values = build_shuffle_summary(
        trial_responses,
        unit_summary,
        n_shuffles=args.n_shuffles,
    )
    shuffle_summary = pd.concat(
        [
            population_null,
            unit_null,
            unit_shuffle_values,
        ],
        ignore_index=True,
        sort=False,
    )
    if args.no_save:
        print("\nBuilt shuffle-null outputs without writing files.")
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shuffle_summary.to_csv(
        args.output_dir / "rate_tuning_shuffle_summary.csv",
        index=False,
    )
    write_pdf(
        args.output_dir / "rate_tuning_shuffle_null.pdf",
        [
            plot_shuffle_null(shuffle_summary),
            plot_observed_vs_shuffle_box(shuffle_summary),
        ],
    )
    print(f"\nSaved outputs -> {args.output_dir}")


if __name__ == "__main__":
    main()
