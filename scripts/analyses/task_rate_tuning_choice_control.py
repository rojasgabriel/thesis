"""Choice-balanced and choice/light model controls for GRB006 rate tuning."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import pandas as pd
import seaborn as sns

from task_rate_tuning_shared import FIGURE_DIR, RANDOM_SEED, load_all_sessions
from ephys.src.utils.analysis_rate_tuning_choice import (
    balanced_choice_trial_responses,
    repeated_choice_balanced_fsi,
    summarize_resampled_units,
)
from ephys.src.utils.analysis_rate_tuning import aggregate_tuning_curves
from ephys.src.utils.analysis_rate_tuning_models import (
    fit_choice_light_encoding_models,
)

N_RESAMPLES = 200


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=FIGURE_DIR)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--n-resamples", type=int, default=N_RESAMPLES)
    return parser.parse_args()


def summarize_population_tuning(tuning_curves: pd.DataFrame) -> pd.DataFrame:
    return (
        tuning_curves.groupby("stim_rate_vision", as_index=False)
        .agg(mean_sp_s=("mean_sp_s", "mean"), sem_sp_s=("mean_sp_s", "sem"))
        .sort_values("stim_rate_vision")
    )


def summarize_balanced_population_tuning(
    trial_responses: pd.DataFrame,
    n_resamples: int,
) -> pd.DataFrame:
    rows = []
    for resample_idx in range(n_resamples):
        balanced = balanced_choice_trial_responses(
            trial_responses,
            class_column="stim_rate_vision",
            seed=RANDOM_SEED + resample_idx,
        )
        if balanced.empty:
            continue
        population = summarize_population_tuning(aggregate_tuning_curves(balanced))
        for row in population.itertuples(index=False):
            rows.append(
                {
                    "resample_idx": resample_idx,
                    "stim_rate_vision": float(row.stim_rate_vision),
                    "mean_sp_s": float(row.mean_sp_s),
                }
            )
    balanced_curves = pd.DataFrame(rows)
    if balanced_curves.empty:
        return balanced_curves
    return (
        balanced_curves.groupby("stim_rate_vision", as_index=False)
        .agg(
            mean_sp_s=("mean_sp_s", "median"),
            p025_sp_s=("mean_sp_s", lambda values: values.quantile(0.025)),
            p975_sp_s=("mean_sp_s", lambda values: values.quantile(0.975)),
        )
        .sort_values("stim_rate_vision")
    )


def build_choice_control_tables(trial_responses: pd.DataFrame, n_resamples: int):
    population_resamples, unit_resamples = repeated_choice_balanced_fsi(
        trial_responses,
        n_resamples=n_resamples,
        seed=RANDOM_SEED,
    )
    unit_summary = summarize_resampled_units(unit_resamples)
    balanced_population = summarize_balanced_population_tuning(
        trial_responses,
        n_resamples=n_resamples,
    )
    model_summary = fit_choice_light_encoding_models(
        trial_responses,
        seed=RANDOM_SEED,
    )
    return (
        population_resamples,
        unit_resamples,
        unit_summary,
        balanced_population,
        model_summary,
    )


def plot_choice_controls(
    trial_responses: pd.DataFrame,
    tuning_curves: pd.DataFrame,
    population_resamples: pd.DataFrame,
    unit_summary: pd.DataFrame,
    balanced_population: pd.DataFrame,
    model_summary: pd.DataFrame,
):
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.3))
    raw_population = summarize_population_tuning(tuning_curves)
    axes[0].errorbar(
        raw_population["stim_rate_vision"],
        raw_population["mean_sp_s"],
        yerr=raw_population["sem_sp_s"],
        color="black",
        marker="o",
    )
    axes[1].plot(
        balanced_population["stim_rate_vision"],
        balanced_population["mean_sp_s"],
        color="black",
        marker="o",
    )
    axes[1].fill_between(
        balanced_population["stim_rate_vision"],
        balanced_population["p025_sp_s"],
        balanced_population["p975_sp_s"],
        color="0.3",
        alpha=0.18,
        linewidth=0,
    )
    sns.pointplot(
        data=model_summary,
        x="regressor",
        y="unique_delta_cv_r2",
        errorbar=("se", 1),
        color="black",
        ax=axes[2],
    )
    axes[2].tick_params(axis="x", labelrotation=25)
    axes[2].axhline(0, color="0.4", linewidth=0.8, linestyle="--")
    axes[0].set(
        title="Raw tuning",
        xlabel="stimulus rate (Hz)",
        ylabel="mean sp/s",
    )
    axes[1].set(
        title="Choice-balanced tuning",
        xlabel="stimulus rate (Hz)",
        ylabel="mean sp/s",
    )
    axes[2].set(
        title="Unique model variance",
        xlabel="",
        ylabel="full - shuffled CV R2",
    )
    median_fsi = population_resamples["median_fsi"].median()
    p025 = population_resamples["median_fsi"].quantile(0.025)
    p975 = population_resamples["median_fsi"].quantile(0.975)
    axes[1].text(
        0.04,
        0.96,
        f"balanced FSI median\n{median_fsi:.2f} [{p025:.2f}, {p975:.2f}]",
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        fontsize=8,
    )
    axes[2].text(
        0.04,
        0.96,
        f"{len(unit_summary)} units",
        transform=axes[2].transAxes,
        ha="left",
        va="top",
        fontsize=8,
    )
    for ax in axes:
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
    _, trial_responses, tuning_curves, _, _, _ = load_all_sessions()
    (
        population_resamples,
        unit_resamples,
        unit_summary,
        balanced_population,
        model_summary,
    ) = build_choice_control_tables(trial_responses, args.n_resamples)
    if args.no_save:
        print("\nBuilt choice-control outputs without writing files.")
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(
        [
            population_resamples.assign(summary_level="population_resample"),
            unit_resamples.assign(summary_level="unit_resample"),
            unit_summary.assign(summary_level="unit_interval"),
            balanced_population.assign(summary_level="balanced_population_tuning"),
        ],
        ignore_index=True,
        sort=False,
    ).to_csv(args.output_dir / "rate_tuning_choice_balanced_summary.csv", index=False)
    model_summary.to_csv(
        args.output_dir / "unit_encoding_model_choice_light_summary.csv",
        index=False,
    )
    write_pdf(
        args.output_dir / "rate_tuning_choice_control.pdf",
        plot_choice_controls(
            trial_responses,
            tuning_curves,
            population_resamples,
            unit_summary,
            balanced_population,
            model_summary,
        ),
    )
    print(f"\nSaved outputs -> {args.output_dir}")


if __name__ == "__main__":
    main()
