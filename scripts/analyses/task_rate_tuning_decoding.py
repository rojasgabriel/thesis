"""Choice-balanced logistic-regression decoding for GRB006 rate tuning."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import pandas as pd
import seaborn as sns

from task_rate_tuning_shared import FIGURE_DIR, RANDOM_SEED, load_all_sessions
from ephys.src.utils.analysis_rate_tuning_decoding import decode_target

N_RESAMPLES = 200


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=FIGURE_DIR)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--n-resamples", type=int, default=N_RESAMPLES)
    return parser.parse_args()


def build_decoding_summary(trial_responses: pd.DataFrame, n_resamples: int):
    category = decode_target(
        trial_responses,
        target="category",
        n_resamples=n_resamples,
        seed=RANDOM_SEED,
    )
    rate = decode_target(
        trial_responses,
        target="rate",
        n_resamples=n_resamples,
        seed=RANDOM_SEED + 1_000,
    )
    return pd.concat([category, rate], ignore_index=True)


def plot_decoding(decoding_summary: pd.DataFrame):
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2))
    sns.boxplot(
        data=decoding_summary,
        x="target",
        y="balanced_accuracy",
        hue="condition",
        fliersize=0,
        ax=axes[0],
    )
    sns.stripplot(
        data=decoding_summary,
        x="target",
        y="balanced_accuracy",
        hue="condition",
        dodge=True,
        alpha=0.25,
        size=2,
        legend=False,
        ax=axes[0],
    )
    rate_rows = decoding_summary[decoding_summary["target"] == "rate"]
    sns.boxplot(
        data=rate_rows,
        x="condition",
        y="rate_correlation",
        fliersize=0,
        ax=axes[1],
    )
    sns.stripplot(
        data=rate_rows,
        x="condition",
        y="rate_correlation",
        alpha=0.25,
        size=2,
        ax=axes[1],
    )
    axes[0].axhline(0.5, color="0.4", linewidth=0.8, linestyle="--")
    axes[0].set(xlabel="", ylabel="balanced accuracy")
    axes[1].axhline(0, color="0.4", linewidth=0.8, linestyle="--")
    axes[1].set(xlabel="", ylabel="predicted-vs-true rate r")
    axes[1].tick_params(axis="x", labelrotation=25)
    axes[0].legend(frameon=False, fontsize=8)
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
    _, trial_responses, _, _, _, _ = load_all_sessions()
    decoding_summary = build_decoding_summary(trial_responses, args.n_resamples)
    if args.no_save:
        print("\nBuilt logistic-decoding outputs without writing files.")
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    decoding_summary.to_csv(
        args.output_dir / "rate_tuning_decoding_summary.csv",
        index=False,
    )
    write_pdf(
        args.output_dir / "rate_tuning_decoding.pdf",
        plot_decoding(decoding_summary),
    )
    print(f"\nSaved outputs -> {args.output_dir}")


if __name__ == "__main__":
    main()
