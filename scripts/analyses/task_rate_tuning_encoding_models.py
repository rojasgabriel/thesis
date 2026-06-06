"""Evidence, category, and choice model controls for task rate tuning."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import seaborn as sns

from task_rate_tuning_shared import FIGURE_DIR, load_all_sessions
from ephys.src.utils.analysis_rate_tuning_models import fit_encoding_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=FIGURE_DIR)
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def plot_model_comparison(model_summary):
    from matplotlib import pyplot as plt

    plot_df = model_summary[model_summary["model"] != "baseline"].copy()
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.2))
    sns.pointplot(
        data=plot_df,
        x="model",
        y="delta_cv_r2",
        errorbar=("se", 1),
        color="black",
        ax=axes[0],
    )
    axes[0].axhline(0, color="0.4", linewidth=0.8, linestyle="--")
    axes[0].tick_params(axis="x", labelrotation=35)
    axes[0].set(xlabel="", ylabel="delta CV R2")

    wide = model_summary.pivot_table(
        index="unit_id", columns="model", values="delta_cv_r2"
    )
    axes[1].scatter(
        wide["signed_evidence"],
        wide["category"],
        s=16,
        alpha=0.7,
        color="black",
        linewidths=0,
    )
    axes[1].axline((0, 0), slope=1, color="red", linewidth=1.0, linestyle="--")
    axes[1].set(xlabel="evidence delta CV R2", ylabel="category delta CV R2")

    axes[2].scatter(
        wide["category"],
        wide["choice"],
        s=16,
        alpha=0.7,
        color="black",
        linewidths=0,
    )
    axes[2].axline((0, 0), slope=1, color="red", linewidth=1.0, linestyle="--")
    axes[2].set(xlabel="category delta CV R2", ylabel="choice delta CV R2")
    for ax in axes:
        ax.grid(False)
    fig.tight_layout()
    return fig


def plot_low_high_tuning_shape(tuning_curves):
    from matplotlib import pyplot as plt

    plot_df = tuning_curves.copy()
    plot_df["z_mean_sp_s"] = plot_df.groupby("unit_id")["mean_sp_s"].transform(
        lambda values: (values - values.mean()) / values.std(ddof=1)
    )
    plot_df["rate_side"] = np.select(
        [plot_df["stim_rate_vision"] < 12, plot_df["stim_rate_vision"] > 12],
        ["low-rate side", "high-rate side"],
        default="boundary",
    )
    plot_df = plot_df[plot_df["rate_side"] != "boundary"]

    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    sns.lineplot(
        data=plot_df,
        x="stim_rate_vision",
        y="z_mean_sp_s",
        hue="rate_side",
        marker="o",
        errorbar=("se", 1),
        ax=ax,
    )
    ax.axhline(0, color="0.4", linewidth=0.8, linestyle="--")
    ax.set(xlabel="stimulus rate (Hz)", ylabel="z-score")
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
    _, trial_responses, tuning_curves, _, _, _ = load_all_sessions()
    model_summary = fit_encoding_models(trial_responses)
    if args.no_save:
        print("\nBuilt encoding-model outputs without writing files.")
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_summary.to_csv(
        args.output_dir / "unit_encoding_model_summary.csv",
        index=False,
    )
    write_pdf(
        args.output_dir / "rate_tuning_encoding_models.pdf",
        [
            plot_model_comparison(model_summary),
            plot_low_high_tuning_shape(tuning_curves),
        ],
    )
    print(f"\nSaved outputs -> {args.output_dir}")


if __name__ == "__main__":
    main()
