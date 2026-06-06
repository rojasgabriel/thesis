"""Light-dose controls for task rate tuning."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import seaborn as sns

from task_rate_tuning_shared import FIGURE_DIR, RANDOM_SEED, load_all_sessions
from ephys.src.utils.analysis_rate_tuning_light import residualize_by_unit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=FIGURE_DIR)
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def residual_tuning(trial_responses):
    residualized = residualize_by_unit(
        trial_responses,
        response_column="response_sp_s",
        predictor_column="total_light_time_s",
    )
    return (
        residualized.groupby(["subject", "session", "unit_id", "stim_rate_vision"])
        .agg(mean_residual_sp_s=("residual_response_sp_s", "mean"))
        .reset_index()
    )


def plot_light_diagnostics(trial_responses, tuning_curves, residual_tuning_df):
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.2))
    spikes_per_flash = trial_responses.groupby(
        ["unit_id", "stim_rate_vision"], as_index=False
    ).agg(mean_spikes_per_flash=("spikes_per_flash", "mean"))
    response_plot = tuning_curves.groupby("stim_rate_vision", as_index=False).agg(
        mean_sp_s=("mean_sp_s", "mean"),
        sem_sp_s=("mean_sp_s", "sem"),
    )
    flash_plot = spikes_per_flash.groupby("stim_rate_vision", as_index=False).agg(
        mean_spikes_per_flash=("mean_spikes_per_flash", "mean"),
        sem_spikes_per_flash=("mean_spikes_per_flash", "sem"),
    )
    axes[0].errorbar(
        response_plot["stim_rate_vision"],
        response_plot["mean_sp_s"],
        yerr=response_plot["sem_sp_s"],
        marker="o",
        color="black",
    )
    twin = axes[0].twinx()
    twin.errorbar(
        flash_plot["stim_rate_vision"],
        flash_plot["mean_spikes_per_flash"],
        yerr=flash_plot["sem_spikes_per_flash"],
        marker="o",
        color="red",
    )
    axes[0].set(xlabel="stimulus rate (Hz)", ylabel="mean sp/s")
    twin.set(ylabel="mean spikes/flash")
    axes[0].grid(False)
    twin.grid(False)

    sample_df = trial_responses.sample(
        n=min(5000, len(trial_responses)),
        random_state=RANDOM_SEED,
    )
    sns.scatterplot(
        data=sample_df,
        x="total_light_time_s",
        y="response_sp_s",
        hue="stim_rate_vision",
        palette="viridis",
        s=8,
        alpha=0.25,
        linewidth=0,
        legend=False,
        ax=axes[1],
    )
    axes[1].set(xlabel="total light time (s)", ylabel="sp/s")

    residual_plot = residual_tuning_df.groupby("stim_rate_vision", as_index=False).agg(
        mean_residual_sp_s=("mean_residual_sp_s", "mean"),
        sem_residual_sp_s=("mean_residual_sp_s", "sem"),
    )
    axes[2].errorbar(
        residual_plot["stim_rate_vision"],
        residual_plot["mean_residual_sp_s"],
        yerr=residual_plot["sem_residual_sp_s"],
        marker="o",
        color="black",
    )
    axes[2].axhline(0, color="0.4", linewidth=0.8, linestyle="--")
    axes[2].set(xlabel="stimulus rate (Hz)", ylabel="residual sp/s")
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
    _, trial_responses, tuning_curves, _, light_exposure, _ = load_all_sessions()
    residual_tuning_df = residual_tuning(trial_responses)
    if args.no_save:
        print("\nBuilt light-control outputs without writing files.")
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    light_exposure.to_csv(args.output_dir / "trial_light_exposure.csv", index=False)
    write_pdf(
        args.output_dir / "rate_tuning_light_control.pdf",
        plot_light_diagnostics(trial_responses, tuning_curves, residual_tuning_df),
    )
    print(f"\nSaved outputs -> {args.output_dir}")


if __name__ == "__main__":
    main()
