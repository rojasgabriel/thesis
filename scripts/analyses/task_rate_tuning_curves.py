"""Descriptive task stimulus-period rate tuning curves."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

from task_rate_tuning_shared import FIGURE_DIR, load_all_sessions, pivot_tuning


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=FIGURE_DIR,
        help="Directory for CSV and PDF outputs.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Build tables and figures without writing outputs.",
    )
    return parser.parse_args()


def zscore_heatmap_values(
    session_tuning: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    displayed_rates = np.asarray(sorted(session_tuning["stim_rate_vision"].unique()))
    matrix = pivot_tuning(session_tuning, rates=displayed_rates)
    row_mean = matrix.mean(axis=1)
    row_std = matrix.std(axis=1).replace(0, np.nan)
    values = matrix.sub(row_mean, axis=0).div(row_std, axis=0)

    preferred_rate = matrix.idxmax(axis=1).fillna(np.inf)
    row_mean_for_sort = matrix.mean(axis=1).fillna(0.0)
    sort_index = (
        pd.DataFrame({"preferred_rate": preferred_rate, "row_mean": row_mean_for_sort})
        .sort_values(["preferred_rate", "row_mean"], ascending=[True, False])
        .index
    )
    return values.loc[sort_index], displayed_rates


def choose_example_curve_units(
    unit_summary: pd.DataFrame,
    n_examples: int = 6,
) -> pd.DataFrame:
    if unit_summary.empty:
        return unit_summary.copy()

    summary = unit_summary.sort_values(
        ["tuning_range_sp_s", "mean_sp_s_all_rates"],
        ascending=[False, False],
    )
    selected_rows = []
    selected_rates = set()
    for _, row in summary.iterrows():
        preferred_rate = float(row["preferred_stim_rate"])
        if preferred_rate in selected_rates:
            continue
        selected_rows.append(row)
        selected_rates.add(preferred_rate)
        if len(selected_rows) == n_examples:
            break

    if len(selected_rows) < n_examples:
        selected_keys = {
            (row["subject"], row["session"], int(row["unit_id"]))
            for row in selected_rows
        }
        for _, row in summary.iterrows():
            key = (row["subject"], row["session"], int(row["unit_id"]))
            if key in selected_keys:
                continue
            selected_rows.append(row)
            selected_keys.add(key)
            if len(selected_rows) == n_examples:
                break

    return pd.DataFrame(selected_rows).reset_index(drop=True)


def plot_heatmap(tuning_curves: pd.DataFrame):
    from matplotlib import pyplot as plt

    session_groups = list(tuning_curves.groupby(["subject", "session"], sort=False))
    if len(session_groups) != 1:
        raise RuntimeError("The heatmap-only figure expects exactly one session.")

    (_subject, _session), session_tuning = session_groups[0]
    values, displayed_rates = zscore_heatmap_values(session_tuning)
    fig, ax = plt.subplots(figsize=(7.0, 8.0))
    sns.heatmap(
        values,
        ax=ax,
        cmap="vlag",
        center=0,
        cbar_kws={"label": "z-score within unit"},
        xticklabels=[str(int(rate)) for rate in displayed_rates],
        yticklabels=False,
    )
    ax.set(
        xlabel="stimulus rate (Hz)",
        ylabel="units sorted by preferred rate",
    )
    fig.tight_layout()
    return fig


def plot_example_curves(tuning_curves: pd.DataFrame, unit_summary: pd.DataFrame):
    from matplotlib import pyplot as plt

    examples = choose_example_curve_units(unit_summary, n_examples=6)
    if examples.empty:
        raise RuntimeError("No units available for example curves.")

    rates = np.asarray(sorted(tuning_curves["stim_rate_vision"].unique()))
    curve_rows = []
    for row in examples.itertuples(index=False):
        unit_df = tuning_curves[
            (tuning_curves["subject"] == row.subject)
            & (tuning_curves["session"] == row.session)
            & (tuning_curves["unit_id"] == row.unit_id)
        ].sort_values("stim_rate_vision")
        values = (
            unit_df.set_index("stim_rate_vision")
            .reindex(rates)["mean_sp_s"]
            .to_numpy(dtype=float)
        )
        z_values = (values - np.nanmean(values)) / np.nanstd(values, ddof=1)
        curve_rows.append(z_values)

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 7.0), sharex=True, sharey=True)
    color = sns.color_palette("Set1", n_colors=1)[0]
    for ax, z_values in zip(axes.ravel(), curve_rows, strict=False):
        ax.plot(rates, z_values, color=color, marker="o", linewidth=1.5)
        ax.axhline(0, color="0.4", linewidth=0.8, linestyle="--")
        ax.grid(False)
        ax.set_xticks(rates)
        ax.tick_params(axis="x", labelrotation=45, labelsize=7)

    for ax in axes[:, 0]:
        ax.set_ylabel("z-score")
    for ax in axes[-1, :]:
        ax.set_xlabel("stimulus rate (Hz)")
    fig.tight_layout()
    return fig


def plot_rate_response_scatter(tuning_curves: pd.DataFrame):
    from matplotlib import pyplot as plt

    scatter_df = tuning_curves.copy()
    scatter_df["z_mean_sp_s"] = scatter_df.groupby("unit_id")["mean_sp_s"].transform(
        lambda values: (values - values.mean()) / values.std(ddof=1)
    )
    x = scatter_df["stim_rate_vision"].to_numpy(dtype=float)
    y = scatter_df["z_mean_sp_s"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, deg=1)
    correlation = np.corrcoef(x, y)[0, 1]
    fit_x = np.linspace(float(x.min()), float(x.max()), 200)
    fit_y = slope * fit_x + intercept

    fig, ax = plt.subplots(figsize=(4.0, 4.0))
    line_color = sns.color_palette("Set1", n_colors=1)[0]
    ax.scatter(x, y, s=14, alpha=0.18, color="black", linewidths=0)
    ax.plot(fit_x, fit_y, color=line_color, linewidth=2, linestyle="--")
    ax.grid(False)
    ax.text(
        0.98,
        0.96,
        f"r = {correlation:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
    )
    ax.set(
        xlabel="stimulus rate (Hz)",
        ylabel="z-score",
        xticks=sorted(tuning_curves["stim_rate_vision"].unique()),
    )
    fig.tight_layout()
    return fig


def plot_frequency_selectivity_box(unit_summary: pd.DataFrame):
    from matplotlib import pyplot as plt

    plot_df = unit_summary.dropna(subset=["frequency_selectivity_index"]).copy()
    if plot_df.empty:
        raise RuntimeError("No finite FSI values available for box plot.")

    plot_df["population"] = (
        plot_df["subject"].astype(str) + "\n" + plot_df["session"].astype(str)
    )
    fig, ax = plt.subplots(figsize=(1.9, 3.9))
    sns.boxplot(
        data=plot_df,
        x="population",
        y="frequency_selectivity_index",
        color="0.8",
        width=0.45,
        linewidth=1.0,
        fliersize=2.0,
        ax=ax,
    )
    ax.set(xlabel="population", ylabel="FSI", ylim=(-0.02, 1.02))
    ax.grid(False)
    fig.tight_layout()
    return fig


def write_one_page_pdf(output_path: Path, figure) -> None:
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib import pyplot as plt

    with PdfPages(output_path) as pdf:
        pdf.savefig(figure, dpi=300)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    matplotlib.use("Agg")
    sns.set_theme(style="whitegrid", context="paper")

    (
        _windows_df,
        trial_responses_df,
        tuning_curves_df,
        unit_summary_df,
        _light_exposure_df,
        _session_payloads,
    ) = load_all_sessions()

    if args.no_save:
        print(
            "\nBuilt descriptive rate tuning tables and figures without writing outputs."
        )
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trial_responses_df.to_csv(args.output_dir / "trial_responses.csv", index=False)
    tuning_curves_df.to_csv(args.output_dir / "unit_tuning_curves.csv", index=False)
    unit_summary_df.to_csv(args.output_dir / "unit_summary.csv", index=False)
    write_one_page_pdf(
        args.output_dir / "rate_tuning_curves.pdf",
        plot_heatmap(tuning_curves_df),
    )
    write_one_page_pdf(
        args.output_dir / "rate_tuning_example_curves.pdf",
        plot_example_curves(tuning_curves_df, unit_summary_df),
    )
    write_one_page_pdf(
        args.output_dir / "rate_tuning_rate_response_scatter.pdf",
        plot_rate_response_scatter(tuning_curves_df),
    )
    write_one_page_pdf(
        args.output_dir / "frequency_selectivity_index_box.pdf",
        plot_frequency_selectivity_box(unit_summary_df),
    )
    print(f"\nSaved outputs -> {args.output_dir}")


if __name__ == "__main__":
    main()
