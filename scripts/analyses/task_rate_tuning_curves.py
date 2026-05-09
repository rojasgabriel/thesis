"""First-pass task stimulus-period rate tuning curves.

For each good unit, this analysis measures mean firing rate from the first
15 ms visual flash in a trial to response-port entry, then groups responses
by visual stimulus rate.
"""

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

from ephys.src.utils.analysis_rate_tuning import (  # noqa: E402
    add_light_exposure_to_responses,
    add_trial_predictors,
    aggregate_tuning_curves,
    build_task_stimulus_windows,
    compute_light_exposure,
    compute_timecourse_responses,
    compute_trial_responses,
    fit_encoding_models,
    residualize_by_unit,
    shuffle_fsi_null,
    summarize_units,
    summarize_timecourse_encoding,
)
from ephys.src.utils.io_chipmunk_trials import fetch_trial_metadata  # noqa: E402
from ephys.src.utils.io_digital_events import fetch_session_events  # noqa: E402
from ephys.src.utils.io_session_units import fetch_good_units  # noqa: E402

FIGURE_ROOT = Path(
    os.environ.get("EPHYS_FIGURE_ROOT", "/Users/gabriel/lib/ephys/figures")
)
FIGURE_DIR = FIGURE_ROOT / "task_rate_tuning"

SUBJECT_SESSIONS = [
    ("GRB006", "20240821_121447"),
]

UNIT_CRITERIA_ID = 1
N_SHUFFLES = 1000
RANDOM_SEED = 0
TIMECOURSE_BIN_EDGES_S = np.arange(0.0, 1.0 + 0.1, 0.1)


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
    parser.add_argument(
        "--n-shuffles",
        type=int,
        default=N_SHUFFLES,
        help="Number of stimulus-rate label shuffles for the FSI null.",
    )
    return parser.parse_args()


def load_session_tables(subject: str, session: str) -> tuple:
    print(f"\nLoading {subject} {session}")
    align_ev = fetch_session_events(subject, session)
    trial_df = fetch_trial_metadata(subject, session, align_ev)
    if trial_df is None:
        raise RuntimeError(f"Could not load Chipmunk trials for {subject} {session}.")
    spike_times_by_unit = fetch_good_units(
        subject,
        session,
        unit_criteria_id=UNIT_CRITERIA_ID,
    )
    windows = build_task_stimulus_windows(align_ev, trial_df)
    if windows.empty:
        raise RuntimeError(f"No valid task stimulus windows for {subject} {session}.")
    trial_responses = compute_trial_responses(windows, spike_times_by_unit)
    tuning_curves = aggregate_tuning_curves(trial_responses)
    unit_summary = summarize_units(tuning_curves)
    light_exposure = compute_light_exposure(windows, align_ev["stim_ev_15ms"])
    trial_responses = add_light_exposure_to_responses(trial_responses, light_exposure)
    trial_responses = add_trial_predictors(trial_responses)
    timecourse_responses = compute_timecourse_responses(
        windows,
        spike_times_by_unit,
        TIMECOURSE_BIN_EDGES_S,
    )

    for table in (
        windows,
        trial_responses,
        tuning_curves,
        unit_summary,
        light_exposure,
        timecourse_responses,
    ):
        table.insert(0, "session", session)
        table.insert(0, "subject", subject)

    print(f"  Units: {len(spike_times_by_unit)}")
    print(f"  Valid trials: {len(windows)}")
    print(
        "  Rates: "
        + ", ".join(
            str(int(rate)) for rate in sorted(windows["stim_rate_vision"].unique())
        )
    )
    return (
        windows,
        trial_responses,
        tuning_curves,
        unit_summary,
        light_exposure,
        timecourse_responses,
    )


def pivot_tuning(
    tuning_curves: pd.DataFrame,
    value_column: str = "mean_sp_s",
    rates: np.ndarray | None = None,
) -> pd.DataFrame:
    pivot = tuning_curves.pivot_table(
        index=["subject", "session", "unit_id"],
        columns="stim_rate_vision",
        values=value_column,
    )
    if rates is None:
        rates = np.asarray(sorted(tuning_curves["stim_rate_vision"].unique()))
    return pivot.reindex(columns=rates)


def zscore_heatmap_values(
    session_tuning: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Row-wise z-score of mean sp/s tuning, sorted by preferred rate."""
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
    """Pick high-modulation examples, spreading preferred rates when possible."""
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


def plot_example_curves(
    tuning_curves: pd.DataFrame,
    unit_summary: pd.DataFrame,
):
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
        value_mean = np.nanmean(values)
        value_std = np.nanstd(values, ddof=1)
        z_values = (values - value_mean) / value_std
        curve_rows.append((row, z_values))

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(10.5, 7.0),
        sharex=True,
        sharey=True,
    )
    color = sns.color_palette("Set1", n_colors=1)[0]
    for ax, (row, z_values) in zip(axes.ravel(), curve_rows, strict=False):
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


def write_pdf(
    output_path: Path,
    windows: pd.DataFrame,
    tuning_curves: pd.DataFrame,
    unit_summary: pd.DataFrame,
) -> None:
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib import pyplot as plt

    del windows, unit_summary
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

    with PdfPages(output_path) as pdf:
        pdf.savefig(fig, dpi=300)
    plt.close(fig)


def write_example_curves_pdf(
    output_path: Path,
    tuning_curves: pd.DataFrame,
    unit_summary: pd.DataFrame,
) -> None:
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib import pyplot as plt

    fig = plot_example_curves(tuning_curves, unit_summary)
    with PdfPages(output_path) as pdf:
        pdf.savefig(fig, dpi=300)
    plt.close(fig)


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


def write_rate_scatter_pdf(output_path: Path, tuning_curves: pd.DataFrame) -> None:
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib import pyplot as plt

    fig = plot_rate_response_scatter(tuning_curves)
    with PdfPages(output_path) as pdf:
        pdf.savefig(fig, dpi=300)
    plt.close(fig)


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
    ax.set(
        xlabel="population",
        ylabel="FSI",
        ylim=(-0.02, 1.02),
    )
    ax.grid(False)
    fig.tight_layout()
    return fig


def write_frequency_selectivity_box_pdf(
    output_path: Path,
    unit_summary: pd.DataFrame,
) -> None:
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib import pyplot as plt

    fig = plot_frequency_selectivity_box(unit_summary)
    with PdfPages(output_path) as pdf:
        pdf.savefig(fig, dpi=300)
    plt.close(fig)


def build_followup_tables(
    trial_responses: pd.DataFrame,
    unit_summary: pd.DataFrame,
    timecourse_responses: pd.DataFrame,
    n_shuffles: int,
) -> dict[str, pd.DataFrame]:
    population_null, unit_null = shuffle_fsi_null(
        trial_responses,
        unit_summary,
        n_shuffles=n_shuffles,
        seed=RANDOM_SEED,
    )
    observed_median = unit_summary["frequency_selectivity_index"].median()
    observed_mean = unit_summary["frequency_selectivity_index"].mean()
    population_null["observed_median_fsi"] = observed_median
    population_null["observed_mean_fsi"] = observed_mean
    unit_null["observed_population_median_fsi"] = observed_median
    unit_null["observed_population_mean_fsi"] = observed_mean
    shuffle_summary = pd.concat(
        [
            population_null.assign(summary_level="population_shuffle"),
            unit_null.assign(summary_level="unit_shuffle_threshold"),
        ],
        ignore_index=True,
        sort=False,
    )

    model_summary = fit_encoding_models(trial_responses)
    residualized = residualize_by_unit(
        trial_responses,
        response_column="response_sp_s",
        predictor_column="total_light_time_s",
    )
    residual_tuning = (
        residualized.groupby(["subject", "session", "unit_id", "stim_rate_vision"])
        .agg(mean_residual_sp_s=("residual_response_sp_s", "mean"))
        .reset_index()
    )
    timecourse_summary, timecourse_coefficients = summarize_timecourse_encoding(
        timecourse_responses
    )
    for table in (model_summary, timecourse_summary, timecourse_coefficients):
        table.insert(0, "session", trial_responses["session"].iloc[0])
        table.insert(0, "subject", trial_responses["subject"].iloc[0])
    timecourse_encoding = pd.concat(
        [
            timecourse_summary.assign(summary_level="population_timecourse"),
            timecourse_coefficients.assign(summary_level="unit_timecourse"),
        ],
        ignore_index=True,
        sort=False,
    )
    return {
        "shuffle_summary": shuffle_summary,
        "model_summary": model_summary,
        "residual_tuning": residual_tuning,
        "timecourse_encoding": timecourse_encoding,
        "timecourse_summary": timecourse_summary,
        "timecourse_coefficients": timecourse_coefficients,
    }


def plot_shuffle_null(shuffle_summary: pd.DataFrame):
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


def plot_model_comparison(model_summary: pd.DataFrame):
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


def plot_low_high_tuning_shape(tuning_curves: pd.DataFrame):
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


def plot_light_diagnostics(
    trial_responses: pd.DataFrame,
    tuning_curves: pd.DataFrame,
    residual_tuning: pd.DataFrame,
):
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
        label="sp/s",
    )
    twin = axes[0].twinx()
    twin.errorbar(
        flash_plot["stim_rate_vision"],
        flash_plot["mean_spikes_per_flash"],
        yerr=flash_plot["sem_spikes_per_flash"],
        marker="o",
        color="red",
        label="spikes/flash",
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

    residual_plot = residual_tuning.groupby("stim_rate_vision", as_index=False).agg(
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


def plot_timecourse_encoding(
    timecourse_summary: pd.DataFrame,
    timecourse_coefficients: pd.DataFrame,
):
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


def write_followup_controls_pdf(
    output_path: Path,
    trial_responses: pd.DataFrame,
    tuning_curves: pd.DataFrame,
    followup_tables: dict[str, pd.DataFrame],
) -> None:
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib import pyplot as plt

    figures = [
        plot_shuffle_null(followup_tables["shuffle_summary"]),
        plot_model_comparison(followup_tables["model_summary"]),
        plot_low_high_tuning_shape(tuning_curves),
        plot_light_diagnostics(
            trial_responses,
            tuning_curves,
            followup_tables["residual_tuning"],
        ),
        plot_timecourse_encoding(
            followup_tables["timecourse_summary"],
            followup_tables["timecourse_coefficients"],
        ),
    ]
    with PdfPages(output_path) as pdf:
        for fig in figures:
            pdf.savefig(fig, dpi=300)
            plt.close(fig)


def main() -> None:
    args = parse_args()
    matplotlib.use("Agg")
    sns.set_theme(style="whitegrid", context="paper")

    all_windows = []
    all_trial_responses = []
    all_tuning_curves = []
    all_unit_summaries = []
    all_light_exposures = []
    all_timecourse_responses = []
    for subject, session in SUBJECT_SESSIONS:
        (
            windows,
            trial_responses,
            tuning_curves,
            unit_summary,
            light_exposure,
            timecourse_responses,
        ) = load_session_tables(subject, session)
        all_windows.append(windows)
        all_trial_responses.append(trial_responses)
        all_tuning_curves.append(tuning_curves)
        all_unit_summaries.append(unit_summary)
        all_light_exposures.append(light_exposure)
        all_timecourse_responses.append(timecourse_responses)

    windows_df = pd.concat(all_windows, ignore_index=True)
    trial_responses_df = pd.concat(all_trial_responses, ignore_index=True)
    tuning_curves_df = pd.concat(all_tuning_curves, ignore_index=True)
    unit_summary_df = pd.concat(all_unit_summaries, ignore_index=True)
    light_exposure_df = pd.concat(all_light_exposures, ignore_index=True)
    timecourse_responses_df = pd.concat(all_timecourse_responses, ignore_index=True)
    followup_tables = build_followup_tables(
        trial_responses_df,
        unit_summary_df,
        timecourse_responses_df,
        n_shuffles=args.n_shuffles,
    )

    if args.no_save:
        print("\nBuilt rate tuning tables and figures without writing outputs.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trial_responses_df.to_csv(args.output_dir / "trial_responses.csv", index=False)
    tuning_curves_df.to_csv(args.output_dir / "unit_tuning_curves.csv", index=False)
    unit_summary_df.to_csv(args.output_dir / "unit_summary.csv", index=False)
    light_exposure_df.to_csv(args.output_dir / "trial_light_exposure.csv", index=False)
    followup_tables["shuffle_summary"].to_csv(
        args.output_dir / "rate_tuning_shuffle_summary.csv",
        index=False,
    )
    followup_tables["model_summary"].to_csv(
        args.output_dir / "unit_encoding_model_summary.csv",
        index=False,
    )
    followup_tables["timecourse_encoding"].to_csv(
        args.output_dir / "timecourse_encoding_summary.csv",
        index=False,
    )
    pdf_path = args.output_dir / "rate_tuning_curves.pdf"
    write_pdf(pdf_path, windows_df, tuning_curves_df, unit_summary_df)
    example_pdf_path = args.output_dir / "rate_tuning_example_curves.pdf"
    write_example_curves_pdf(example_pdf_path, tuning_curves_df, unit_summary_df)
    scatter_pdf_path = args.output_dir / "rate_tuning_rate_response_scatter.pdf"
    write_rate_scatter_pdf(scatter_pdf_path, tuning_curves_df)
    fsi_pdf_path = args.output_dir / "frequency_selectivity_index_box.pdf"
    write_frequency_selectivity_box_pdf(fsi_pdf_path, unit_summary_df)
    followup_pdf_path = args.output_dir / "rate_tuning_followup_controls.pdf"
    write_followup_controls_pdf(
        followup_pdf_path,
        trial_responses_df,
        tuning_curves_df,
        followup_tables,
    )
    print(f"\nSaved outputs -> {args.output_dir}")


if __name__ == "__main__":
    main()
