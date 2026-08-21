"""Archived adaptation summary for GRB006 20240821_121447.

This is a local-data analysis script, not part of the main figure surface. It
summarizes how peak visual responses change across the first four
stationary flashes and the first movement flash, with an additional split by
response side.
"""

import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from thesis.ephys.config.typing_params import PopulationPethParams
from thesis.ephys.utils.analysis_peth import compute_population_peth
from thesis.ephys.utils.grb006_data import (
    fetch_grb006_spike_times,
    load_grb006_aligned_trial_data,
)

matplotlib.use("Agg")

SUBJECT = "GRB006"
SESSION = "20240821_121447"
FIGURE_ROOT = Path(os.environ.get("THESIS_FIGURE_ROOT", "figures"))
FIGURE_DIR = FIGURE_ROOT / "adaptation"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

N_STATIONARY_STIMS = 4
STIM_LABELS = [
    "peak_first_flash",
    "peak_second_flash",
    "peak_third_flash",
    "peak_fourth_flash",
    "peak_movement_flash",
]
PETH_KWARGS: PopulationPethParams = dict(
    pre_seconds=0.025,
    post_seconds=0.175,
    binwidth_ms=25,
    t_rise=0.001,
    t_decay=0.025,
)
RESP_WINDOW = (0.04, 0.10)


def load_inputs() -> tuple[pd.DataFrame, list[np.ndarray]]:
    _, trial_ts = load_grb006_aligned_trial_data()
    _, spike_times = fetch_grb006_spike_times()
    return trial_ts, spike_times


def filter_trials(trial_ts: pd.DataFrame) -> pd.DataFrame:
    return trial_ts[
        trial_ts["stationary_stims"].apply(lambda x: len(x) >= N_STATIONARY_STIMS)
        & trial_ts["movement_stims"].apply(lambda x: len(x) > 0)
        & trial_ts["center_port_entries"].apply(lambda x: len(x) > 0)
    ].copy()


def extract_stimuli(trial_ts: pd.DataFrame) -> list[np.ndarray]:
    stationary = [
        np.array([stims[i] for stims in trial_ts["stationary_stims"]], dtype=float)
        for i in range(N_STATIONARY_STIMS)
    ]
    movement = np.array(
        [stims[0] for stims in trial_ts["movement_stims"]],
        dtype=float,
    )
    return stationary + [movement]


def peak_matrix_for_stims(
    spike_times_per_unit: list[np.ndarray],
    stimuli: list[np.ndarray],
) -> pd.DataFrame:
    peak_responses = []
    for stim_times in stimuli:
        peth, bin_edges, _ = compute_population_peth(
            spike_times_per_unit=spike_times_per_unit,
            alignment_times=stim_times,
            **PETH_KWARGS,
        )
        stim_mask = (bin_edges[:-1] >= RESP_WINDOW[0]) & (
            bin_edges[:-1] <= RESP_WINDOW[1]
        )
        peak_response = np.max(np.mean(peth[:, :, stim_mask], axis=1), axis=1)
        peak_responses.append(peak_response)
    mat = np.stack(peak_responses, axis=1)
    return pd.DataFrame({lbl: mat[:, i] for i, lbl in enumerate(STIM_LABELS)})


def plot_population_points(
    ax, values: np.ndarray, color: str, alpha: float, markersize: float
) -> None:
    x = np.arange(values.shape[1])
    for row in values:
        ax.plot(x, row, "o", color=color, alpha=alpha, markersize=markersize)


def make_mean_plot(peak_subtracted_df: pd.DataFrame) -> plt.Figure:
    x = np.arange(len(STIM_LABELS))
    fig, ax = plt.subplots(1, figsize=(5, 5))
    plot_population_points(ax, peak_subtracted_df.to_numpy(), "lightgray", 0.3, 3)
    ax.errorbar(
        x,
        peak_subtracted_df.mean(),
        yerr=peak_subtracted_df.sem(),
        fmt="o-",
        color="red",
        linewidth=2,
        markersize=8,
        capsize=5,
        label=f"mean ± SEM\n{len(peak_subtracted_df)} units",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(STIM_LABELS, rotation=45, ha="right")
    ax.set_ylabel("peak response relative to 1st flash (sp/s)")
    ax.legend()
    fig.tight_layout()
    return fig


def make_boxplot(peak_subtracted_df: pd.DataFrame) -> plt.Figure:
    melted = peak_subtracted_df.melt(var_name="stimulus", value_name="response")
    fig, ax = plt.subplots(1, figsize=(5, 5))
    sns.boxplot(
        data=melted,
        x="stimulus",
        y="response",
        color="red",
        showfliers=False,
        fill=False,
        ax=ax,
    )
    ax.set_xticks(range(len(STIM_LABELS)))
    ax.set_xticklabels(STIM_LABELS, rotation=45, ha="right")
    ax.set_ylabel("peak response relative to 1st flash (sp/s)")
    fig.tight_layout()
    return fig


def make_response_side_mean_plot(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    n_left_trials: int,
    n_right_trials: int,
) -> plt.Figure:
    x = np.arange(len(STIM_LABELS))
    fig, ax = plt.subplots(1, figsize=(5, 5))
    plot_population_points(ax, left_df.to_numpy(), "blue", 0.1, 2)
    plot_population_points(ax, right_df.to_numpy(), "orange", 0.1, 2)
    ax.errorbar(
        x - 0.1,
        left_df.mean(),
        yerr=left_df.sem(),
        fmt="o-",
        color="blue",
        linewidth=2,
        markersize=8,
        capsize=5,
        label=f"Left trials (n={n_left_trials})",
    )
    ax.errorbar(
        x + 0.1,
        right_df.mean(),
        yerr=right_df.sem(),
        fmt="o-",
        color="orange",
        linewidth=2,
        markersize=8,
        capsize=5,
        label=f"Right trials (n={n_right_trials})",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(STIM_LABELS, rotation=45, ha="right")
    ax.set_ylabel("Peak response relative to 1st flash (sp/s)")
    ax.legend()
    fig.tight_layout()
    return fig


def make_response_side_boxplot(
    left_df: pd.DataFrame, right_df: pd.DataFrame
) -> plt.Figure:
    left_melted = left_df.melt(var_name="stimulus", value_name="response")
    left_melted["side"] = "Left"
    right_melted = right_df.melt(var_name="stimulus", value_name="response")
    right_melted["side"] = "Right"
    combined_df = pd.concat([left_melted, right_melted], ignore_index=True)
    fig, ax = plt.subplots(1, figsize=(5, 5))
    sns.boxplot(
        data=combined_df,
        x="stimulus",
        y="response",
        hue="side",
        palette=["blue", "orange"],
        showfliers=False,
        fill=False,
        ax=ax,
    )
    ax.set_xticks(range(len(STIM_LABELS)))
    ax.set_xticklabels(STIM_LABELS, rotation=45, ha="right")
    ax.set_ylabel("Peak response relative to 1st flash (sp/s)")
    ax.legend(title="Response Side")
    fig.tight_layout()
    return fig


def main() -> None:
    trial_ts, spike_times_per_unit = load_inputs()
    trial_ts = filter_trials(trial_ts)

    peak_df = peak_matrix_for_stims(spike_times_per_unit, extract_stimuli(trial_ts))
    peak_subtracted_df = peak_df.subtract(peak_df["peak_first_flash"], axis=0)

    left_trials = trial_ts[trial_ts["response_side"] == 0].copy()
    right_trials = trial_ts[trial_ts["response_side"] == 1].copy()
    print(f"Subject/session: {SUBJECT} {SESSION}")
    print(f"Units: {len(spike_times_per_unit)}")
    print(f"Trials kept: {len(trial_ts)}")
    print(f"Left response trials: {len(left_trials)}")
    print(f"Right response trials: {len(right_trials)}")

    left_peak_df = peak_matrix_for_stims(
        spike_times_per_unit, extract_stimuli(left_trials)
    )
    right_peak_df = peak_matrix_for_stims(
        spike_times_per_unit, extract_stimuli(right_trials)
    )
    left_peak_subtracted_df = left_peak_df.subtract(
        left_peak_df["peak_first_flash"], axis=0
    )
    right_peak_subtracted_df = right_peak_df.subtract(
        right_peak_df["peak_first_flash"], axis=0
    )

    figures = {
        "adaptation_mean.pdf": make_mean_plot(peak_subtracted_df),
        "adaptation_boxplot.pdf": make_boxplot(peak_subtracted_df),
        "adaptation_by_side_mean.pdf": make_response_side_mean_plot(
            left_peak_subtracted_df,
            right_peak_subtracted_df,
            len(left_trials),
            len(right_trials),
        ),
        "adaptation_by_side_boxplot.pdf": make_response_side_boxplot(
            left_peak_subtracted_df,
            right_peak_subtracted_df,
        ),
    }
    for name, fig in figures.items():
        fig.savefig(FIGURE_DIR / name, dpi=300)
        plt.close(fig)
    print(f"Saved adaptation figures to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
