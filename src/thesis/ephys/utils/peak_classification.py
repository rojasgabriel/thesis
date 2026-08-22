from __future__ import annotations

import numpy as np
import pandas as pd

from thesis.ephys.config.double_peak import (
    BASELINE_WINDOW,
    MIN_PEAK_HEIGHT_ABS,
    PEAK_KWARGS,
    PETH_KWARGS,
    SELECTIVITY_KWARGS,
)
from thesis.ephys.utils.analysis_peak_counts import classify_peak_count
from thesis.ephys.utils.analysis_peth import compute_population_peth
from thesis.ephys.utils.analysis_selectivity import compute_unit_selectivity


def baseline_mean(
    peth_trials: np.ndarray,
    bin_centers: np.ndarray,
    baseline_window: tuple[float, float],
) -> float:
    mask = (bin_centers >= baseline_window[0]) & (bin_centers < baseline_window[1])
    return float(peth_trials.mean(axis=0)[mask].mean())


def classify_double_peak_units(
    spike_times: list[np.ndarray], alignment_times: np.ndarray, unit_ids: list[int]
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[int]]:
    peth, bin_edges, bin_centers = compute_population_peth(
        spike_times_per_unit=spike_times,
        alignment_times=alignment_times,
        **PETH_KWARGS,
    )
    _, masks = compute_unit_selectivity(
        peth, bin_edges, unit_ids=unit_ids, **SELECTIVITY_KWARGS
    )
    excited_indices = np.where(masks["excited"])[0]
    excited_unit_ids = [unit_ids[i] for i in excited_indices]
    excited_peth = peth[excited_indices]
    peak_rows = classify_peak_count(
        excited_peth, bin_centers, unit_ids=excited_unit_ids, **PEAK_KWARGS
    )

    double_rows = []
    for _, peak_row in peak_rows.loc[peak_rows["n_peaks"] == 2].iterrows():
        unit_id = int(peak_row["unit"])
        excited_index = excited_unit_ids.index(unit_id)
        baseline = baseline_mean(
            excited_peth[excited_index], bin_centers, BASELINE_WINDOW
        )
        heights_above = [
            float(height - baseline) for height in peak_row["peak_heights"]
        ]
        if min(heights_above) < MIN_PEAK_HEIGHT_ABS:
            continue
        row = peak_row.copy()
        row["baseline"] = baseline
        row["min_peak_height_above_baseline"] = min(heights_above)
        row["max_peak_height_above_baseline"] = max(heights_above)
        double_rows.append(row)

    double_peak_rows = pd.DataFrame(double_rows)
    if double_peak_rows.empty:
        empty = peak_rows.iloc[0:0].copy()
        double_peak_rows = empty.reindex(
            columns=list(peak_rows.columns)
            + [
                "baseline",
                "min_peak_height_above_baseline",
                "max_peak_height_above_baseline",
            ]
        )
    return double_peak_rows, peth, bin_centers, excited_unit_ids


def plot_mean_sem_trace(
    ax,
    bin_centers: np.ndarray,
    peth_trials: np.ndarray,
    color: str,
    label: str | None = None,
) -> None:
    mean = peth_trials.mean(axis=0)
    sem = peth_trials.std(axis=0) / np.sqrt(peth_trials.shape[0])
    ax.plot(bin_centers, mean, color=color, linewidth=1.5, label=label)
    ax.fill_between(bin_centers, mean - sem, mean + sem, alpha=0.25, color=color)


def mark_peaks(ax, peak_row, color: str, marker: str = "v", markersize: float = 7):
    for peak_time, peak_height in zip(peak_row["peak_times"], peak_row["peak_heights"]):
        ax.plot(
            peak_time, peak_height, marker, color=color, markersize=markersize, zorder=5
        )
