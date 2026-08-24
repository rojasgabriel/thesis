"""Classify and plot stimulus-response peaks."""

from __future__ import annotations

from collections.abc import Collection, Sequence

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_prominences
from spks.event_aligned import population_peth

PETH_PRE_SECONDS = 0.1
PETH_POST_SECONDS = 0.15
PETH_BINWIDTH_MS = 10
BASELINE_WINDOW = (-0.04, 0.0)
PEAK_SEARCH_WINDOW = (0.0, 0.12)


def classify_peak_count(
    peth: np.ndarray,
    bin_centers: np.ndarray,
    unit_ids: Sequence,
    search_window: tuple[float, float] = PEAK_SEARCH_WINDOW,
    baseline_window: tuple[float, float] = BASELINE_WINDOW,
    min_prominence_frac: float = 0.25,
    min_prominence_abs: float = 1.0,
    min_distance_ms: float = 20.0,
    binwidth_ms: float = 10.0,
    mode: str = "peaks",
) -> pd.DataFrame:
    """Classify units by peaks or dips in their trial-averaged PETH."""
    if mode not in ("peaks", "dips"):
        raise ValueError("mode must be 'peaks' or 'dips'")
    if len(unit_ids) != peth.shape[0]:
        raise ValueError(
            f"len(unit_ids)={len(unit_ids)} != peth n_units={peth.shape[0]}"
        )

    base_mask = (bin_centers >= baseline_window[0]) & (bin_centers < baseline_window[1])
    search_mask = (bin_centers >= search_window[0]) & (bin_centers < search_window[1])
    search_idx = np.where(search_mask)[0]
    if not search_idx.size:
        raise ValueError("search_window does not overlap available bins.")

    records = []
    for unit_index, unit_id in enumerate(unit_ids):
        mean_peth = peth[unit_index].mean(axis=0)
        baseline = mean_peth[base_mask].mean() if base_mask.any() else 0.0
        excess = mean_peth - baseline
        signal = -excess if mode == "dips" else excess
        max_signal = float(signal[search_mask].max())
        prominence = max(min_prominence_frac * max_signal, min_prominence_abs)
        detected, _ = find_peaks(
            signal,
            prominence=prominence,
            distance=max(1, round(min_distance_ms / binwidth_ms)),
        )
        detected = detected[search_mask[detected]]
        detected = detected[
            excess[detected] < 0 if mode == "dips" else excess[detected] > 0
        ]

        if not detected.size and max_signal > 0:
            best_idx = int(search_idx[np.argmax(signal[search_mask])])
            if (
                0 < best_idx < len(signal) - 1
                and signal[best_idx] > signal[best_idx - 1]
                and signal[best_idx] > signal[best_idx + 1]
                and peak_prominences(signal, [best_idx])[0][0] >= prominence
            ):
                detected = np.array([best_idx])

        records.append(
            {
                "unit": unit_id,
                "n_peaks": len(detected),
                "peak_times": bin_centers[detected].tolist(),
                "peak_heights": mean_peth[detected].tolist(),
            }
        )

    return pd.DataFrame(records)


def classify_double_peak_units(
    spike_times: list[np.ndarray],
    alignment_times: np.ndarray,
    unit_ids: list[int],
    excited_unit_ids: Collection[int],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[int]]:
    peth, bin_edges, _ = population_peth(
        all_spike_times=spike_times,
        alignment_times=alignment_times,
        pre_seconds=PETH_PRE_SECONDS,
        post_seconds=PETH_POST_SECONDS,
        binwidth_ms=PETH_BINWIDTH_MS,
    )
    peth = peth / (PETH_BINWIDTH_MS / 1000)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    excited_set = set(excited_unit_ids)
    excited_indices = np.array(
        [index for index, unit_id in enumerate(unit_ids) if unit_id in excited_set],
        dtype=int,
    )
    excited_unit_ids = [unit_ids[index] for index in excited_indices]
    excited_peth = peth[excited_indices]
    if not excited_unit_ids:
        return (
            pd.DataFrame(
                {
                    "unit": [],
                    "n_peaks": [],
                    "peak_times": [],
                    "peak_heights": [],
                    "baseline": [],
                    "min_peak_height_above_baseline": [],
                    "max_peak_height_above_baseline": [],
                }
            ),
            peth,
            bin_centers,
            [],
        )
    peak_rows = classify_peak_count(excited_peth, bin_centers, excited_unit_ids)

    double_rows = []
    for _, peak_row in peak_rows.loc[peak_rows["n_peaks"] == 2].iterrows():
        unit_id = int(peak_row["unit"])
        excited_index = excited_unit_ids.index(unit_id)
        baseline_mask = (bin_centers >= BASELINE_WINDOW[0]) & (
            bin_centers < BASELINE_WINDOW[1]
        )
        baseline = float(excited_peth[excited_index].mean(axis=0)[baseline_mask].mean())
        heights_above = [
            float(height - baseline) for height in peak_row["peak_heights"]
        ]
        if min(heights_above) < 5.0:
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
    linestyle: str = "-",
) -> None:
    mean = peth_trials.mean(axis=0)
    sem = peth_trials.std(axis=0) / np.sqrt(peth_trials.shape[0])
    ax.plot(
        bin_centers,
        mean,
        color=color,
        linewidth=1.5,
        linestyle=linestyle,
        label=label,
    )
    ax.fill_between(bin_centers, mean - sem, mean + sem, alpha=0.25, color=color)
