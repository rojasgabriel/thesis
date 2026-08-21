"""Peak / trough counting on trial-averaged PSTHs.

**Naming convention**

- ``classify_*`` — label units or bins by a rule (here: peak count on mean PSTH).

For double-peak *pipeline* helpers (selectivity + height floor), see ``peak_classification``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_prominences


def classify_peak_count(
    peth: np.ndarray,
    bin_centers: np.ndarray,
    unit_ids: Sequence,
    search_window: tuple[float, float] = (0.0, 0.15),
    baseline_window: tuple[float, float] = (-0.1, 0.0),
    min_prominence_frac: float = 0.25,
    min_prominence_abs: float = 1.0,
    min_distance_ms: float = 20.0,
    binwidth_ms: float = 10.0,
    mode: str = "peaks",
) -> pd.DataFrame:
    """Classify units by the number of peaks (or dips) in their trial-averaged PSTH.

    Uses scipy.signal.find_peaks with a prominence threshold that is the
    *larger* of an adaptive per-unit fraction and an absolute floor.

    In "peaks" mode, only excursions above baseline are counted.
    In "dips" mode, the signal is inverted so that suppression troughs
    are detected instead.

    Args:
        peth: array (n_units, n_trials, n_timebins) firing rates (sp/s).
        bin_centers: array (n_timebins,) in seconds relative to event.
        unit_ids: sequence of unit IDs matching peth's first axis.
        search_window: (start, end) seconds to search for peaks/dips.
        baseline_window: (start, end) seconds for baseline subtraction.
        min_prominence_frac: minimum prominence as a fraction of
            the unit's max excursion (0-1).
        min_prominence_abs: absolute minimum prominence in sp/s. Acts as
            a floor so that noise bumps in low-rate units are ignored.
        min_distance_ms: minimum distance between peaks/dips in ms.
        binwidth_ms: bin width in ms (used to convert distance to bins).
        mode: "peaks" to detect excitatory peaks, "dips" to detect
            suppression troughs.

    Returns:
        DataFrame with columns: unit, n_peaks, peak_times, peak_heights.
        In "dips" mode, peak_heights are the actual (sub-baseline) firing
        rates at the trough locations.
    """
    if mode not in ("peaks", "dips"):
        raise ValueError("mode must be 'peaks' or 'dips'")

    n_units = peth.shape[0]
    if len(unit_ids) != n_units:
        raise ValueError(f"len(unit_ids)={len(unit_ids)} != peth n_units={n_units}")

    base_mask = (bin_centers >= baseline_window[0]) & (bin_centers < baseline_window[1])
    search_mask = (bin_centers >= search_window[0]) & (bin_centers < search_window[1])
    search_idx = np.where(search_mask)[0]
    if len(search_idx) == 0:
        raise ValueError("search_window does not overlap available bins.")

    dist_bins = max(1, int(round(min_distance_ms / binwidth_ms)))

    records = []
    for u in range(n_units):
        mean_psth = peth[u].mean(axis=0)
        baseline_mean = mean_psth[base_mask].mean() if base_mask.any() else 0.0
        excess = mean_psth - baseline_mean

        signal = -excess if mode == "dips" else excess

        max_signal = float(signal[search_mask].max()) if search_mask.any() else 0.0
        if max_signal <= 0:
            records.append(
                dict(unit=unit_ids[u], n_peaks=0, peak_times=[], peak_heights=[])
            )
            continue

        prom_thresh = max(min_prominence_frac * max_signal, min_prominence_abs)
        detected, _ = find_peaks(signal, prominence=prom_thresh, distance=dist_bins)

        in_window = search_mask[detected]
        detected = detected[in_window]

        if mode == "dips":
            valid = excess[detected] < 0
        else:
            valid = excess[detected] > 0
        detected = detected[valid]

        peak_times = bin_centers[detected].tolist()
        peak_heights = mean_psth[detected].tolist()

        if len(detected) == 0:
            window_signal = signal[search_mask]
            if window_signal.size == 0 or window_signal.max() <= 0:
                records.append(
                    dict(unit=unit_ids[u], n_peaks=0, peak_times=[], peak_heights=[])
                )
                continue

            best_i = int(np.argmax(window_signal))
            best_idx = int(search_idx[best_i])
            if best_idx <= 0 or best_idx >= len(signal) - 1:
                records.append(
                    dict(unit=unit_ids[u], n_peaks=0, peak_times=[], peak_heights=[])
                )
                continue

            is_strict_local_max = (signal[best_idx] > signal[best_idx - 1]) and (
                signal[best_idx] > signal[best_idx + 1]
            )
            if not is_strict_local_max:
                records.append(
                    dict(unit=unit_ids[u], n_peaks=0, peak_times=[], peak_heights=[])
                )
                continue

            prom = float(peak_prominences(signal, np.array([best_idx]))[0][0])
            if prom < prom_thresh:
                records.append(
                    dict(unit=unit_ids[u], n_peaks=0, peak_times=[], peak_heights=[])
                )
                continue

            if mode == "dips" and excess[best_idx] >= 0:
                records.append(
                    dict(unit=unit_ids[u], n_peaks=0, peak_times=[], peak_heights=[])
                )
                continue
            if mode == "peaks" and excess[best_idx] <= 0:
                records.append(
                    dict(unit=unit_ids[u], n_peaks=0, peak_times=[], peak_heights=[])
                )
                continue

            records.append(
                dict(
                    unit=unit_ids[u],
                    n_peaks=1,
                    peak_times=[float(bin_centers[best_idx])],
                    peak_heights=[float(mean_psth[best_idx])],
                )
            )
        else:
            records.append(
                dict(
                    unit=unit_ids[u],
                    n_peaks=len(detected),
                    peak_times=peak_times,
                    peak_heights=peak_heights,
                )
            )

    return pd.DataFrame(records)
