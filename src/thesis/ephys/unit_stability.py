"""Unit stability metrics used for unit selection."""

import numpy as np
import pandas as pd
from diptest import diptest
from statsmodels.stats.multitest import multipletests


def compute_unit_stability(
    spike_times: list[np.ndarray],
    spike_amplitudes: list[np.ndarray],
    unit_ids: list[int],
    recording_end: float,
    *,
    n_time_windows: int = 5,
    max_amplitude_drift: float = 0.10,
    dip_alpha: float = 0.05,
    max_dip_samples: int = 72_000,
) -> pd.DataFrame:
    """Return amplitude-stability and unimodality results for each unit."""
    if not (len(spike_times) == len(spike_amplitudes) == len(unit_ids)):
        raise ValueError(
            "Spike times, amplitudes, and unit IDs must have equal length."
        )
    if not unit_ids:
        raise ValueError("At least one unit is required.")

    rows = []
    time_edges = np.linspace(0, recording_end, n_time_windows + 1)[1:-1]
    for unit_id, times, amplitudes in zip(
        unit_ids, spike_times, spike_amplitudes, strict=True
    ):
        times = np.asarray(times)
        amplitudes = np.asarray(amplitudes)
        if len(times) != len(amplitudes) or not len(amplitudes):
            raise ValueError(f"Unit {unit_id} has invalid spike amplitude data.")

        window_index = np.digitize(times, time_edges)
        window_means = np.array(
            [
                amplitudes[window_index == index].mean()
                if np.any(window_index == index)
                else np.nan
                for index in range(n_time_windows)
            ]
        )
        mean_amplitude = amplitudes.mean()
        amplitude_drift = (
            (np.nanmax(window_means) - np.nanmin(window_means)) / mean_amplitude
            if mean_amplitude
            else np.nan
        )

        dip_sample = amplitudes
        if len(dip_sample) > max_dip_samples:
            dip_sample = np.random.default_rng(0).choice(
                dip_sample, max_dip_samples, replace=False
            )
        dip_statistic, dip_p_value = diptest(dip_sample)
        rows.append(
            {
                "unit_id": unit_id,
                "amplitude_drift": amplitude_drift,
                "dip_statistic": dip_statistic,
                "dip_p_value": dip_p_value,
                "dip_sample_size": len(dip_sample),
            }
        )

    results = pd.DataFrame(rows)
    _, results["dip_q_value"], _, _ = multipletests(
        results["dip_p_value"], alpha=dip_alpha, method="fdr_bh"
    )
    results["passes_amplitude_stability"] = np.isfinite(results["amplitude_drift"]) & (
        results["amplitude_drift"] <= max_amplitude_drift
    )
    results["passes_unimodality"] = results["dip_q_value"] >= dip_alpha
    results["passes"] = (
        results["passes_amplitude_stability"] & results["passes_unimodality"]
    )
    return results
