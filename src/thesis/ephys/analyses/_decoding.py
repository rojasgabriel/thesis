"""Shared data preparation for the focused decoding analyses."""

from __future__ import annotations

import numpy as np
import pandas as pd

from thesis.ephys.trials import build_trial_table
from thesis.ephys.units import fetch_unit_table


def load_decoding_data(
    subject: str,
    session: str,
    *,
    unit_criteria_id: int,
    stability_param_id: int | None,
    window_start_s: float,
    window_stop_s: float,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Return units, eligible trials, and trial-by-unit firing rates."""
    if window_stop_s <= window_start_s:
        raise ValueError("window_stop_s must be greater than window_start_s")

    trial_table = build_trial_table(subject, session)
    eligible = trial_table[
        trial_table["response"].isin((-1, 1))
        & trial_table["stim_category"].ne("boundary")
        & trial_table["stim_pulse_times_s"].str.len().gt(0)
    ].copy()
    if eligible.empty:
        raise ValueError(
            f"No eligible non-boundary choice trials for {subject} {session}"
        )
    eligible = eligible.reset_index(drop=True)

    unit_table = fetch_unit_table(
        subject,
        session,
        unit_criteria_id=unit_criteria_id,
        stability_param_id=stability_param_id,
    )
    first_stim_s = eligible["stim_pulse_times_s"].str[0].to_numpy(dtype=float)
    starts_s = first_stim_s + window_start_s
    stops_s = first_stim_s + window_stop_s
    duration_s = window_stop_s - window_start_s
    firing_rates = np.empty((len(eligible), len(unit_table)), dtype=float)
    for unit_idx, spike_times_s in enumerate(unit_table["spike_times_s"]):
        spikes = np.sort(np.asarray(spike_times_s, dtype=float))
        firing_rates[:, unit_idx] = (
            np.searchsorted(spikes, stops_s, side="left")
            - np.searchsorted(spikes, starts_s, side="left")
        ) / duration_s

    return unit_table, eligible, firing_rates
