"""Shared data preparation for the focused decoding analyses."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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
    include_mua: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Return units, eligible trials, and trial-by-unit firing rates.

    With `include_mua` (the default), the units are every cluster of the criteria that is not
    noise (`passes` 1 = single unit, 0 = multi-unit; -1 = noise is left out),
    with no stability filter: UnitStability has rows for single units only.
    """
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

    if include_mua:
        if stability_param_id is not None:
            raise ValueError(
                "Multi-units have no stability rows; use single units only (--units sua) for a stability filter."
            )
        unit_table = _units_with_mua(subject, session, unit_criteria_id)
    else:
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


def _units_with_mua(subject: str, session: str, unit_criteria_id: int) -> pd.DataFrame:
    """Return the spike times of all single- and multi-units (not noise)."""
    from labdata.schema import SpikeSorting, UnitCount

    query = {
        "subject_name": subject,
        "session_name": session,
        "unit_criteria_id": unit_criteria_id,
    }
    units = SpikeSorting.Unit & (UnitCount.Unit & query & "passes >= 0")
    table = pd.DataFrame(units.get_spike_times(include_metrics=False))
    return table.rename(columns={"spike_times": "spike_times_s"})


def decoder():
    """Return the linear decoder of the decoding analyses (z-score, logistic)."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000),
    )


def balanced_indices(
    category: np.ndarray,
    choice: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return trial indices with equal numbers in the four category x choice groups."""
    category_choice_pairs = [(cat, response) for cat in (0, 1) for response in (-1, 1)]
    counts = [
        np.sum((category == cat) & (choice == response))
        for cat, response in category_choice_pairs
    ]
    n_trials_per_pair = int(min(counts))
    if n_trials_per_pair < 2:
        raise ValueError(
            "Too few category x choice trials; least represented pair has "
            f"{n_trials_per_pair}"
        )
    return np.sort(
        np.concatenate(
            [
                rng.choice(
                    np.flatnonzero((category == cat) & (choice == response)),
                    n_trials_per_pair,
                    replace=False,
                )
                for cat, response in category_choice_pairs
            ]
        )
    )


def window_rates(
    units: pd.DataFrame, starts_s: np.ndarray, stops_s: np.ndarray
) -> np.ndarray:
    """Return the rate (spikes/s) of each unit in each trial window, (trial, unit)."""
    rates = np.empty((len(starts_s), len(units)), dtype=float)
    duration = np.asarray(stops_s) - np.asarray(starts_s)
    for unit, spike_times_s in enumerate(units["spike_times_s"]):
        spikes = np.sort(np.asarray(spike_times_s, dtype=float))
        rates[:, unit] = (
            np.searchsorted(spikes, stops_s) - np.searchsorted(spikes, starts_s)
        ) / duration
    return rates
