"""Session-level locomotion peak analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd

from thesis.ephys.config.locomotion import BASELINE_WINDOW, PETH_KWARGS, RESP_WINDOW
from thesis.ephys.utils.analysis_conditioned_stim import (
    build_trial_stim_classification,
    extract_conditioned_stim_anchors,
)
from thesis.ephys.utils.analysis_peth import compute_population_peth
from thesis.ephys.utils.io_chipmunk_trials import fetch_trial_metadata
from thesis.ephys.utils.io_digital_events import fetch_session_events
from thesis.ephys.utils.io_session_units import fetch_good_units


def compute_locomotion_peaks(
    subject: str,
    session: str,
    unit_criteria_id: int = 1,
) -> pd.DataFrame:
    """Return stationary and movement peaks for all good units in one session."""
    aligned_events = fetch_session_events(subject, session)
    trial_table = fetch_trial_metadata(subject, session, aligned_events)
    if trial_table is None:
        raise RuntimeError(f"Could not load trial metadata for {subject} {session}.")

    trial_classification = build_trial_stim_classification(aligned_events, trial_table)
    anchors = extract_conditioned_stim_anchors(trial_classification)
    stationary_event_times = anchors["paired_last_stationary"]
    movement_event_times = anchors["paired_first_movement"]
    if stationary_event_times.size == 0 or movement_event_times.size == 0:
        raise RuntimeError(f"No paired locomotion trials for {subject} {session}.")

    spike_times_by_unit = fetch_good_units(subject, session, unit_criteria_id)
    unit_ids = sorted(spike_times_by_unit)
    if not unit_ids:
        raise RuntimeError(f"No good units found for {subject} {session}.")
    spike_times = [spike_times_by_unit[unit_id] for unit_id in unit_ids]

    stationary_peth, _, bin_centers_s = compute_population_peth(
        spike_times, stationary_event_times, **PETH_KWARGS
    )
    movement_peth, _, _ = compute_population_peth(
        spike_times, movement_event_times, **PETH_KWARGS
    )
    stationary_mean_rate = stationary_peth.mean(axis=1)
    movement_mean_rate = movement_peth.mean(axis=1)

    baseline_mask = (bin_centers_s >= BASELINE_WINDOW[0]) & (
        bin_centers_s < BASELINE_WINDOW[1]
    )
    stationary_baseline_rate = stationary_mean_rate[:, baseline_mask].mean(axis=1)
    response_mask = (bin_centers_s >= RESP_WINDOW[0]) & (bin_centers_s < RESP_WINDOW[1])
    response_bin_centers_s = bin_centers_s[response_mask]
    stationary_response = (
        stationary_mean_rate[:, response_mask] - stationary_baseline_rate[:, None]
    )
    movement_response = (
        movement_mean_rate[:, response_mask] - stationary_baseline_rate[:, None]
    )
    stationary_peak_idx = np.argmax(stationary_response, axis=1)
    movement_peak_idx = np.argmax(movement_response, axis=1)
    unit_index = np.arange(len(unit_ids))

    return pd.DataFrame(
        {
            "unit_id": unit_ids,
            "stat_peak": stationary_response[unit_index, stationary_peak_idx],
            "stat_latency": response_bin_centers_s[stationary_peak_idx],
            "move_peak": movement_response[unit_index, movement_peak_idx],
            "move_latency": response_bin_centers_s[movement_peak_idx],
        }
    )
