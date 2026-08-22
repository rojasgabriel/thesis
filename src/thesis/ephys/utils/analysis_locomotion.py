"""Session-level locomotion peak analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
from spks.event_aligned import population_peth

from thesis.ephys.config.locomotion import BASELINE_WINDOW, PETH_KWARGS, RESP_WINDOW
from thesis.ephys.utils.io_session_units import fetch_good_units


def load_trial_classification(subject: str, session: str) -> pd.DataFrame:
    """Load the conditioned-stim classification for one session."""
    from thesis.ephys.utils.io_chipmunk_trials import fetch_trial_metadata
    from thesis.ephys.utils.io_digital_events import fetch_session_events

    align_ev = fetch_session_events(subject, session)
    trial_df = fetch_trial_metadata(subject, session, align_ev)
    return build_trial_stim_classification(align_ev, trial_df).reset_index(drop=True)


def build_trial_stim_classification(align_ev: dict, trial_df) -> pd.DataFrame:
    """Classify 15 ms pulses as stationary or movement for each trial."""
    stim_times = np.asarray(align_ev["stim_ev_15ms"])
    cp_entries = np.asarray(align_ev["center_port"])
    cp_exits = np.asarray(align_ev.get("center_port_exit", []), dtype=float)
    left_entries = np.asarray(align_ev["left_port"])
    right_entries = np.asarray(align_ev["right_port"])
    obx_trial_starts = np.asarray(align_ev["trial_start"])

    n = min(len(trial_df), len(obx_trial_starts))
    bpod_sync = trial_df["t_sync"].iloc[:n].to_numpy(dtype=float)
    obx_sync = obx_trial_starts[:n].astype(float)
    valid = np.isfinite(bpod_sync) & np.isfinite(obx_sync)
    cp_exit_obx = np.interp(
        trial_df["t_react"].iloc[:n].to_numpy(dtype=float),
        bpod_sync[valid],
        obx_sync[valid],
    )
    t_react = trial_df["t_react"].iloc[:n].to_numpy(dtype=float)
    response = trial_df["response"].iloc[:n].to_numpy()

    rows = []
    for i in range(n):
        if not np.isfinite(t_react[i]):
            continue

        trial_start = obx_trial_starts[i]
        trial_end = obx_trial_starts[i + 1] if i + 1 < len(obx_trial_starts) else np.inf
        trial_cp_exits = cp_exits[(cp_exits > trial_start) & (cp_exits < trial_end)]
        cp_exit = (
            trial_cp_exits[np.argmin(np.abs(trial_cp_exits - cp_exit_obx[i]))]
            if trial_cp_exits.size
            else cp_exit_obx[i]
        )
        cp_mask = (
            (cp_entries > trial_start)
            & (cp_entries < cp_exit)
            & (cp_entries < trial_end)
        )
        if not cp_mask.any():
            continue
        cp_entry = cp_entries[cp_mask][-1]

        rp_pool = right_entries if response[i] == 1 else left_entries
        if response[i] not in (-1, 1):
            continue
        rp_mask = (rp_pool > cp_exit) & (rp_pool < trial_end)
        if not rp_mask.any():
            continue
        rp_entry = rp_pool[rp_mask][0]

        stationary_stims = stim_times[
            (stim_times >= cp_entry) & (stim_times < cp_exit)
        ].tolist()
        movement_stims = stim_times[
            (stim_times >= cp_exit) & (stim_times <= rp_entry)
        ].tolist()
        if stationary_stims and movement_stims:
            rows.append(
                {
                    "trial_idx": i,
                    "cp_entry": cp_entry,
                    "cp_exit_obx": cp_exit,
                    "rp_entry": rp_entry,
                    "stationary_stims": stationary_stims,
                    "movement_stims": movement_stims,
                    "n_cp_entries": int(cp_mask.sum()),
                }
            )

    return pd.DataFrame(rows)


def extract_paired_stim_anchors(
    trial_ts: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Return last-stationary and first-movement stimulus times."""
    paired = trial_ts[
        trial_ts["stationary_stims"].str.len().gt(0)
        & trial_ts["movement_stims"].str.len().gt(0)
    ]
    return (
        np.asarray([stims[-1] for stims in paired["stationary_stims"]], dtype=float),
        np.asarray([stims[0] for stims in paired["movement_stims"]], dtype=float),
    )


def compute_locomotion_peaks(
    subject: str,
    session: str,
    unit_criteria_id: int = 1,
) -> pd.DataFrame:
    """Return stationary and movement peaks for all good units in one session."""
    trial_classification = load_trial_classification(subject, session)
    stationary_event_times, movement_event_times = extract_paired_stim_anchors(
        trial_classification
    )
    if stationary_event_times.size == 0 or movement_event_times.size == 0:
        raise RuntimeError(f"No paired locomotion trials for {subject} {session}.")

    spike_times_by_unit = fetch_good_units(subject, session, unit_criteria_id)
    unit_ids = sorted(spike_times_by_unit)
    if not unit_ids:
        raise RuntimeError(f"No good units found for {subject} {session}.")
    spike_times = [spike_times_by_unit[unit_id] for unit_id in unit_ids]

    stationary_peth, bin_edges, _ = population_peth(
        all_spike_times=spike_times,
        alignment_times=stationary_event_times,
        **PETH_KWARGS,
    )
    stationary_peth = stationary_peth / (PETH_KWARGS["binwidth_ms"] / 1000)
    bin_centers_s = (bin_edges[:-1] + bin_edges[1:]) / 2
    movement_peth, _, _ = population_peth(
        all_spike_times=spike_times,
        alignment_times=movement_event_times,
        **PETH_KWARGS,
    )
    movement_peth = movement_peth / (PETH_KWARGS["binwidth_ms"] / 1000)
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
