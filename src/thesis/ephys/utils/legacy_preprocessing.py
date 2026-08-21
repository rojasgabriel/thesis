from __future__ import annotations

from glob import glob
from os.path import join as pjoin

import numpy as np
import pandas as pd
import spks


def load_sync_data(sessionpath: str, sync_port: int = 0):
    """Load NI sync data and map it into probe time.

    This preserves the historical preprocessing logic used to generate the
    local `trial_ts.pkl` exports for GRB006-style sessions.
    """
    print("Loading nisync data...")
    (nionsets, nioffsets), (nisync, nimeta), (apsyncdata) = spks.sync.load_ni_sync_data(
        sessionpath=sessionpath
    )
    aponsets = apsyncdata[0]["file0_sync_onsets"][6]

    corrected_onsets = {}
    corrected_offsets = {}
    for key in nionsets.keys():
        corrected_onsets[key] = spks.sync.interp1d(
            nionsets[sync_port], aponsets, fill_value="extrapolate"
        )(nionsets[key]).astype("uint64")
        corrected_offsets[key] = spks.sync.interp1d(
            nionsets[sync_port], aponsets, fill_value="extrapolate"
        )(nioffsets[key]).astype("uint64")

    nitime = spks.sync.interp1d(
        nionsets[sync_port], aponsets, fill_value="extrapolate"
    )(np.arange(len(nisync)))
    srate = apsyncdata[0]["sampling_rate"]
    time_vector = nitime / srate
    analog_signals = {
        "visual": nisync[:, 0],
        "audio": nisync[:, 1],
        "raw": nisync,
        "metadata": nimeta,
    }
    print("Success!\n-----")
    return corrected_onsets, corrected_offsets, time_vector, srate, analog_signals


def process_port_events(
    corrected_onsets: dict[int, np.ndarray],
    corrected_offsets: dict[int, np.ndarray],
    srate: float,
):
    print("Loading nidaq events...")
    trial_starts = corrected_onsets[2] / srate

    if len(corrected_onsets.keys()) <= 3:
        print("No port events registered in this session. Proceeding without them...")
        return trial_starts, None

    print("Port events found. Proceeding with extracting them...")
    port_events = {
        "center_port": {
            "entries": corrected_onsets[4] / srate,
            "exits": corrected_offsets[4] / srate,
        },
        "left_port": {
            "entries": corrected_onsets[3] / srate,
            "exits": corrected_offsets[3] / srate,
        },
        "right_port": {
            "entries": corrected_onsets[5] / srate,
            "exits": corrected_offsets[5] / srate,
        },
    }
    return trial_starts, port_events


def process_trial_data(
    sessionpath: str,
    trial_starts: np.ndarray,
    time_vector: np.ndarray,
    srate: float,
    analog_signals: dict[str, np.ndarray],
    port_events,
    animal: str,
    session: str,
):
    import chiCa.chiCa as chiCa

    behavior_files = glob(
        pjoin(sessionpath, f"chipmunk/{animal}_{session}_chipmunk_*.mat")
    )
    if not behavior_files:
        raise FileNotFoundError(
            f"Could not find chipmunk MAT file under {sessionpath}/chipmunk"
        )

    behavior_data = chiCa.load_trialdata(behavior_files[0])

    stim_ts_per_channel: dict[str, np.ndarray] = {}
    for channel_name, signal in analog_signals.items():
        if not isinstance(signal, np.ndarray) or signal.ndim != 1:
            continue
        stim_ts_per_channel[channel_name] = detect_stim_events(
            time_vector, srate, signal, amp_threshold=5000
        )

    if not stim_ts_per_channel:
        raise ValueError("No 1-D analog signals provided for stimulus detection.")

    if port_events is None:
        trial_ts = get_trial_ts(trial_starts, stim_ts_per_channel, behavior_data)
    else:
        trial_ts = get_trial_ts(
            trial_starts,
            stim_ts_per_channel,
            behavior_data,
            port_events,
        )
        trial_ts = trial_ts[trial_ts.trial_outcome != -1].copy()

        for itrial, data in trial_ts.iterrows():
            if len(data.center_port_entries) == 0:
                continue
            mask = data.center_port_entries < data.first_stim_ts
            true_entry = data.center_port_entries[mask][-1]
            trial_ts.loc[itrial, "center_port_entries"] = [true_entry]

            mask = data.center_port_exits > true_entry
            true_exit = data.center_port_exits[mask][0]
            trial_ts.loc[itrial, "center_port_exits"] = [true_exit]

        trial_ts.insert(
            trial_ts.shape[1], "response", trial_ts.apply(get_response_ts, axis=1)
        )

    trial_ts.insert(
        trial_ts.shape[1],
        "stationary_stims",
        trial_ts.apply(get_stationary_stims, axis=1),
    )
    trial_ts.insert(
        trial_ts.shape[1],
        "movement_stims",
        trial_ts.apply(get_movement_stims, axis=1),
    )
    trial_ts.insert(
        0,
        "category",
        trial_ts.apply(
            lambda row: (
                "left"
                if row.trial_rate < 12
                else ("right" if row.trial_rate > 12 else "boundary")
            ),
            axis=1,
        ),
    )

    print("Success!")
    return behavior_data, trial_ts, stim_ts_per_channel


def get_trial_ts(
    trial_starts: np.ndarray,
    stim_ts_per_channel: dict[str, np.ndarray],
    behavior_data,
    port_events=None,
) -> pd.DataFrame:
    trial_data = []

    channel_names = list(stim_ts_per_channel.keys())
    channel_events = {
        name: np.searchsorted(stim_ts_per_channel[name], trial_starts)
        for name in channel_names
    }

    if channel_names:
        combined_stim_ts = np.sort(
            np.unique(
                np.concatenate(
                    [
                        stim_ts_per_channel[name]
                        for name in channel_names
                        if stim_ts_per_channel[name].size > 0
                    ]
                )
            )
        )
    else:
        combined_stim_ts = np.array([])

    combined_events = (
        np.searchsorted(combined_stim_ts, trial_starts)
        if combined_stim_ts.size > 0
        else np.zeros_like(trial_starts, dtype=int)
    )

    for trial_idx in range(len(trial_starts) - 1):
        start_time = trial_starts[trial_idx]
        end_time = trial_starts[trial_idx + 1]
        stim_dict = {}

        for name in channel_names:
            channel_start = channel_events[name][trial_idx]
            channel_end = channel_events[name][trial_idx + 1]
            channel_ts = stim_ts_per_channel[name][channel_start:channel_end]
            stim_dict[f"stim_ts_{name}"] = channel_ts

        combined_start = combined_events[trial_idx]
        combined_end = combined_events[trial_idx + 1]
        stim_events_in_interval = combined_stim_ts[combined_start:combined_end]

        detected_events = combined_end - combined_start
        first_stim_ts = (
            stim_events_in_interval[0] if len(stim_events_in_interval) > 0 else np.nan
        )

        trial_dict = {
            "trial_rate": len(behavior_data.stimulus_event_timestamps[trial_idx]),
            "detected_events": detected_events,
            "trial_start": start_time,
            "stim_ts": stim_events_in_interval,
            "first_stim_ts": first_stim_ts,
            "stimulus_modality": behavior_data.stimulus_modality[trial_idx],
            "response_side": behavior_data.response_side[trial_idx],
            "correct_side": behavior_data.correct_side[trial_idx],
            "trial_outcome": behavior_data.outcome_record[trial_idx],
        }

        if port_events is not None:
            for port_name, events in port_events.items():
                for event_type in ("entries", "exits"):
                    event_times = events[event_type]
                    mask = (event_times > start_time) & (event_times < end_time)
                    trial_dict[f"{port_name}_{event_type}"] = event_times[mask]

        trial_dict.update(stim_dict)
        trial_data.append(trial_dict)

    return pd.DataFrame(trial_data)


def detect_stim_events(
    time_vector: np.ndarray,
    srate: float,
    signal: np.ndarray,
    amp_threshold: float = 5000,
    time_threshold: float = 0.04,
) -> np.ndarray:
    rising_edges = np.where(np.diff(signal > amp_threshold) == 1)[0]
    return time_vector[
        rising_edges[np.diff(np.hstack([0, rising_edges])) > time_threshold * srate]
    ]


def get_response_ts(row: pd.Series):
    try:
        withdrawal = row["center_port_exits"][-1]
    except Exception:
        withdrawal = None
        print(f"Couldn't find a withdrawal time for trial {row.name}.")

    left = [entry for entry in row["left_port_entries"] if entry > withdrawal]
    right = [entry for entry in row["right_port_entries"] if entry > withdrawal]

    left = left[0] if left else None
    right = right[0] if right else None

    if pd.notna(left):
        return left
    if pd.notna(right):
        return right
    return None


def get_stationary_stims(row: pd.Series) -> np.ndarray:
    return row.stim_ts[row.stim_ts < row.center_port_exits[0]]


def get_movement_stims(row: pd.Series, max_tseconds: float = 0.4) -> np.ndarray:
    if "center_port_exits" in row.index:
        if len(row.center_port_exits) != 0:
            return row.stim_ts[row.stim_ts > row.center_port_exits[0]]
    return row.stim_ts[
        np.logical_and(
            row.first_stim_ts + 0.5 < row.stim_ts,
            row.stim_ts < row.first_stim_ts + 0.5 + max_tseconds,
        )
    ]


def get_cluster_spike_times(
    spike_times: np.ndarray,
    spike_clusters: np.ndarray,
    good_unit_ids: np.ndarray,
) -> list[np.ndarray]:
    return [
        spike_times[good_unit_ids][spike_clusters[good_unit_ids] == unit_id]
        for unit_id in np.unique(spike_clusters[good_unit_ids])
    ]


def get_good_units(clusters_obj, spike_clusters: np.ndarray):
    mask = (
        (
            np.abs(
                clusters_obj.cluster_info.trough_amplitude
                - clusters_obj.cluster_info.peak_amplitude
            )
            > 50
        )
        & (clusters_obj.cluster_info.amplitude_cutoff < 0.1)
        & (clusters_obj.cluster_info.isi_contamination < 0.1)
        & (clusters_obj.cluster_info.presence_ratio >= 0.6)
        & (clusters_obj.cluster_info.spike_duration > 0.1)
        & (clusters_obj.cluster_info.firing_rate > 2)
    )

    good_unit_ids = np.isin(
        spike_clusters, clusters_obj.cluster_info[mask].cluster_id.values
    )
    n_units = len(clusters_obj.cluster_info[mask])
    return good_unit_ids, n_units
