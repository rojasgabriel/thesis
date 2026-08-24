"""Trial tables combining Chipmunk metadata with ephys event timestamps."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

# Silence the setuptools pkg_resources deprecation notice
warnings.filterwarnings("ignore", category=UserWarning, module="datajoint.plugin")


TRIAL_EVENT_COLUMNS = (
    "stim",
    "frames",
    "left_port",
    "center_port",
    "right_port",
    "left_port_exit",
    "center_port_exit",
    "right_port_exit",
    "stimulus_pulse",
    "first_stimulus",
)


def build_trial_table(subject: str, session: str) -> pd.DataFrame:
    """Return one row per trial with Chipmunk metadata and hardware events.

    Event columns contain lists of absolute timestamps from the selected OBX or
    NIDQ stream. For task events, hardware timestamps are used directly. Bpod
    timing only selects between multiple valid center-entry, center-exit, and
    response-port sequences.
    """
    from thesis.ephys.events import fetch_session_events

    align_ev, stimulus_pulses = fetch_session_events(subject, session)

    from chipmunk import Chipmunk

    trial_data = (
        Chipmunk.trial_query(subject_name=subject, session_name=session).proj(
            "response",
            "with_choice",
            "rewarded",
            "early_withdrawal",
            "t_start",
            "t_sync",
            "t_initiate",
            "t_stim",
            "t_gocue",
            "t_react",
            "t_response",
            "stim_duration",
            "stim_rate_vision",
            "category_boundary",
        )
    ).fetch(format="frame")
    trial_df = (
        trial_data.reset_index(
            level=["subject_name", "session_name", "dataset_name"], drop=True
        )
        .sort_index()
        .reset_index()
    )

    trial_starts = align_ev["trial_start"]
    n_events = len(trial_starts)
    n_chipmunk = len(trial_df)
    n = min(n_events, n_chipmunk)
    mismatch = abs(n_events - n_chipmunk)
    if n == 0:
        raise ValueError(
            f"No aligned trials available for {subject} {session}: "
            f"event trials={n_events}, Chipmunk={n_chipmunk}"
        )
    if mismatch > 1:
        raise ValueError(
            f"Suspicious trial-count mismatch for {subject} {session}: "
            f"event trial_start pulses={n_events}, Chipmunk trials={n_chipmunk}. "
            "Refusing to silently truncate."
        )

    trial_df = trial_df.iloc[:n].copy()
    trial_df["event_trial_start_s"] = trial_starts[:n]
    trial_df["prev_rewarded"] = trial_df["rewarded"].shift(1)
    trial_df["prev_response"] = trial_df["response"].shift(1)
    trial_df["prev_stim_rate"] = trial_df["stim_rate_vision"].shift(1)
    trial_df["stim_category"] = pd.cut(
        trial_df["stim_rate_vision"] - trial_df["category_boundary"],
        bins=[-np.inf, -1e-9, 1e-9, np.inf],
        labels=["low_rate", "boundary", "high_rate"],
    )

    trial_ends = np.r_[trial_starts[1:n], np.inf]
    for event_name, timestamps in align_ev.items():
        if event_name == "trial_start":
            continue
        timestamps = np.asarray(timestamps, dtype=float)
        trial_df[event_name] = [
            timestamps[(timestamps >= start) & (timestamps < end)].tolist()
            for start, end in zip(trial_starts[:n], trial_ends, strict=True)
        ]

    pulse_times = stimulus_pulses["timestamp"].to_numpy(dtype=float)
    pulse_widths = stimulus_pulses["width_ms"].to_numpy(dtype=float)
    first_pulses = stimulus_pulses.loc[
        stimulus_pulses["first_in_train"], "timestamp"
    ].to_numpy(dtype=float)
    pulse_masks = [
        (pulse_times >= start) & (pulse_times < end)
        for start, end in zip(trial_starts[:n], trial_ends, strict=True)
    ]
    trial_df["stimulus_pulse"] = [pulse_times[mask].tolist() for mask in pulse_masks]
    trial_df["stimulus_width_ms"] = [
        pulse_widths[mask].tolist() for mask in pulse_masks
    ]
    trial_df["first_stimulus"] = [
        first_pulses[(first_pulses >= start) & (first_pulses < end)].tolist()
        for start, end in zip(trial_starts[:n], trial_ends, strict=True)
    ]

    bpod_sync = trial_df["t_sync"].to_numpy(dtype=float)
    bpod_react = trial_df["t_react"].to_numpy(dtype=float)
    valid_sync = np.isfinite(bpod_sync)
    predicted_react = np.full(n, np.nan)
    valid_react = np.isfinite(bpod_react)
    if valid_sync.sum() >= 2:
        predicted_react[valid_react] = np.interp(
            bpod_react[valid_react],
            bpod_sync[valid_sync],
            trial_starts[:n][valid_sync],
        )

    selected_sequences = []
    for row_index, row in trial_df.reset_index(drop=True).iterrows():
        response = row["response"]
        if response == 1:
            response_entries = row["right_port"]
        elif response == -1:
            response_entries = row["left_port"]
        else:
            response_entries = []
        candidates = []
        for center_exit in row["center_port_exit"]:
            center_entries = [
                timestamp for timestamp in row["center_port"] if timestamp < center_exit
            ]
            following_responses = [
                timestamp for timestamp in response_entries if timestamp > center_exit
            ]
            if center_entries and following_responses:
                candidates.append(
                    (center_entries[-1], center_exit, following_responses[0])
                )

        selected = None
        used_bpod = False
        if len(candidates) == 1:
            selected = candidates[0]
        elif len(candidates) > 1 and np.isfinite(predicted_react[row_index]):
            selected = min(
                candidates,
                key=lambda candidate: abs(candidate[1] - predicted_react[row_index]),
            )
            used_bpod = True
        selected_sequences.append(
            {
                "center_entry_s": np.nan if selected is None else selected[0],
                "center_exit_s": np.nan if selected is None else selected[1],
                "response_port_entry_s": np.nan if selected is None else selected[2],
                "n_task_event_sequences": len(candidates),
                "task_events_disambiguated_with_bpod": used_bpod,
            }
        )

    trial_df = pd.concat(
        [trial_df.reset_index(drop=True), pd.DataFrame(selected_sequences)], axis=1
    )
    return trial_df
