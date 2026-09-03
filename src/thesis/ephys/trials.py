"""Trial tables combining Chipmunk metadata with ephys event timestamps."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

# Silence the setuptools pkg_resources deprecation notice
warnings.filterwarnings("ignore", category=UserWarning, module="datajoint.plugin")


ALIGNMENT_EV_COLUMNS = (
    "first_stim_times_s",
    "stim_pulse_times_s",
    "frame_times_s",
    "left_port_entry_times_s",
    "left_port_exit_times_s",
    "center_port_entry_times_s",
    "center_port_exit_times_s",
    "right_port_entry_times_s",
    "right_port_exit_times_s",
)


def _split_bpod_pokes(pokes: object) -> tuple[np.ndarray, np.ndarray]:
    """Return Bpod entry and exit times from a trial's poke array."""
    if pokes is None or (np.isscalar(pokes) and pd.isna(pokes)):
        return np.array([]), np.array([])
    poke_array = np.asarray(pokes, dtype=float)
    if poke_array.size == 0:
        return np.array([]), np.array([])
    if poke_array.ndim != 2 or poke_array.shape[1] != 2:
        raise ValueError(f"Expected an n-by-2 Bpod poke array, got {poke_array.shape}")
    if not np.all(np.isin(poke_array[:, 1], (0, 1))):
        raise ValueError("Bpod poke states must be 0 or 1")
    return poke_array[poke_array[:, 1] == 1, 0], poke_array[poke_array[:, 1] == 0, 0]


def _select_task_ev_sequence(trial: pd.Series, predicted_react_s: float) -> dict:
    """Select the hardware sequence that matches the Bpod reaction event."""
    response_entries_s = {
        -1: trial["left_port_entry_times_s"],
        1: trial["right_port_entry_times_s"],
    }.get(trial["response"], [])
    task_ev_candidates = []
    for center_exit_s in trial["center_port_exit_times_s"]:
        center_entries_s = [
            timestamp_s
            for timestamp_s in trial["center_port_entry_times_s"]
            if timestamp_s < center_exit_s
        ]
        following_response_entries_s = [
            timestamp_s
            for timestamp_s in response_entries_s
            if timestamp_s > center_exit_s
        ]
        if center_entries_s and following_response_entries_s:
            task_ev_candidates.append(
                (
                    center_entries_s[-1],
                    center_exit_s,
                    following_response_entries_s[0],
                )
            )

    selected_task_ev = None
    if len(task_ev_candidates) == 1:
        selected_task_ev = task_ev_candidates[0]
    elif len(task_ev_candidates) > 1 and np.isfinite(predicted_react_s):
        selected_task_ev = min(
            task_ev_candidates,
            key=lambda task_ev: abs(task_ev[1] - predicted_react_s),
        )

    return {
        "center_entry_s": (np.nan if selected_task_ev is None else selected_task_ev[0]),
        "center_exit_s": np.nan if selected_task_ev is None else selected_task_ev[1],
        "response_port_entry_s": (
            np.nan if selected_task_ev is None else selected_task_ev[2]
        ),
        "n_task_ev_candidates": len(task_ev_candidates),
    }


def build_trial_table(subject: str, session: str) -> pd.DataFrame:
    """Return one row per trial with Chipmunk metadata and hardware events.

    Event columns contain lists of absolute timestamps from the selected OBX or
    NIDQ stream. When several hardware sequences are possible, Bpod identifies
    which center exit triggered the task response; the returned timestamps
    remain the more precise hardware timestamps. Scalar timestamps end in
    ``_s``; timestamp lists end in ``_times_s``.
    """
    from thesis.ephys.events import fetch_session_events

    sess_ev, stim_pulses = fetch_session_events(subject, session, allow_incomplete=True)

    from chipmunk import Chipmunk

    chipmunk_trials = (
        Chipmunk.trial_query(subject_name=subject, session_name=session).proj(
            "response",
            "rewarded",
            "early_withdrawal",
            "t_react",
            "left_poke",
            "center_poke",
            "right_poke",
            "stim_duration",
            "stim_rate_vision",
            "category_boundary",
        )
    ).fetch(format="frame")
    trial_table = (
        chipmunk_trials.reset_index(
            level=["subject_name", "session_name", "dataset_name"], drop=True
        )
        .sort_index()
        .reset_index()
        .rename(
            columns={
                "stim_duration": "stim_duration_s",
                "stim_rate_vision": "visual_stim_rate_hz",
                "category_boundary": "category_boundary_hz",
            }
        )
    )

    trial_starts_s = sess_ev["trial_start"]
    n_ev_trials = len(trial_starts_s)
    n_chipmunk_trials = len(trial_table)
    n_trials = min(n_ev_trials, n_chipmunk_trials)
    trial_count_mismatch = abs(n_ev_trials - n_chipmunk_trials)
    if n_trials == 0:
        raise ValueError(
            f"No aligned trials available for {subject} {session}: "
            f"event trials={n_ev_trials}, Chipmunk={n_chipmunk_trials}"
        )
    if trial_count_mismatch > 1:
        raise ValueError(
            f"Suspicious trial-count mismatch for {subject} {session}: "
            f"event trial_start pulses={n_ev_trials}, "
            f"Chipmunk trials={n_chipmunk_trials}. "
            "Refusing to silently truncate."
        )

    trial_table = trial_table.iloc[:n_trials].copy()
    trial_table["trial_start_s"] = trial_starts_s[:n_trials]
    trial_table["prev_rewarded"] = trial_table["rewarded"].shift(1)
    trial_table["prev_response"] = trial_table["response"].shift(1)
    trial_table["stim_category"] = pd.cut(
        trial_table["visual_stim_rate_hz"] - trial_table["category_boundary_hz"],
        bins=[-np.inf, -1e-9, 1e-9, np.inf],
        labels=["low_rate", "boundary", "high_rate"],
    )

    trial_ends_s = np.r_[trial_starts_s[1:n_trials], np.inf]
    ev_column_by_role = {
        "frames": "frame_times_s",
        "left_port": "left_port_entry_times_s",
        "left_port_exit": "left_port_exit_times_s",
        "center_port": "center_port_entry_times_s",
        "center_port_exit": "center_port_exit_times_s",
        "right_port": "right_port_entry_times_s",
        "right_port_exit": "right_port_exit_times_s",
    }
    for ev_name, ev_timestamps in sess_ev.items():
        if ev_name in ("trial_start", "stim"):
            continue
        ev_timestamps = np.asarray(ev_timestamps, dtype=float)
        trial_table[ev_column_by_role[ev_name]] = [
            ev_timestamps[
                (ev_timestamps >= trial_start_s) & (ev_timestamps < trial_end_s)
            ].tolist()
            for trial_start_s, trial_end_s in zip(
                trial_starts_s[:n_trials], trial_ends_s, strict=True
            )
        ]

    pulse_times_s = stim_pulses["timestamp"].to_numpy(dtype=float)
    pulse_widths_ms = stim_pulses["width_ms"].to_numpy(dtype=float)
    first_pulse_times_s = stim_pulses.loc[
        stim_pulses["first_in_train"], "timestamp"
    ].to_numpy(dtype=float)
    pulse_masks = [
        (pulse_times_s >= trial_start_s) & (pulse_times_s < trial_end_s)
        for trial_start_s, trial_end_s in zip(
            trial_starts_s[:n_trials], trial_ends_s, strict=True
        )
    ]
    trial_table["stim_pulse_times_s"] = [
        pulse_times_s[pulse_mask].tolist() for pulse_mask in pulse_masks
    ]
    trial_table["stim_pulse_widths_ms"] = [
        pulse_widths_ms[pulse_mask].tolist() for pulse_mask in pulse_masks
    ]
    trial_table["first_stim_times_s"] = [
        first_pulse_times_s[
            (first_pulse_times_s >= trial_start_s) & (first_pulse_times_s < trial_end_s)
        ].tolist()
        for trial_start_s, trial_end_s in zip(
            trial_starts_s[:n_trials], trial_ends_s, strict=True
        )
    ]

    missing_port_roles = [
        port_role
        for port_role in ("left_port", "center_port", "right_port")
        if port_role not in sess_ev
    ]
    missing_frames = "frames" not in sess_ev
    bpod_react_s = trial_table["t_react"].to_numpy(dtype=float)
    predicted_react_s = np.full(n_trials, np.nan)
    valid_bpod_react = np.isfinite(bpod_react_s)
    if valid_bpod_react.any() or missing_port_roles:
        from labdata.schema import StreamSync

        bpod_sync = StreamSync() & {
            "subject_name": subject,
            "session_name": session,
            "dataset_name": "chipmunk",
            "stream_name": "bpod",
            "event_name": "sync",
        }
        if len(bpod_sync) != 1:
            raise ValueError(
                f"Missing Bpod StreamSync for {subject} {session}; "
                "run uv run insert-stream-sync for its ephys dataset"
            )

        for port_role in missing_port_roles:
            entries_by_trial = []
            exits_by_trial = []
            for pokes in trial_table[f"{port_role.removesuffix('_port')}_poke"]:
                bpod_entries, bpod_exits = _split_bpod_pokes(pokes)
                entries_by_trial.append(
                    bpod_sync.apply(bpod_entries, force=True, warn=False).tolist()
                    if bpod_entries.size
                    else []
                )
                exits_by_trial.append(
                    bpod_sync.apply(bpod_exits, force=True, warn=False).tolist()
                    if bpod_exits.size
                    else []
                )
            trial_table[f"{port_role}_entry_times_s"] = entries_by_trial
            trial_table[f"{port_role}_exit_times_s"] = exits_by_trial

        predicted_react_s[valid_bpod_react] = bpod_sync.apply(
            bpod_react_s[valid_bpod_react], force=True, warn=False
        )

    if missing_frames:
        trial_table["frame_times_s"] = [[] for _ in range(n_trials)]
    if missing_port_roles or missing_frames:
        fallback_notes = []
        if missing_port_roles:
            fallback_notes.append(
                f"using Bpod times for {', '.join(missing_port_roles)}"
            )
        if missing_frames:
            fallback_notes.append(
                "screen-frame times are unavailable because Chipmunk does not "
                "store them"
            )
        warnings.warn(
            f"Incomplete hardware events for {subject} {session}: "
            + "; ".join(fallback_notes),
            RuntimeWarning,
            stacklevel=2,
        )

    task_ev_selections = [
        _select_task_ev_sequence(trial, predicted_react_s[trial_index])
        for trial_index, trial in trial_table.reset_index(drop=True).iterrows()
    ]
    trial_table = pd.concat(
        [trial_table.reset_index(drop=True), pd.DataFrame(task_ev_selections)], axis=1
    )
    trial_table["center_exit_bpod_error_s"] = (
        trial_table["center_exit_s"] - predicted_react_s
    )
    return trial_table[
        [
            "trial_num",
            "visual_stim_rate_hz",
            "category_boundary_hz",
            "stim_category",
            "stim_duration_s",
            "response",
            "rewarded",
            "early_withdrawal",
            "prev_response",
            "prev_rewarded",
            "trial_start_s",
            "center_entry_s",
            "first_stim_times_s",
            "center_exit_s",
            "response_port_entry_s",
            "stim_pulse_times_s",
            "stim_pulse_widths_ms",
            "frame_times_s",
            "left_port_entry_times_s",
            "left_port_exit_times_s",
            "center_port_entry_times_s",
            "center_port_exit_times_s",
            "right_port_entry_times_s",
            "right_port_exit_times_s",
            "n_task_ev_candidates",
            "center_exit_bpod_error_s",
        ]
    ]
