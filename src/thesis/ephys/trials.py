"""Trial tables combining Chipmunk metadata with ephys event timestamps.

Sessions with passive replay have more OBX/NIDQ trial_start (Bpod sync) pulses
than task trials: the N task pulses come first, then one pulse per replay. A
trial cut short by Stop adds one unsaved pulse; the Bpod sync event values (clock
pulse numbers) skip it.
`build_trial_table` uses only the task pulses; `build_replay_table` gives the
replay trials.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from thesis.ephys.cache import cached

ALIGNMENT_EV_COLUMNS = (
    "first_stim_times_s",
    "stim_pulse_times_s",
    "audio_stim_times_s",
    "go_cue_times_s",
    "punish_wrong_times_s",
    "punish_early_times_s",
    "left_port_entry_times_s",
    "left_port_exit_times_s",
    "center_port_entry_times_s",
    "center_port_exit_times_s",
    "right_port_entry_times_s",
    "right_port_exit_times_s",
)


# Largest difference between the hardware and the Bpod response entry that is
# taken as the same event. The hardware entry is about 8 ms before the Bpod one.
RESPONSE_TOLERANCE_S = 0.02
CATEGORY_BOUNDARY_HZ = 12.0
STRENGTH_SPAN_HZ = 8.0  # the strongest rates, 4 and 20 flashes/s, give -1 and +1


def stimulus_strength(rate_hz):
    """Signed stimulus strength: 0 at the category boundary (no evidence for a
    side), negative toward the left reward, positive toward the right."""
    return (np.asarray(rate_hz, dtype=float) - CATEGORY_BOUNDARY_HZ) / STRENGTH_SPAN_HZ


def check_strength_axis(trials: pd.DataFrame) -> None:
    """Raise if `stimulus_strength` does not fit these trials.

    The axis needs the boundary `CATEGORY_BOUNDARY_HZ` and fast trains
    rewarded on the right (response 1), slow trains on the left (-1).
    """
    boundary = np.unique(trials["category_boundary_hz"].to_numpy(float))
    if not np.allclose(boundary, CATEGORY_BOUNDARY_HZ):
        raise ValueError(
            f"Category boundary {boundary} Hz, not {CATEGORY_BOUNDARY_HZ}."
        )
    rewarded = trials[(trials["rewarded"] == 1) & trials["response"].isin((-1, 1))]
    side = rewarded.groupby("stim_category")["response"].unique()
    for category, expected in (("high_rate", 1), ("low_rate", -1)):
        if category in side and set(side[category]) != {expected}:
            raise ValueError(
                f"Rewarded {category} trials have responses {side[category]}, "
                f"not {expected}: the strength axis assumes fast = right."
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


def _select_task_ev_sequence(
    trial: pd.Series,
    predicted_react_s: float,
    predicted_entry_s: float = np.nan,
    predicted_response_s: float = np.nan,
) -> dict:
    """Match hardware entry, exit, and response entry to the Bpod events.

    A hardware port line can give bursts of false entries (GRB064 20260916_132939,
    left port). So the response entry is the hardware entry nearest to the Bpod
    response entry, not the first one after the center exit. When no hardware
    entry is within `RESPONSE_TOLERANCE_S` of it, the Bpod time is used
    (`response_entry_source`).
    """
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
        has_bpod_response = (
            np.isfinite(predicted_response_s) and predicted_response_s > center_exit_s
        )
        if not center_entries_s:
            continue
        if following_response_entries_s:
            response_s = (
                min(
                    following_response_entries_s,
                    key=lambda t: abs(t - predicted_response_s),
                )
                if np.isfinite(predicted_response_s)
                else following_response_entries_s[0]
            )
            source = "hardware"
        elif has_bpod_response:
            response_s, source = predicted_response_s, "bpod"
        else:
            continue
        task_ev_candidates.append(
            (
                (
                    min(center_entries_s, key=lambda t: abs(t - predicted_entry_s))
                    if np.isfinite(predicted_entry_s)
                    else center_entries_s[-1]
                ),
                center_exit_s,
                response_s,
                source,
            )
        )

    selected_task_ev = (np.nan, np.nan, np.nan, "hardware")
    if len(task_ev_candidates) == 1:
        selected_task_ev = task_ev_candidates[0]
    elif len(task_ev_candidates) > 1 and np.isfinite(predicted_react_s):
        selected_task_ev = min(
            task_ev_candidates,
            key=lambda task_ev: abs(task_ev[1] - predicted_react_s),
        )

    if (
        np.isfinite(selected_task_ev[2])
        and np.isfinite(predicted_response_s)
        and abs(selected_task_ev[2] - predicted_response_s) > RESPONSE_TOLERANCE_S
    ):
        selected_task_ev = (*selected_task_ev[:2], predicted_response_s, "bpod")
    return {
        "center_entry_s": selected_task_ev[0],
        "center_exit_s": selected_task_ev[1],
        "response_port_entry_s": selected_task_ev[2],
        "response_entry_source": selected_task_ev[3],
        "n_task_ev_candidates": len(task_ev_candidates),
    }


def _replay_trials(subject: str, session: str) -> pd.DataFrame:
    """Return ChipmunkPassive.Trial rows in replay order (empty without the table)."""
    columns = ["passive_trial_num", "source_trial_num"]
    try:
        from chipmunk import ChipmunkPassive
    except ImportError:  # plugin version before the passive dataset
        return pd.DataFrame(columns=columns)
    rows = (
        ChipmunkPassive.Trial & {"subject_name": subject, "session_name": session}
    ).fetch(*columns, order_by="passive_trial_num", as_dict=True)
    return pd.DataFrame(rows, columns=columns)


def _trial_start_indices(
    trial_starts_s: np.ndarray, n_task: int, n_replay: int, subject: str, session: str
) -> np.ndarray | None:
    """Return the trial_start pulse index of each task trial, then each replay.

    None for sessions without replay, which keep the old one-pulse tolerance.
    Pulses are paired by the clock pulse numbers of the Bpod sync stream, so the
    unsaved pulse of a trial cut short by Stop is skipped.
    """
    if not n_replay:
        return None
    from labdata.schema import DatasetEvents

    from thesis.ephys.events import saved_pulse_indices

    rows = (
        DatasetEvents.Digital()
        & {
            "subject_name": subject,
            "session_name": session,
            "dataset_name": "chipmunk",
            "stream_name": "bpod",
            "event_name": "sync",
        }
    ).fetch("event_values")
    numbers = rows[0] if len(rows) == 1 else None
    try:
        return saved_pulse_indices(len(trial_starts_s), n_task + n_replay, numbers)
    except ValueError as error:
        raise ValueError(f"{subject} {session}: trial-start pulses: {error}") from error


def _split_by_windows(windows, times_s, values=None) -> list[list[float]]:
    """Return values (default: the times) inside each (start, end) window."""
    times_s = np.asarray(times_s, dtype=float)
    values = times_s if values is None else np.asarray(values, dtype=float)
    return [
        values[(times_s >= start_s) & (times_s < end_s)].tolist()
        for start_s, end_s in windows
    ]


@cached
def build_trial_table(
    subject: str, session: str, *, include_frames: bool = True
) -> pd.DataFrame:
    """Return one row per trial with Chipmunk metadata and hardware events.

    Event columns contain lists of absolute timestamps from the selected OBX or
    NIDQ stream. When several hardware sequences are possible, Bpod identifies
    which center exit triggered the task response; the returned timestamps
    remain the more precise hardware timestamps. Scalar timestamps end in
    ``_s``; timestamp lists end in ``_times_s``.

    Audio columns use measured OBX events when available and synchronized Bpod
    command times otherwise. They are empty when neither source is available.
    Set `include_frames=False` when screen-frame timestamps are not needed.
    """
    from thesis.ephys.events import fetch_session_events

    sess_ev, stim_pulses = fetch_session_events(
        subject,
        session,
        allow_incomplete=True,
        include_frames=include_frames,
    )

    from chipmunk import Chipmunk

    chipmunk_trials = (
        Chipmunk.trial_query(subject_name=subject, session_name=session).proj(
            "response",
            "rewarded",
            "early_withdrawal",
            "t_react",
            "t_initiate",
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

    all_starts_s = sess_ev["trial_start"]
    start_index = _trial_start_indices(
        all_starts_s,
        len(trial_table),
        len(_replay_trials(subject, session)),
        subject,
        session,
    )
    trial_starts_s = (
        all_starts_s
        if start_index is None
        else all_starts_s[start_index[: len(trial_table)]]
    )
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
    # The side of the last response before this trial: the trial before can have
    # no response (early withdrawal), so go back to the last trial with one.
    # ponytail: from the response, not from the side-port pokes; use the poke
    # times if the animal goes to a side port in trials with no response.
    trial_table["last_side"] = (
        trial_table["response"].where(trial_table["response"] != 0).ffill().shift(1)
    )
    trial_table["stim_category"] = pd.cut(
        trial_table["visual_stim_rate_hz"] - trial_table["category_boundary_hz"],
        bins=[-np.inf, -1e-9, 1e-9, np.inf],
        labels=["low_rate", "boundary", "high_rate"],
    )

    # With replay, each trial ends at the next clock pulse, so the events of a
    # trial cut short by Stop and of the replays stay out of the last task trial.
    if start_index is None:
        trial_ends_s = np.r_[trial_starts_s[1:n_trials], np.inf]
    else:
        trial_ends_s = np.r_[all_starts_s, np.inf][start_index[:n_trials] + 1]
    trial_windows = list(zip(trial_starts_s[:n_trials], trial_ends_s, strict=True))

    def split_by_trial(times_s, values=None) -> list[list[float]]:
        return _split_by_windows(trial_windows, times_s, values)

    ev_column_by_role = {
        "frames": "frame_times_s",
        "left_port": "left_port_entry_times_s",
        "left_port_exit": "left_port_exit_times_s",
        "center_port": "center_port_entry_times_s",
        "center_port_exit": "center_port_exit_times_s",
        "right_port": "right_port_entry_times_s",
        "right_port_exit": "right_port_exit_times_s",
        "audio_stim": "audio_stim_times_s",
        "go_cue": "go_cue_times_s",
        "punish_wrong": "punish_wrong_times_s",
        "punish_early": "punish_early_times_s",
    }
    for ev_name, ev_timestamps in sess_ev.items():
        if ev_name not in ("trial_start", "stim"):
            trial_table[ev_column_by_role[ev_name]] = split_by_trial(ev_timestamps)
    for ev_column in ev_column_by_role.values():
        if ev_column not in trial_table:
            trial_table[ev_column] = [[] for _ in range(n_trials)]

    pulse_times_s = stim_pulses["timestamp"]
    trial_table["stim_pulse_times_s"] = split_by_trial(pulse_times_s)
    trial_table["stim_pulse_widths_ms"] = split_by_trial(
        pulse_times_s, stim_pulses["width_ms"]
    )
    trial_table["first_stim_times_s"] = split_by_trial(
        stim_pulses.loc[stim_pulses["first_in_train"], "timestamp"]
    )

    missing_port_roles = [
        port_role
        for port_role in ("left_port", "center_port", "right_port")
        if port_role not in sess_ev
    ]
    missing_frames = include_frames and "frames" not in sess_ev
    bpod_react_s = trial_table["t_react"].to_numpy(dtype=float)
    predicted_react_s = np.full(n_trials, np.nan)
    bpod_entry_s = np.full(n_trials, np.nan)
    bpod_response_s = np.full(n_trials, np.nan)
    for i, trial in trial_table.iterrows():
        entries, _ = _split_bpod_pokes(trial["center_poke"])
        if entries.size and np.isfinite(trial["t_initiate"]):
            # Stored state times can be rounded to 10 ms. Match the nearest
            # Bpod poke, not the last poke before that rounded state time.
            bpod_entry_s[i] = entries[np.argmin(abs(entries - trial["t_initiate"]))]
        # The first Bpod entry into the response port after the reaction.
        side = {-1: "left_poke", 1: "right_poke"}.get(trial["response"])
        if side and np.isfinite(trial["t_react"]):
            entries, _ = _split_bpod_pokes(trial[side])
            entries = entries[entries > trial["t_react"]]
            if entries.size:
                bpod_response_s[i] = entries[0]
    predicted_entry_s = np.full(n_trials, np.nan)
    predicted_response_s = np.full(n_trials, np.nan)
    valid_bpod_entry = np.isfinite(bpod_entry_s)
    valid_bpod_react = np.isfinite(bpod_react_s)
    valid_bpod_response = np.isfinite(bpod_response_s)
    if valid_bpod_react.any() or valid_bpod_entry.any() or missing_port_roles:
        from labdata.schema import StreamSync

        from thesis.ephys.events import bpod_to_clock

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
        to_clock = bpod_to_clock(subject, session)

        for port_role in missing_port_roles:
            entries_by_trial = []
            exits_by_trial = []
            for pokes in trial_table[f"{port_role.removesuffix('_port')}_poke"]:
                bpod_entries, bpod_exits = _split_bpod_pokes(pokes)
                entries_by_trial.append(
                    to_clock(bpod_entries).tolist() if bpod_entries.size else []
                )
                exits_by_trial.append(
                    to_clock(bpod_exits).tolist() if bpod_exits.size else []
                )
            trial_table[f"{port_role}_entry_times_s"] = entries_by_trial
            trial_table[f"{port_role}_exit_times_s"] = exits_by_trial

        predicted_react_s[valid_bpod_react] = to_clock(bpod_react_s[valid_bpod_react])
        predicted_entry_s[valid_bpod_entry] = to_clock(bpod_entry_s[valid_bpod_entry])
        if valid_bpod_response.any():
            predicted_response_s[valid_bpod_response] = to_clock(
                bpod_response_s[valid_bpod_response]
            )

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
        _select_task_ev_sequence(
            trial,
            predicted_react_s[trial_index],
            predicted_entry_s[trial_index],
            predicted_response_s[trial_index],
        )
        for trial_index, trial in trial_table.reset_index(drop=True).iterrows()
    ]
    trial_table = pd.concat(
        [trial_table.reset_index(drop=True), pd.DataFrame(task_ev_selections)], axis=1
    )
    trial_table["center_exit_bpod_error_s"] = (
        trial_table["center_exit_s"] - predicted_react_s
    )
    trial_table["response_entry_bpod_error_s"] = (
        trial_table["response_port_entry_s"] - predicted_response_s
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
            "last_side",
            "trial_start_s",
            "center_entry_s",
            "first_stim_times_s",
            "center_exit_s",
            "response_port_entry_s",
            "stim_pulse_times_s",
            "stim_pulse_widths_ms",
            "audio_stim_times_s",
            "go_cue_times_s",
            "punish_wrong_times_s",
            "punish_early_times_s",
            "frame_times_s",
            "left_port_entry_times_s",
            "left_port_exit_times_s",
            "center_port_entry_times_s",
            "center_port_exit_times_s",
            "right_port_entry_times_s",
            "right_port_exit_times_s",
            "n_task_ev_candidates",
            "center_exit_bpod_error_s",
            "response_entry_bpod_error_s",
            "response_entry_source",
        ]
    ]


@cached
def build_replay_table(subject: str, session: str) -> pd.DataFrame:
    """Return one row per passive replay trial with hardware events.

    Replay j is trial_start pulse N + j, where N is the number of task trials.
    Its window ends at the next pulse; the last one ends at the end of the
    recording. The replay plays the full intended train of its source task
    trial, so the rate and boundary come from that trial. Empty for sessions
    without replay.
    """
    from chipmunk import Chipmunk

    from thesis.ephys.events import fetch_session_events

    replays = _replay_trials(subject, session)
    columns = [
        "passive_trial_num",
        "source_trial_num",
        "visual_stim_rate_hz",
        "category_boundary_hz",
        "trial_start_s",
        "first_stim_times_s",
        "stim_pulse_times_s",
        "stim_pulse_widths_ms",
        "left_port_entry_times_s",
        "center_port_entry_times_s",
        "right_port_entry_times_s",
    ]
    if replays.empty:
        return pd.DataFrame(columns=columns)
    key = {"subject_name": subject, "session_name": session}
    source = (
        (Chipmunk.trial_query(**key).proj("stim_rate_vision", "category_boundary"))
        .fetch(format="frame")
        .reset_index()
    )
    sess_ev, stim_pulses = fetch_session_events(
        subject, session, allow_incomplete=True, include_frames=False
    )
    replay_index = _trial_start_indices(
        sess_ev["trial_start"], len(source), len(replays), subject, session
    )[len(source) :]
    bounds_s = np.r_[sess_ev["trial_start"], np.inf]
    windows = list(zip(bounds_s[replay_index], bounds_s[replay_index + 1]))
    table = replays.merge(
        source[["trial_num", "stim_rate_vision", "category_boundary"]].rename(
            columns={
                "trial_num": "source_trial_num",
                "stim_rate_vision": "visual_stim_rate_hz",
                "category_boundary": "category_boundary_hz",
            }
        ),
        on="source_trial_num",
        how="left",
        validate="many_to_one",
    )
    table["trial_start_s"] = bounds_s[replay_index]
    pulse_times_s = stim_pulses["timestamp"]
    table["stim_pulse_times_s"] = _split_by_windows(windows, pulse_times_s)
    table["stim_pulse_widths_ms"] = _split_by_windows(
        windows, pulse_times_s, stim_pulses["width_ms"]
    )
    table["first_stim_times_s"] = _split_by_windows(
        windows, stim_pulses.loc[stim_pulses["first_in_train"], "timestamp"]
    )
    for side in ("left", "center", "right"):
        table[f"{side}_port_entry_times_s"] = _split_by_windows(
            windows, sess_ev.get(f"{side}_port", [])
        )
    return table[columns]
