"""Behavioral event timing from measured ephys inputs or synchronized Bpod."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

# Silence the setuptools pkg_resources deprecation notice
warnings.filterwarnings("ignore", category=UserWarning, module="datajoint.plugin")

REQUIRED_EV_ROLES = (
    "visual_stim",
    "trial_start",
    "frames",
    "left_port",
    "center_port",
    "right_port",
)
CORE_EV_ROLES = ("visual_stim", "trial_start")
EV_STREAM_PRIORITY = ("obx", "nidq")
AUDIO_EVENT_ROLES = ("audio_stim", "go_cue", "punish_wrong", "punish_early")


def classify_audio_events(
    onsets: np.ndarray, offsets: np.ndarray
) -> dict[str, np.ndarray]:
    """Group audio epoch onsets by task role using epoch duration."""
    onsets = np.asarray(onsets, dtype=float)
    offsets = np.asarray(offsets, dtype=float)
    if onsets.ndim != 1 or onsets.shape != offsets.shape:
        raise ValueError("Audio onsets and offsets must be equal-length 1-D arrays")
    if np.any(offsets <= onsets):
        raise ValueError("Each audio offset must follow its onset")

    durations = offsets - onsets
    masks = {
        "audio_stim": (durations >= 0.015) & (durations < 0.050),
        "go_cue": (durations >= 0.050) & (durations <= 0.250),
        "punish_wrong": (durations >= 0.750) & (durations <= 1.250),
        "punish_early": (durations >= 1.750) & (durations <= 2.250),
    }
    known = np.logical_or.reduce(list(masks.values()))
    return {
        **{name: onsets[mask] for name, mask in masks.items()},
        "unknown": onsets[~known],
    }


def _audio_epochs_from_event_row(event: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return paired audio onsets and offsets from a digital-event row."""
    timestamps = np.asarray(event["event_timestamps"], dtype=float)
    values = event.get("event_values")
    if values is None:
        if len(timestamps) % 2:
            raise ValueError("Audio event timestamps do not contain onset-offset pairs")
        onsets, offsets = timestamps[::2], timestamps[1::2]
    else:
        values = np.asarray(values)
        if values.shape != timestamps.shape or not np.all(np.isin(values, (0, 1))):
            raise ValueError(
                "Audio event values must be timestamp-matched binary states"
            )
        onsets, offsets = timestamps[values == 1], timestamps[values == 0]
    if len(onsets) != len(offsets) or np.any(offsets <= onsets):
        raise ValueError("Audio event onsets and offsets are not paired")
    return onsets, offsets


def _bpod_sound_events(trials: list[dict]) -> dict[str, np.ndarray]:
    """Return software-command sound times from Chipmunk trial rows."""
    specs = {
        "go_cue": ("t_gocue", None),
        "punish_wrong": ("t_response", "punished"),
        "punish_early": ("t_earlywithdraw", "early_withdrawal"),
    }
    auditory_stim_times = []
    for trial in trials:
        if trial["rewarded_modality"] not in ("audio", "visual+audio"):
            continue
        if trial["t_sync"] is None or not np.isfinite(trial["t_sync"]):
            continue
        stim_events = np.atleast_1d(np.asarray(trial["stim_events"], dtype=float))
        auditory_stim_times.extend(
            trial["t_sync"] + stim_events[np.isfinite(stim_events)]
        )

    events = {"audio_stim": np.asarray(auditory_stim_times)}
    for role, (time_field, flag_field) in specs.items():
        events[role] = np.asarray(
            [
                float(trial[time_field])
                for trial in trials
                if trial[time_field] is not None
                and np.isfinite(trial[time_field])
                and (flag_field is None or trial[flag_field])
            ]
        )
    return events


def _fetch_audio_events(
    subject: str,
    session: str,
) -> dict[str, np.ndarray]:
    """Return task-sound onsets in the ephys clock when available.

    Measured ``obx/io1`` events are preferred. If they are absent, Bpod sound
    command times are transformed through the session's Bpod StreamSync row.
    The Bpod fallback does not verify speaker output or acoustic latency.
    """
    from labdata.schema import DatasetEvents, EphysRecording, StreamSync

    restriction = {"subject_name": subject, "session_name": session}
    measured = (
        DatasetEvents.Digital()
        & (EphysRecording() & restriction)
        & {"stream_name": "obx", "event_name": "io1"}
    )
    if len(measured) > 1:
        raise ValueError(f"Multiple measured audio streams found for {restriction}")
    if len(measured) == 1:
        onsets, offsets = _audio_epochs_from_event_row(measured.fetch1())
        if not len(onsets):
            raise ValueError(f"Measured audio stream is empty for {restriction}")
        classified = classify_audio_events(onsets, offsets)
        if len(classified["unknown"]) / len(onsets) > 0.05:
            raise ValueError(
                f"Measured audio events failed validation for {restriction}"
            )
        return {
            role: np.asarray(classified[role], dtype=float)
            for role in AUDIO_EVENT_ROLES
        }

    bpod_sync = StreamSync() & {
        **restriction,
        "dataset_name": "chipmunk",
        "stream_name": "bpod",
        "event_name": "sync",
    }
    if len(bpod_sync) != 1:
        warnings.warn(
            f"Audio event timing unavailable for {subject} {session}: "
            "no measured obx/io1 or Bpod StreamSync",
            RuntimeWarning,
            stacklevel=3,
        )
        return {}

    from chipmunk import Chipmunk

    trials = (
        Chipmunk.trial_query(**restriction)
        .proj(
            "t_gocue",
            "t_response",
            "t_earlywithdraw",
            "punished",
            "early_withdrawal",
            "t_sync",
            "stim_events",
            "rewarded_modality",
        )
        .fetch(as_dict=True, order_by="trial_num")
    )
    bpod_events = _bpod_sound_events(trials)
    warnings.warn(
        f"No measured obx/io1 for {subject} {session}; "
        "using synchronized Bpod sound-command times",
        RuntimeWarning,
        stacklevel=3,
    )
    return {
        role: (
            np.asarray(bpod_sync.apply(timestamps, force=True, warn=False), dtype=float)
            if len(timestamps)
            else np.array([])
        )
        for role, timestamps in bpod_events.items()
    }


def _find_sess_ev_sources(
    subject: str,
    sess: str,
    *,
    allow_incomplete: bool = False,
) -> dict[str, dict[str, str]]:
    """Return source keys for the known event set in one ephys recording."""
    from labdata.schema import DatasetEvents, EphysRecording

    from labdata_plugin.schema import EventMapping

    sess_restriction = {"subject_name": subject, "session_name": sess}
    available_ev_sources = {
        (
            digital_ev["dataset_name"],
            digital_ev["stream_name"],
            digital_ev["event_name"],
        )
        for digital_ev in (
            DatasetEvents.Digital() & (EphysRecording() & sess_restriction)
        ).fetch("dataset_name", "stream_name", "event_name", as_dict=True)
    }
    ev_mappings = list(EventMapping().fetch(as_dict=True))
    required_roles = CORE_EV_ROLES if allow_incomplete else REQUIRED_EV_ROLES

    for stream_name in EV_STREAM_PRIORITY:
        ev_names_by_role = {
            ev_mapping["event_role"]: ev_mapping["event_name"]
            for ev_mapping in ev_mappings
            if ev_mapping["stream_name"] == stream_name
            and ev_mapping["event_role"] in REQUIRED_EV_ROLES
        }
        if not set(required_roles) <= set(ev_names_by_role):
            continue
        candidate_dsets = {
            dset_name
            for dset_name, ev_stream, _ in available_ev_sources
            if ev_stream == stream_name
        }
        matching_dsets = [
            dset_name
            for dset_name in candidate_dsets
            if all(
                (dset_name, stream_name, ev_names_by_role[ev_role])
                in available_ev_sources
                for ev_role in required_roles
            )
        ]
        if len(matching_dsets) == 1:
            return {
                ev_role: {
                    **sess_restriction,
                    "dataset_name": matching_dsets[0],
                    "stream_name": stream_name,
                    "event_name": ev_name,
                }
                for ev_role, ev_name in ev_names_by_role.items()
                if (matching_dsets[0], stream_name, ev_name) in available_ev_sources
            }
        if len(matching_dsets) > 1:
            raise ValueError(
                f"Multiple ephys datasets contain the required events for "
                f"{subject} {sess}: {matching_dsets}"
            )

    raise ValueError(f"No complete ephys event set found for {subject} {sess}")


def _build_stimulus_pulses(stim_edges: np.ndarray) -> pd.DataFrame:
    """Merge raw stimulus TTL edges into one row per pulse."""
    stim_edges = np.asarray(stim_edges, dtype=float)

    max_within_pulse_gap_s = 0.020
    if stim_edges.size > 0:
        sorted_stim_edges = np.sort(stim_edges)
        pulse_boundaries = (
            np.where(np.diff(sorted_stim_edges) > max_within_pulse_gap_s)[0] + 1
        )
        pulse_edge_groups = np.split(sorted_stim_edges, pulse_boundaries)

        pulse_onsets = np.array([pulse_edges[0] for pulse_edges in pulse_edge_groups])
        pulse_durations_s = np.array(
            [pulse_edges[-1] - pulse_edges[0] for pulse_edges in pulse_edge_groups]
        )
    else:
        pulse_onsets = np.array([])
        pulse_durations_s = np.array([])

    width_tolerance_s = 2e-3
    if pulse_durations_s.size and np.allclose(pulse_durations_s, 0.0):
        # Historical GRB006 repairs insert onset-only visual events instead of
        # raw TTL edges, so treat the mapped row as a 15 ms-only stim stream.
        pulse_widths_ms = np.full(pulse_durations_s.shape, 15.0)
    else:
        distance_from_15_ms_s = np.abs(pulse_durations_s - 0.015)
        distance_from_30_ms_s = np.abs(pulse_durations_s - 0.030)
        is_15_ms = distance_from_15_ms_s <= width_tolerance_s
        is_30_ms = distance_from_30_ms_s <= width_tolerance_s
        pulse_widths_ms = np.where(is_15_ms, 15.0, np.where(is_30_ms, 30.0, np.nan))

    is_first_in_train = (
        np.r_[True, np.diff(pulse_onsets) > 1.0]
        if pulse_onsets.size
        else np.array([], dtype=bool)
    )
    return pd.DataFrame(
        {
            "timestamp": pulse_onsets,
            "width_ms": pulse_widths_ms,
            "first_in_train": is_first_in_train,
        }
    )


def fetch_session_events(
    subject: str,
    session: str,
    *,
    allow_incomplete: bool = False,
    include_frames: bool = True,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Fetch digital event arrays and the processed stimulus-pulse table.

    Audio roles are included when timing is available. Measured ``obx/io1``
    events are preferred; otherwise synchronized Bpod command times are used.
    If neither source exists, audio roles are omitted and a warning is emitted.

    Raw digital edges on the stim channel are noisy: a single logical pulse
    toggles many times. They are merged into discrete bursts by splitting on
    any gap > 20 ms, then each burst's duration is classified against the two
    expected pulse widths (15 ms and 30 ms, ±2 ms tolerance). Bursts that
    match neither are labeled "unknown" and excluded from the width-specific
    streams but remain in the pulse table with a missing `width_ms` value.

    The pulse table has `timestamp`, `width_ms`, and `first_in_train` columns.
    `first_in_train` marks the first pulse after a gap longer than one second.
    With `allow_incomplete=True`, stimulus and trial-start events remain
    required, but missing frame and port roles are omitted from `sess_ev`.
    Set `include_frames=False` when screen-frame timestamps are not needed.
    """
    from labdata.schema import DatasetEvents

    ev_sources = _find_sess_ev_sources(
        subject, session, allow_incomplete=allow_incomplete
    )
    if not include_frames:
        ev_sources.pop("frames", None)
    stim_source = ev_sources["visual_stim"]
    digital_ev_records = [
        {
            **stim_source,
            "event_timestamps": (DatasetEvents.Digital() & stim_source).fetch1(
                "event_timestamps"
            ),
        }
    ]
    other_sources = [
        source for role, source in ev_sources.items() if role != "visual_stim"
    ]
    if other_sources:
        digital_ev_records.extend(
            (DatasetEvents.Digital() & other_sources).fetch(as_dict=True)
        )
    digital_ev_by_source = {
        (
            digital_ev["dataset_name"],
            digital_ev["stream_name"],
            digital_ev["event_name"],
        ): digital_ev
        for digital_ev in digital_ev_records
    }
    ev_data_by_role = {}
    for ev_role, ev_source in ev_sources.items():
        digital_ev = digital_ev_by_source[
            (
                ev_source["dataset_name"],
                ev_source["stream_name"],
                ev_source["event_name"],
            )
        ]
        ev_timestamps = np.asarray(digital_ev["event_timestamps"], dtype=float)
        ev_values = digital_ev.get("event_values")
        ev_values = None if ev_values is None else np.asarray(ev_values)
        if ev_values is not None and ev_values.shape != ev_timestamps.shape:
            raise ValueError(f"Event values do not match timestamps for {ev_role}")
        ev_data_by_role[ev_role] = {
            "timestamps": ev_timestamps,
            "values": ev_values,
        }

    sess_ev: dict[str, np.ndarray] = {
        "stim": np.asarray(ev_data_by_role["visual_stim"]["timestamps"], dtype=float),
    }
    for ev_role in ("trial_start", "frames"):
        if ev_role not in ev_data_by_role:
            continue
        ev_timestamps = np.asarray(ev_data_by_role[ev_role]["timestamps"], dtype=float)
        ev_values = ev_data_by_role[ev_role]["values"]
        sess_ev[ev_role] = (
            ev_timestamps[::2] if ev_values is None else ev_timestamps[ev_values == 1]
        )

    for port_role in ("left_port", "center_port", "right_port"):
        if port_role not in ev_data_by_role:
            continue
        ev_timestamps = np.asarray(
            ev_data_by_role[port_role]["timestamps"], dtype=float
        )
        ev_values = ev_data_by_role[port_role]["values"]
        if ev_values is None:
            sess_ev[port_role] = ev_timestamps
            sess_ev[f"{port_role}_exit"] = np.array([])
        else:
            sess_ev[port_role] = ev_timestamps[ev_values == 1]
            sess_ev[f"{port_role}_exit"] = ev_timestamps[ev_values == 0]

    sess_ev.update(_fetch_audio_events(subject, session))
    return sess_ev, _build_stimulus_pulses(sess_ev["stim"])


def _self_check() -> None:
    onsets, offsets = _audio_epochs_from_event_row(
        {"event_timestamps": [1.0, 1.1, 2.0, 3.0], "event_values": None}
    )
    np.testing.assert_allclose(onsets, [1.0, 2.0])
    np.testing.assert_allclose(offsets, [1.1, 3.0])

    events = _bpod_sound_events(
        [
            {
                "t_gocue": 1.0,
                "t_response": 2.0,
                "t_earlywithdraw": None,
                "punished": True,
                "early_withdrawal": False,
                "t_sync": 0.5,
                "stim_events": [0.1, 0.3],
                "rewarded_modality": "audio",
            },
            {
                "t_gocue": None,
                "t_response": None,
                "t_earlywithdraw": 3.0,
                "punished": False,
                "early_withdrawal": True,
                "t_sync": 2.0,
                "stim_events": [0.2],
                "rewarded_modality": "visual",
            },
        ]
    )
    np.testing.assert_allclose(events["audio_stim"], [0.6, 0.8])
    np.testing.assert_allclose(events["go_cue"], [1.0])
    np.testing.assert_allclose(events["punish_wrong"], [2.0])
    np.testing.assert_allclose(events["punish_early"], [3.0])


if __name__ == "__main__":
    _self_check()
