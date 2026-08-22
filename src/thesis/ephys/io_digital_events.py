"""Digital behavioral events from labdata: TTL rows and alignment arrays.

**Naming convention (this module)**

- ``fetch_*`` — return the full alignment dict.
- Other top-level functions — pure transforms on event arrays.

"""

from __future__ import annotations

import warnings

import numpy as np

# Silence the setuptools pkg_resources deprecation notice
warnings.filterwarnings("ignore", category=UserWarning, module="datajoint.plugin")

import datajoint as dj  # noqa: E402

REQUIRED_LOGICAL_EVENTS = (
    "visual_stim",
    "trial_start",
    "frames",
    "left_port",
    "center_port",
    "right_port",
)
STREAM_PRIORITY = ("obx", "nidq")


def session_event_keys(
    subject: str,
    session: str,
) -> dict[str, dict[str, str]]:
    """Return source keys for the known event set in one ephys recording."""
    from labdata.schema import DatasetEvents, EphysRecording, get_user_schema

    restriction = {"subject_name": subject, "session_name": session}
    available = {
        (row["dataset_name"], row["stream_name"], row["event_name"])
        for row in (DatasetEvents.Digital() & (EphysRecording() & restriction)).fetch(
            "dataset_name", "stream_name", "event_name", as_dict=True
        )
    }
    user_schema = get_user_schema()
    event_mapping = dj.FreeTable(
        user_schema.connection,
        f"`{user_schema.database}`.`#event_mapping`",
    )
    mapping_rows = list(event_mapping.fetch(as_dict=True))

    for stream_name in STREAM_PRIORITY:
        mapping = {
            row["event_role"]: row["event_name"]
            for row in mapping_rows
            if row["stream_name"] == stream_name
            and row["event_role"] in REQUIRED_LOGICAL_EVENTS
        }
        if set(mapping) != set(REQUIRED_LOGICAL_EVENTS):
            continue
        datasets = {
            dataset_name
            for dataset_name, source_stream, _ in available
            if source_stream == stream_name
        }
        matches = [
            dataset_name
            for dataset_name in datasets
            if all(
                (dataset_name, stream_name, event_name) in available
                for event_name in mapping.values()
            )
        ]
        if len(matches) == 1:
            return {
                role: {
                    **restriction,
                    "dataset_name": matches[0],
                    "stream_name": stream_name,
                    "event_name": event_name,
                }
                for role, event_name in mapping.items()
            }
        if len(matches) > 1:
            raise ValueError(
                f"Multiple ephys datasets contain the required events for "
                f"{subject} {session}: {matches}"
            )

    raise ValueError(f"No complete ephys event set found for {subject} {session}")


def has_session_events(subject: str, session: str) -> bool:
    """Return whether one ephys dataset has the known event set."""
    try:
        session_event_keys(subject, session)
    except ValueError:
        return False
    return True


def extract_digital_onsets(event_row: dict[str, np.ndarray | None]) -> np.ndarray:
    """Rising-edge onsets: timestamps where value==1, or every other if no values."""
    timestamps = np.asarray(event_row["timestamps"], dtype=float)
    values = event_row["values"]
    if values is None:
        return timestamps[::2]
    values = np.asarray(values)
    return timestamps[values == 1]


def extract_port_poke_onsets(event_row: dict[str, np.ndarray | None]) -> np.ndarray:
    """Port poke onsets: value==1, or all timestamps if no value column."""
    timestamps = np.asarray(event_row["timestamps"], dtype=float)
    values = event_row["values"]
    if values is None:
        return timestamps
    values = np.asarray(values)
    return timestamps[values == 1]


def extract_port_poke_exits(event_row: dict[str, np.ndarray | None]) -> np.ndarray:
    """Port exit times (value==0); empty array if no `event_values` on the row."""
    timestamps = np.asarray(event_row["timestamps"], dtype=float)
    values = event_row["values"]
    if values is None:
        return np.array([])
    values = np.asarray(values)
    return timestamps[values == 0]


def first_timestamp_per_train(ev_array: np.ndarray, sep_s: float = 1.0) -> np.ndarray:
    """Keep the first timestamp in each run separated by more than `sep_s` seconds."""
    if ev_array.size == 0:
        return ev_array
    keep = np.r_[True, np.diff(ev_array) > sep_s]
    return ev_array[keep]


def derive_merged_stim_pulse_arrays(stim: np.ndarray) -> dict[str, np.ndarray]:
    """Merge raw stim TTL edges into bursts, label 15/30 ms, build stim_ev* arrays.

    See ``fetch_session_events`` for key meanings (`stim_ev`, `first_stim_ev_*`, etc.).
    """
    stim = np.asarray(stim, dtype=float)

    max_within_pulse_gap_s = 0.020
    if stim.size > 0:
        stim_sorted = np.sort(stim)
        split_idx = np.where(np.diff(stim_sorted) > max_within_pulse_gap_s)[0] + 1
        bursts = np.split(stim_sorted, split_idx)

        onsets = np.array([b[0] for b in bursts])
        durations = np.array([b[-1] - b[0] for b in bursts])
    else:
        onsets = np.array([])
        durations = np.array([])

    tol_s = 2e-3
    if durations.size and np.allclose(durations, 0.0):
        # Historical GRB006 repairs insert onset-only visual events instead of
        # raw TTL edges, so treat the mapped row as a 15 ms-only stim stream.
        labels = np.full(durations.shape, "15ms", dtype=object)
    else:
        diff_15 = np.abs(durations - 0.015)
        diff_30 = np.abs(durations - 0.030)
        is_15 = diff_15 <= tol_s
        is_30 = diff_30 <= tol_s
        labels = np.where(is_15, "15ms", np.where(is_30, "30ms", "unknown"))

    stim_ev = onsets
    stim_ev_15ms = onsets[labels == "15ms"]
    stim_ev_30ms = onsets[labels == "30ms"]

    return {
        "stim_ev": stim_ev,
        "first_stim_ev": first_timestamp_per_train(stim_ev),
        "stim_ev_15ms": stim_ev_15ms,
        "stim_ev_30ms": stim_ev_30ms,
        "first_stim_ev_15ms": first_timestamp_per_train(stim_ev_15ms),
        "first_stim_ev_30ms": first_timestamp_per_train(stim_ev_30ms),
    }


def fetch_session_events(
    subject: str,
    session: str,
) -> dict[str, np.ndarray]:
    """Fetch digital events for a session and derive stimulus event arrays.

    Raw digital edges on the stim channel are noisy: a single logical pulse
    toggles many times. They are merged into discrete bursts by splitting on
    any gap > 20 ms, then each burst's duration is classified against the two
    expected pulse widths (15 ms and 30 ms, ±2 ms tolerance). Bursts that
    match neither are labeled "unknown" and excluded from the width-specific
    streams but still appear in `stim_ev` / `first_stim_ev`.

    `first_*` variants keep only the first onset within each 1 s window, so
    they approximate the first pulse of each stimulus train — the cleaner
    alignment event when comparing single-pulse responses (e.g. for the
    double-peak analysis).

    Returns a dict with the following keys (all np.ndarray of timestamps in
    seconds, possibly empty):
      - `stim`              : raw stim edges (no merging)
      - `trial_start`, `frames`, `left_port`, `center_port`, `right_port`
      - `left_port_exit`, `center_port_exit`, `right_port_exit` when DB
        `event_values` are available
      - `stim_ev`           : onsets of all merged pulses (any width)
      - `first_stim_ev`     : first-of-train onsets across all widths
      - `stim_ev_15ms`      : onsets of pulses classified as 15 ms
      - `stim_ev_30ms`      : onsets of pulses classified as 30 ms
      - `first_stim_ev_15ms`: first-of-train onsets for 15 ms pulses only
      - `first_stim_ev_30ms`: first-of-train onsets for 30 ms pulses only
    """
    from labdata.schema import DatasetEvents

    source_keys = session_event_keys(subject, session)
    rows = list(
        (DatasetEvents.Digital() & list(source_keys.values())).fetch(as_dict=True)
    )
    rows_by_key = {
        (row["dataset_name"], row["stream_name"], row["event_name"]): row
        for row in rows
    }
    resolved = {}
    for role, key in source_keys.items():
        row = rows_by_key[(key["dataset_name"], key["stream_name"], key["event_name"])]
        timestamps = np.asarray(row["event_timestamps"], dtype=float)
        values = row.get("event_values")
        values = None if values is None else np.asarray(values)
        if values is not None and values.shape != timestamps.shape:
            raise ValueError(f"Event values do not match timestamps for {role}")
        resolved[role] = {"timestamps": timestamps, "values": values}

    # Trial-start TTLs still arrive as on/off edge pairs, regardless of the
    # physical source row used for this session.
    trial_start = extract_digital_onsets(resolved["trial_start"])
    align_ev: dict[str, np.ndarray] = {
        "stim": np.asarray(resolved["visual_stim"]["timestamps"], dtype=float),
        "trial_start": trial_start,
        "frames": extract_digital_onsets(resolved["frames"]),
        "left_port": extract_port_poke_onsets(resolved["left_port"]),
        "center_port": extract_port_poke_onsets(resolved["center_port"]),
        "right_port": extract_port_poke_onsets(resolved["right_port"]),
        "left_port_exit": extract_port_poke_exits(resolved["left_port"]),
        "center_port_exit": extract_port_poke_exits(resolved["center_port"]),
        "right_port_exit": extract_port_poke_exits(resolved["right_port"]),
    }
    align_ev.update(derive_merged_stim_pulse_arrays(align_ev["stim"]))
    return align_ev
