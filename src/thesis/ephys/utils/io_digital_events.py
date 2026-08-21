"""Digital behavioral events from labdata: TTL rows, semantic mapping, alignment dict.

**Naming convention (this module)**

- ``fetch_*`` — load rows from labdata / the analysis plugin, or return the full alignment dict.
- Other top-level functions — pure transforms on in-memory rows/arrays used when building
  ``fetch_session_events``.

For orchestration-only naming, ``load_session_align_event_arrays`` is an alias of
``fetch_session_events``.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from labdata.schema import (
    Dataset,
    DatasetEvents,
    SpikeSorting,
)

PortEventDict = Dict[str, Dict[str, np.ndarray]]
REQUIRED_LOGICAL_EVENTS = (
    "visual_stim",
    "trial_start",
    "frames",
    "left_port",
    "center_port",
    "right_port",
)


def format_available_event_rows(events: pd.DataFrame) -> list[str]:
    """Sorted unique `dataset:stream:event` keys from digital event rows (for errors)."""
    return sorted(
        {
            f"{row.dataset_name}:{row.stream_name}:{row.event_name}"
            for row in events[["dataset_name", "stream_name", "event_name"]].itertuples(
                index=False
            )
        }
    )


def fetch_digital_events_dataframe(subject: str, session: str) -> pd.DataFrame:
    """Load synced `DatasetEvents.Digital` rows for the session; raise if none."""
    sess_query = (
        SpikeSorting() & f'subject_name = "{subject}"' & f'session_name = "{session}"'
    ).proj()
    dset = (Dataset() & sess_query).proj()
    events = pd.DataFrame((DatasetEvents.Digital() & dset).fetch_synced())
    if events.empty:
        raise ValueError(f"No DatasetEvents.Digital rows found for {subject} {session}")
    return events


def fetch_event_mapping_dataframe(subject: str, session: str) -> pd.DataFrame:
    """Load `EventMapping` table rows for the session; raise if empty."""
    from labdata_plugin.schema import EventMapping

    mapping = pd.DataFrame(
        (
            EventMapping()
            & f'subject_name = "{subject}"'
            & f'session_name = "{session}"'
        ).fetch(as_dict=True)
    )
    if mapping.empty:
        raise ValueError(f"No EventMapping rows found for {subject} {session}")
    return mapping


def resolve_logical_event_rows(
    events: pd.DataFrame,
    mapping: pd.DataFrame,
    subject: str,
    session: str,
) -> dict[str, dict[str, np.ndarray | None]]:
    """Map each required logical event to timestamps and optional value array.

    Validates that all `REQUIRED_LOGICAL_EVENTS` exist in mapping, are unique,
    and each points to an existing digital event row.
    """
    mapped_event_names = mapping["event_name"].tolist()
    missing = [
        name for name in REQUIRED_LOGICAL_EVENTS if name not in mapped_event_names
    ]
    if missing:
        available_mappings = sorted(set(mapped_event_names))
        raise ValueError(
            f"Missing EventMapping rows for {subject} {session}: {missing}. "
            f"Available mapped names: {available_mappings}"
        )

    duplicates = mapping["event_name"][mapping["event_name"].duplicated()].unique()
    if duplicates.size:
        raise ValueError(
            f"Duplicate EventMapping rows for {subject} {session}: "
            f"{sorted(duplicates.tolist())}"
        )

    resolved: dict[str, dict[str, np.ndarray | None]] = {}
    available_rows = format_available_event_rows(events)
    for logical_name in REQUIRED_LOGICAL_EVENTS:
        row = mapping.loc[mapping["event_name"] == logical_name].iloc[0]
        mask = (
            (events["dataset_name"] == row["source_dataset_name"])
            & (events["stream_name"] == row["source_stream_name"])
            & (events["event_name"] == row["source_event_name"])
        )
        if not mask.any():
            raise ValueError(
                f"Mapped source row is missing for {subject} {session} "
                f"{logical_name}: {row['source_dataset_name']}:"
                f"{row['source_stream_name']}:{row['source_event_name']}. "
                f"Available rows: {available_rows}"
            )
        source_row = events.loc[mask].iloc[0]
        event_values = (
            None
            if "event_values" not in source_row.index
            or source_row["event_values"] is None
            else np.asarray(source_row["event_values"])
        )
        resolved[logical_name] = {
            "timestamps": np.asarray(source_row["event_timestamps"], dtype=float),
            "values": event_values,
        }
    return resolved


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
    events = fetch_digital_events_dataframe(subject, session)
    mapping = fetch_event_mapping_dataframe(subject, session)
    resolved = resolve_logical_event_rows(events, mapping, subject, session)

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


load_session_align_event_arrays = fetch_session_events
