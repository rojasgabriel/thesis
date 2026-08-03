from __future__ import annotations

from importlib import import_module
from typing import Any

import numpy as np

TRIALSET_DATASET_KEY_FIELDS = ("subject_name", "session_name", "dataset_name")
REQUIRED_NIDAQ_EVENTS = (
    "visual_stim",
    "trial_start",
    "left_port",
    "center_port",
    "right_port",
)
MAX_NIDAQ_ALIGNMENT_ERROR_S = 0.1


def fetch_pooled_kernel_inputs(
    trialset_keys: list[dict[str, Any]],
    trialset_description: str,
    *,
    observation_window: str,
) -> dict[str, Any]:
    """Fetch pooled trial inputs, preferring validated NIDAQ timing per session."""
    if observation_window not in {"center_exit", "response"}:
        raise ValueError(
            "observation_window must be 'center_exit' or 'response', "
            f"got {observation_window!r}"
        )

    session_inputs = []
    seen_datasets = set()
    for trialset_key in trialset_keys:
        dataset_key = {
            field: trialset_key[field] for field in TRIALSET_DATASET_KEY_FIELDS
        }
        dataset_identity = tuple(dataset_key.values())
        if dataset_identity in seen_datasets:
            continue
        seen_datasets.add(dataset_identity)

        trial_rows = _fetch_chipmunk_trial_rows(dataset_key)
        session_key = {
            field: dataset_key[field] for field in ("subject_name", "session_name")
        }
        mapping_rows = _fetch_event_mapping_rows(session_key)
        if has_nidaq_visual_timing(mapping_rows):
            event_rows = _fetch_mapped_digital_event_rows(session_key, mapping_rows)
            aligned_events = resolve_nidaq_event_arrays(
                event_rows,
                mapping_rows,
                session_key["subject_name"],
                session_key["session_name"],
            )
            inputs = extract_nidaq_kernel_inputs(
                aligned_events,
                trial_rows,
                trialset_description,
                observation_window=observation_window,
            )
            inputs["timing_source"] = "nidaq"
        else:
            inputs = extract_bpod_kernel_inputs(
                trial_rows,
                trialset_description,
                observation_window=observation_window,
            )
            inputs["timing_source"] = "bpod"
        session_inputs.append(inputs)

    if not session_inputs:
        raise ValueError("No selected Chipmunk trial sets were available")
    return combine_kernel_inputs(session_inputs)


def extract_bpod_kernel_inputs(
    trial_rows: list[dict[str, Any]],
    trialset_description: str,
    *,
    observation_window: str,
) -> dict[str, Any]:
    """Extract fixed-window inputs using native Bpod trial timestamps."""
    result = _new_kernel_inputs()
    end_field = "t_react" if observation_window == "center_exit" else "t_response"
    for row in sorted(trial_rows, key=lambda item: int(item["trial_num"])):
        if row["rewarded_modality"] != trialset_description:
            continue
        response = row["response"]
        observation_end = row.get(end_field)
        trial_sync = row.get("t_sync")
        stims = np.asarray(row.get("stim_events", []), dtype=float)
        stims = stims[np.isfinite(stims)]
        if (
            response not in (-1, 1)
            or observation_end is None
            or not np.isfinite(observation_end)
            or trial_sync is None
            or not np.isfinite(trial_sync)
            or stims.size == 0
        ):
            continue
        # Chipmunk stimulus events are relative to the Bpod sync pulse, while
        # state-transition timestamps are absolute within the session.
        observation_end = float(observation_end) - float(trial_sync)
        stims = stims[stims < float(observation_end)]
        if stims.size == 0 or observation_end <= stims[0]:
            continue
        _append_kernel_trial(result, stims, observation_end, row)
    return result


def extract_nidaq_kernel_inputs(
    aligned_events: dict[str, np.ndarray],
    trial_rows: list[dict[str, Any]],
    trialset_description: str,
    *,
    observation_window: str,
) -> dict[str, Any]:
    """Extract fixed-window inputs from NIDAQ events aligned to Bpod trials."""
    rows = sorted(trial_rows, key=lambda item: int(item["trial_num"]))
    trial_starts = np.asarray(aligned_events["trial_start"], dtype=float)
    if trial_starts.size == 0:
        raise ValueError("NIDAQ trial_start contains no rising edges")

    sync_rows = [
        row
        for row in rows
        if int(row["trial_num"]) < trial_starts.size
        and row.get("t_sync") is not None
        and np.isfinite(row["t_sync"])
    ]
    if len(sync_rows) < 2:
        raise ValueError(
            "Insufficient finite Bpod/NIDAQ sync points to interpolate trial timing"
        )
    bpod_sync = np.asarray([row["t_sync"] for row in sync_rows], dtype=float)
    nidaq_sync = np.asarray(
        [trial_starts[int(row["trial_num"])] for row in sync_rows], dtype=float
    )
    order = np.argsort(bpod_sync)
    bpod_sync = bpod_sync[order]
    nidaq_sync = nidaq_sync[order]

    stims = np.asarray(aligned_events["visual_stim"], dtype=float)
    center_exits = np.asarray(aligned_events["center_port_exit"], dtype=float)
    left_entries = np.asarray(aligned_events["left_port"], dtype=float)
    right_entries = np.asarray(aligned_events["right_port"], dtype=float)
    result = _new_kernel_inputs()

    for row in rows:
        if (
            row["rewarded_modality"] != trialset_description
            or row["response"] not in (-1, 1)
            or row.get("t_react") is None
            or not np.isfinite(row["t_react"])
            or row.get("t_sync") is None
            or not np.isfinite(row["t_sync"])
        ):
            continue
        bpod_stims = np.asarray(row.get("stim_events", []), dtype=float)
        bpod_stims = bpod_stims[np.isfinite(bpod_stims)]
        if bpod_stims.size == 0:
            continue
        trial_number = int(row["trial_num"])
        if trial_number >= trial_starts.size:
            continue
        trial_start = trial_starts[trial_number]
        trial_end = (
            trial_starts[trial_number + 1]
            if trial_number + 1 < trial_starts.size
            else np.inf
        )
        interpolated_first_stim = float(
            np.interp(
                float(row["t_sync"]) + float(bpod_stims[0]),
                bpod_sync,
                nidaq_sync,
            )
        )
        interpolated_exit = float(
            np.interp(float(row["t_react"]), bpod_sync, nidaq_sync)
        )
        trial_exits = center_exits[
            (center_exits > trial_start) & (center_exits < trial_end)
        ]
        center_exit = _nearest_aligned_event(trial_exits, interpolated_exit)
        if center_exit is None:
            continue

        observation_end = center_exit
        if observation_window == "response":
            response_time = row.get("t_response")
            if response_time is None or not np.isfinite(response_time):
                continue
            interpolated_response = float(
                np.interp(float(response_time), bpod_sync, nidaq_sync)
            )
            response_entries = (
                right_entries if int(row["response"]) == 1 else left_entries
            )
            trial_responses = response_entries[
                (response_entries > center_exit) & (response_entries < trial_end)
            ]
            response_entry = _nearest_aligned_event(
                trial_responses, interpolated_response
            )
            if response_entry is None:
                continue
            observation_end = response_entry

        trial_stims = stims[
            (stims >= trial_start) & (stims < observation_end) & (stims < trial_end)
        ]
        if trial_stims.size == 0:
            continue
        # Use the Bpod schedule only to identify the first hardware-timed flash.
        first_stim = _nearest_aligned_event(trial_stims, interpolated_first_stim)
        if first_stim is None:
            continue
        trial_stims = trial_stims[trial_stims >= first_stim]
        _append_kernel_trial(result, trial_stims, observation_end, row)
    return result


def has_nidaq_visual_timing(mapping_rows: list[dict[str, Any]]) -> bool:
    """Return whether a session declares a mapped NIDAQ/OneBox visual stream."""
    return any(row.get("event_name") == "visual_stim" for row in mapping_rows)


def resolve_nidaq_event_arrays(
    event_rows: list[dict[str, Any]],
    mapping_rows: list[dict[str, Any]],
    subject: str,
    session: str,
) -> dict[str, np.ndarray]:
    """Resolve and validate logical NIDAQ event arrays for one session."""
    mapped_names = [row["event_name"] for row in mapping_rows]
    missing = [name for name in REQUIRED_NIDAQ_EVENTS if name not in mapped_names]
    if missing:
        raise ValueError(
            f"Missing EventMapping rows for {subject} {session}: {missing}"
        )
    duplicates = sorted({name for name in mapped_names if mapped_names.count(name) > 1})
    if duplicates:
        raise ValueError(
            f"Duplicate EventMapping rows for {subject} {session}: {duplicates}"
        )

    resolved = {}
    for logical_name in REQUIRED_NIDAQ_EVENTS:
        mapping = next(row for row in mapping_rows if row["event_name"] == logical_name)
        matches = [
            row
            for row in event_rows
            if row["dataset_name"] == mapping["source_dataset_name"]
            and row["stream_name"] == mapping["source_stream_name"]
            and row["event_name"] == mapping["source_event_name"]
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one mapped NIDAQ row for {subject} {session} "
                f"{logical_name}; found {len(matches)}"
            )
        timestamps = np.asarray(matches[0]["event_timestamps"], dtype=float)
        if timestamps.size == 0 or np.any(~np.isfinite(timestamps)):
            raise ValueError(
                f"Mapped NIDAQ row is empty or nonfinite for {subject} {session} "
                f"{logical_name}"
            )
        values = matches[0].get("event_values")
        if values is not None:
            values = np.asarray(values)
            if values.shape != timestamps.shape:
                raise ValueError(
                    f"NIDAQ event values do not match timestamps for "
                    f"{subject} {session} {logical_name}"
                )
        resolved[logical_name] = (timestamps, values)

    visual_stim = _merge_visual_stim_edges(resolved["visual_stim"][0])
    if visual_stim.size == 0:
        raise ValueError(f"No visual flashes found for {subject} {session}")
    return {
        "visual_stim": visual_stim,
        "trial_start": _digital_onsets(*resolved["trial_start"]),
        "left_port": _port_entries(*resolved["left_port"]),
        "left_port_exit": _port_exits(*resolved["left_port"]),
        "center_port": _port_entries(*resolved["center_port"]),
        "center_port_exit": _port_exits(*resolved["center_port"]),
        "right_port": _port_entries(*resolved["right_port"]),
        "right_port_exit": _port_exits(*resolved["right_port"]),
    }


def combine_kernel_inputs(session_inputs: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine per-session inputs and summarize their timing provenance."""
    sources = {inputs["timing_source"] for inputs in session_inputs}
    timing_source = next(iter(sources)) if len(sources) == 1 else "mixed"
    return {
        "stim_times_per_trial": [
            stims
            for inputs in session_inputs
            for stims in inputs["stim_times_per_trial"]
        ],
        "first_stim_times": _concatenate(
            session_inputs, "first_stim_times", dtype=float
        ),
        "observation_end_times": _concatenate(
            session_inputs, "observation_end_times", dtype=float
        ),
        "response_values": _concatenate(session_inputs, "response_values", dtype=int),
        "trial_rate_hz": _concatenate(session_inputs, "trial_rate_hz", dtype=float),
        "timing_source": timing_source,
    }


def _fetch_chipmunk_trial_rows(dataset_key: dict[str, Any]) -> list[dict[str, Any]]:
    from behavior_analyses.io import get_chipmunk_table

    Chipmunk = get_chipmunk_table()
    relation = Chipmunk() * Chipmunk.Trial() * Chipmunk.TrialParameters() & dataset_key
    return list(
        relation.fetch(
            "trial_num",
            "rewarded_modality",
            "stim_events",
            "stim_rate_vision",
            "response",
            "t_sync",
            "t_react",
            "t_response",
            as_dict=True,
            order_by="trial_num",
        )
    )


def _fetch_event_mapping_rows(
    session_key: dict[str, Any],
) -> list[dict[str, Any]]:
    try:
        import labdata

        labdata.plugins["gephys"].__file__
        module = import_module("gephys.analysisschema")
    except KeyError:
        return []
    except ModuleNotFoundError as error:
        if error.name in {"gephys", "gephys.analysisschema"}:
            return []
        raise
    return list((module.EventMapping() & session_key).fetch(as_dict=True))


def _fetch_mapped_digital_event_rows(
    session_key: dict[str, Any],
    mapping_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    from labdata.schema import DatasetEvents

    source_keys = [
        {
            **session_key,
            "dataset_name": row["source_dataset_name"],
            "stream_name": row["source_stream_name"],
            "event_name": row["source_event_name"],
        }
        for row in mapping_rows
        if row["event_name"] in REQUIRED_NIDAQ_EVENTS
    ]
    return list((DatasetEvents.Digital() & source_keys).fetch_synced())


def _new_kernel_inputs() -> dict[str, Any]:
    return {
        "stim_times_per_trial": [],
        "first_stim_times": [],
        "observation_end_times": [],
        "response_values": [],
        "trial_rate_hz": [],
    }


def _append_kernel_trial(
    result: dict[str, Any],
    stims: np.ndarray,
    observation_end: float,
    row: dict[str, Any],
) -> None:
    rate = row.get("stim_rate_vision")
    if rate is None or not np.isfinite(rate):
        raise ValueError(f"Trial {row['trial_num']} has no finite visual stimulus rate")
    result["stim_times_per_trial"].append(np.asarray(stims, dtype=float))
    result["first_stim_times"].append(float(stims[0]))
    result["observation_end_times"].append(float(observation_end))
    result["response_values"].append(int(row["response"]))
    result["trial_rate_hz"].append(float(rate))


def _nearest_aligned_event(events: np.ndarray, target: float) -> float | None:
    if events.size == 0:
        return None
    event = float(events[np.argmin(np.abs(events - target))])
    if abs(event - target) > MAX_NIDAQ_ALIGNMENT_ERROR_S:
        return None
    return event


def _digital_onsets(timestamps: np.ndarray, values: np.ndarray | None) -> np.ndarray:
    return timestamps[::2] if values is None else timestamps[values == 1]


def _port_entries(timestamps: np.ndarray, values: np.ndarray | None) -> np.ndarray:
    return timestamps if values is None else timestamps[values == 1]


def _port_exits(timestamps: np.ndarray, values: np.ndarray | None) -> np.ndarray:
    return np.array([], dtype=float) if values is None else timestamps[values == 0]


def _merge_visual_stim_edges(timestamps: np.ndarray) -> np.ndarray:
    timestamps = np.sort(np.asarray(timestamps, dtype=float))
    if timestamps.size == 0:
        return timestamps
    split_indices = np.where(np.diff(timestamps) > 0.020)[0] + 1
    return np.asarray([burst[0] for burst in np.split(timestamps, split_indices)])


def _concatenate(
    session_inputs: list[dict[str, Any]], field: str, *, dtype: type
) -> np.ndarray:
    values = [np.asarray(inputs[field], dtype=dtype) for inputs in session_inputs]
    return np.concatenate(values) if values else np.empty((0,), dtype=dtype)
