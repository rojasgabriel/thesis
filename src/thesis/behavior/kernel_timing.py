from __future__ import annotations

from typing import Any

import numpy as np

TRIALSET_DATASET_KEY_FIELDS = ("subject_name", "session_name", "dataset_name")
TIMING_SOURCES = ("nidq", "bpod")
MAX_NIDAQ_ALIGNMENT_ERROR_S = 0.1


def available_timing_sources(trialset_keys: list[dict[str, Any]]) -> list[str]:
    """Return timing sources the selected trial sets can supply."""
    if not trialset_keys:
        return []
    session_keys = {(key["subject_name"], key["session_name"]) for key in trialset_keys}
    sources = ["bpod"]
    from thesis.ephys.utils.io_digital_events import has_session_events

    if all(has_session_events(subject, session) for subject, session in session_keys):
        sources.insert(0, "nidq")
    return sources


def fetch_pooled_kernel_inputs(
    trialset_keys: list[dict[str, Any]],
    trialset_description: str,
    *,
    observation_window: str,
    timing_source: str,
) -> dict[str, Any]:
    """Fetch pooled trial inputs from one requested timing source."""
    if observation_window not in {"center_exit", "response"}:
        raise ValueError(
            "observation_window must be 'center_exit' or 'response', "
            f"got {observation_window!r}"
        )
    if timing_source not in TIMING_SOURCES:
        raise ValueError(
            f"timing_source must be one of {TIMING_SOURCES}, got {timing_source!r}"
        )
    if timing_source not in available_timing_sources(trialset_keys):
        raise ValueError(
            f"Selected trial sets cannot supply timing_source={timing_source!r}"
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
        if timing_source == "nidq":
            from thesis.ephys.utils.io_digital_events import fetch_session_events

            aligned_events = fetch_session_events(**session_key)
            inputs = extract_nidq_kernel_inputs(
                aligned_events,
                trial_rows,
                trialset_description,
                observation_window=observation_window,
            )
        else:
            inputs = extract_bpod_kernel_inputs(
                trial_rows,
                trialset_description,
                observation_window=observation_window,
            )
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


def extract_nidq_kernel_inputs(
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
    nidq_sync = np.asarray(
        [trial_starts[int(row["trial_num"])] for row in sync_rows], dtype=float
    )
    order = np.argsort(bpod_sync)
    bpod_sync = bpod_sync[order]
    nidq_sync = nidq_sync[order]

    stims = np.asarray(aligned_events["stim_ev"], dtype=float)
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
                nidq_sync,
            )
        )
        interpolated_exit = float(
            np.interp(float(row["t_react"]), bpod_sync, nidq_sync)
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
                np.interp(float(response_time), bpod_sync, nidq_sync)
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


def combine_kernel_inputs(session_inputs: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine per-session inputs from one timing source."""
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
    }


def _fetch_chipmunk_trial_rows(dataset_key: dict[str, Any]) -> list[dict[str, Any]]:
    from thesis.behavior.io import get_chipmunk_table

    Chipmunk = get_chipmunk_table()
    relation = Chipmunk.trial_query(**dataset_key)
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


def _concatenate(
    session_inputs: list[dict[str, Any]], field: str, *, dtype: type
) -> np.ndarray:
    values = [np.asarray(inputs[field], dtype=dtype) for inputs in session_inputs]
    return np.concatenate(values) if values else np.empty((0,), dtype=dtype)
