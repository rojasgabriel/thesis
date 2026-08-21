from __future__ import annotations

import numpy as np
import pandas as pd


def fetch_waveform_durations_ms(
    subject: str,
    session: str,
    unit_ids: list[int],
    *,
    strict: bool,
    unit_criteria_id: int = 1,
) -> np.ndarray:
    from labdata.schema import EphysRecording, SpikeSorting, UnitCount, UnitMetrics

    session_query = (
        SpikeSorting() & f'subject_name = "{subject}"' & f'session_name = "{session}"'
    ).proj()
    good_unit_rows = (
        session_query
        * (UnitCount.Unit & f"unit_criteria_id = {unit_criteria_id}" & "passes = 1")
    ).fetch("subject_name", "session_name", "unit_id", as_dict=True)
    metric_table = pd.DataFrame(
        ((SpikeSorting.Unit & good_unit_rows) * UnitMetrics).fetch(
            "unit_id", "spike_duration", as_dict=True
        )
    )
    if metric_table.empty:
        if strict:
            raise RuntimeError(
                f"No waveform duration rows returned for {subject} {session}."
            )
        return np.full(len(unit_ids), np.nan, dtype=float)

    sampling_rate_hz = float(
        (EphysRecording.ProbeSetting() & session_query).fetch("sampling_rate")[0]
    )
    duration_by_unit = dict(
        zip(
            metric_table["unit_id"].astype(int).tolist(),
            metric_table["spike_duration"].astype(float).tolist(),
        )
    )
    raw_durations = np.array(
        [duration_by_unit.get(int(unit_id), np.nan) for unit_id in unit_ids],
        dtype=float,
    )
    missing_unit_ids = [
        int(unit_id)
        for unit_id, duration in zip(unit_ids, raw_durations)
        if not np.isfinite(duration)
    ]
    if missing_unit_ids and strict:
        raise RuntimeError(
            f"Missing waveform duration for {subject} {session} units: "
            f"{missing_unit_ids[:10]}"
        )
    if np.any(raw_durations[np.isfinite(raw_durations)] <= 0):
        raise RuntimeError(
            f"Non-positive waveform durations encountered for {subject} {session}."
        )

    finite_durations = raw_durations[np.isfinite(raw_durations)]
    if finite_durations.size == 0:
        return raw_durations

    durations_look_like_ms = np.all(
        (finite_durations >= 0.05) & (finite_durations < 10)
    )
    converted_finite = finite_durations / sampling_rate_hz * 1000.0
    durations_look_like_samples = np.all(
        (converted_finite >= 0.05) & (converted_finite < 10)
    )

    if durations_look_like_ms and not durations_look_like_samples:
        return raw_durations
    if durations_look_like_samples and not durations_look_like_ms:
        return raw_durations / sampling_rate_hz * 1000.0
    if not strict:
        median_duration = np.nanmedian(raw_durations)
        if median_duration > 100:
            return raw_durations / sampling_rate_hz * 1000.0
        return raw_durations
    raise RuntimeError(
        "Waveform duration units are ambiguous. Expected either ms-scale values "
        "or sample counts that convert cleanly to ms."
    )


def fetch_spike_duration_ms(
    subject: str, session: str, unit_ids: list[int], *, unit_criteria_id: int = 1
) -> np.ndarray:
    return fetch_waveform_durations_ms(
        subject, session, unit_ids, strict=False, unit_criteria_id=unit_criteria_id
    )
