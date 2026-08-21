from __future__ import annotations

import numpy as np

from thesis.ephys.utils.io_session_units import fetch_good_unit_metrics_table


def fetch_waveform_durations_ms(
    subject: str,
    session: str,
    unit_ids: list[int],
    *,
    unit_criteria_id: int = 1,
) -> np.ndarray:
    metric_table, sampling_rate_hz = fetch_good_unit_metrics_table(
        subject, session, unit_criteria_id
    )
    if metric_table.empty:
        raise RuntimeError(
            f"No waveform duration rows returned for {subject} {session}."
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
    if missing_unit_ids:
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
    raise RuntimeError(
        "Waveform duration units are ambiguous. Expected either ms-scale values "
        "or sample counts that convert cleanly to ms."
    )
