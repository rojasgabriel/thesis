from __future__ import annotations

import numpy as np


def _as_float_array(value) -> np.ndarray:
    return np.asarray(value if value is not None else [], dtype=float)


def summarize_trialset(row: dict) -> dict:
    response_values = _as_float_array(row.get("response_values"))
    correct_values = _as_float_array(row.get("correct_values"))
    initiation_times = _as_float_array(row.get("initiation_times"))
    reaction_times = _as_float_array(row.get("reaction_times"))
    intensity_values = _as_float_array(row.get("intensity_values"))

    valid_choices = np.isfinite(response_values) & (response_values != 0)
    valid_correct = np.isfinite(correct_values)

    return {
        "n_trials": int(row.get("n_trials", len(response_values))),
        "n_with_choice": int(np.sum(valid_choices)),
        "n_correct": int(np.nansum(correct_values[valid_correct]))
        if valid_correct.any()
        else 0,
        "performance": _finite_or_none(row.get("performance")),
        "performance_easy": _finite_or_none(row.get("performance_easy")),
        "mean_initiation_time": _nanmean_or_none(initiation_times),
        "mean_reaction_time": _nanmean_or_none(reaction_times),
        "stim_values": np.sort(
            np.unique(intensity_values[np.isfinite(intensity_values)])
        ),
        "response_values": response_values,
        "correct_values": correct_values,
        "intensity_values": intensity_values,
    }


def _nanmean_or_none(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return float(np.mean(finite))


def _finite_or_none(value) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value):
        return None
    return value
