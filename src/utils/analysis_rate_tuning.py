"""Task stimulus-period rate tuning helpers.

**Naming convention**

- ``build_*`` — construct analysis tables from source events and trials.
- ``compute_*`` — deterministic array transforms.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


STIM_RATE_MIN_HZ = 4
STIM_RATE_MAX_HZ = 20
CATEGORY_BOUNDARY_HZ = 12


def first_event_in_window(
    events: np.ndarray,
    start: float,
    end: float,
    include_start: bool = True,
) -> float:
    """Return the first event in a half-open time window, else NaN."""
    event_array = np.asarray(events, dtype=float)
    finite_events = event_array[np.isfinite(event_array)]
    if include_start:
        mask = (finite_events >= start) & (finite_events < end)
    else:
        mask = (finite_events > start) & (finite_events < end)
    if not mask.any():
        return np.nan
    return float(finite_events[mask][0])


def response_events_for_choice(align_ev: Mapping[str, np.ndarray], response: float):
    """Return side-port events matching the Chipmunk response code."""
    if response == -1:
        return align_ev["left_port"]
    if response == 1:
        return align_ev["right_port"]
    return np.array([], dtype=float)


def build_task_stimulus_windows(
    align_ev: Mapping[str, np.ndarray],
    trial_df: pd.DataFrame,
    min_rate_hz: int = STIM_RATE_MIN_HZ,
    max_rate_hz: int = STIM_RATE_MAX_HZ,
) -> pd.DataFrame:
    """Build first-flash-to-response trial windows for rate tuning."""
    required_columns = {
        "trial_start_ts",
        "stim_rate_vision",
        "response",
        "with_choice",
    }
    missing_columns = required_columns.difference(trial_df.columns)
    if missing_columns:
        raise ValueError(f"trial_df is missing columns: {sorted(missing_columns)}")

    trial_starts = trial_df["trial_start_ts"].to_numpy(dtype=float)
    first_stim_events = np.asarray(align_ev["first_stim_ev_15ms"], dtype=float)

    rows = []
    for position, (trial_idx, trial) in enumerate(trial_df.iterrows()):
        stim_rate = float(trial["stim_rate_vision"])
        if int(trial["with_choice"]) != 1:
            continue
        if not min_rate_hz <= stim_rate <= max_rate_hz:
            continue

        trial_start = float(trial["trial_start_ts"])
        if not np.isfinite(trial_start):
            continue
        trial_end = (
            float(trial_starts[position + 1])
            if position + 1 < len(trial_starts)
            else np.inf
        )

        first_flash = first_event_in_window(first_stim_events, trial_start, trial_end)
        if not np.isfinite(first_flash):
            continue

        response_events = response_events_for_choice(align_ev, trial["response"])
        response_onset = first_event_in_window(
            response_events,
            first_flash,
            trial_end,
            include_start=False,
        )
        if not np.isfinite(response_onset) or response_onset <= first_flash:
            continue

        rows.append(
            {
                "trial_idx": int(trial_idx),
                "stim_rate_vision": stim_rate,
                "response_side": int(trial["response"]),
                "with_choice": int(trial["with_choice"]),
                "trial_start_s": trial_start,
                "trial_end_s": trial_end,
                "window_start_s": first_flash,
                "window_end_s": response_onset,
                "window_duration_s": response_onset - first_flash,
                "category_boundary": float(
                    trial.get("category_boundary", CATEGORY_BOUNDARY_HZ)
                ),
                "rewarded": float(trial.get("rewarded", np.nan)),
                "early_withdrawal": float(trial.get("early_withdrawal", np.nan)),
            }
        )

    return pd.DataFrame(rows)


def compute_trial_responses(
    windows_df: pd.DataFrame,
    spike_times_by_unit: Mapping[int, np.ndarray],
) -> pd.DataFrame:
    """Compute spike counts and rates for every unit and valid trial window."""
    rows = []
    for unit_id, spike_times in spike_times_by_unit.items():
        spikes = np.asarray(spike_times, dtype=float)
        for window in windows_df.itertuples(index=False):
            start = float(window.window_start_s)
            end = float(window.window_end_s)
            spike_count = int(np.count_nonzero((spikes >= start) & (spikes < end)))
            duration = float(window.window_duration_s)
            rows.append(
                {
                    "unit_id": int(unit_id),
                    "trial_idx": int(window.trial_idx),
                    "stim_rate_vision": float(window.stim_rate_vision),
                    "response_side": int(window.response_side),
                    "with_choice": int(window.with_choice),
                    "category_boundary": float(
                        getattr(window, "category_boundary", CATEGORY_BOUNDARY_HZ)
                    ),
                    "rewarded": float(getattr(window, "rewarded", np.nan)),
                    "early_withdrawal": float(
                        getattr(window, "early_withdrawal", np.nan)
                    ),
                    "window_start_s": start,
                    "window_end_s": end,
                    "window_duration_s": duration,
                    "spike_count": spike_count,
                    "response_sp_s": spike_count / duration,
                }
            )

    return pd.DataFrame(rows)


def aggregate_tuning_curves(trial_responses: pd.DataFrame) -> pd.DataFrame:
    """Aggregate unit-by-trial responses into unit-by-rate tuning curves."""
    if trial_responses.empty:
        return pd.DataFrame(
            columns=[
                "unit_id",
                "stim_rate_vision",
                "n_trials",
                "mean_sp_s",
                "sem_sp_s",
                "median_sp_s",
                "window_duration_mean_s",
            ]
        )

    tuning = (
        trial_responses.groupby(["unit_id", "stim_rate_vision"], as_index=False)
        .agg(
            n_trials=("response_sp_s", "size"),
            mean_sp_s=("response_sp_s", "mean"),
            sem_sp_s=("response_sp_s", "sem"),
            median_sp_s=("response_sp_s", "median"),
            window_duration_mean_s=("window_duration_s", "mean"),
        )
        .sort_values(["unit_id", "stim_rate_vision"])
        .reset_index(drop=True)
    )
    tuning["sem_sp_s"] = tuning["sem_sp_s"].fillna(0.0)
    return tuning


def summarize_units(tuning_curves: pd.DataFrame) -> pd.DataFrame:
    """Summarize each unit's descriptive tuning curve shape."""
    if tuning_curves.empty:
        return pd.DataFrame(
            columns=[
                "unit_id",
                "mean_sp_s_all_rates",
                "min_rate_sp_s",
                "max_rate_sp_s",
                "preferred_stim_rate",
                "tuning_range_sp_s",
                "frequency_selectivity_index",
                "normalized_tuning_range",
            ]
        )

    rows = []
    for unit_id, unit_df in tuning_curves.groupby("unit_id"):
        unit_df = unit_df.sort_values("stim_rate_vision")
        max_idx = unit_df["mean_sp_s"].idxmax()
        min_rate = float(unit_df["mean_sp_s"].min())
        max_rate = float(unit_df["mean_sp_s"].max())
        tuning_range = max_rate - min_rate
        fsi_denominator = max_rate + min_rate
        frequency_selectivity_index = (
            tuning_range / fsi_denominator if fsi_denominator > 0 else np.nan
        )
        mean_all = float(unit_df["mean_sp_s"].mean())
        rows.append(
            {
                "unit_id": int(unit_id),
                "mean_sp_s_all_rates": mean_all,
                "min_rate_sp_s": min_rate,
                "max_rate_sp_s": max_rate,
                "preferred_stim_rate": float(unit_df.loc[max_idx, "stim_rate_vision"]),
                "tuning_range_sp_s": tuning_range,
                "frequency_selectivity_index": frequency_selectivity_index,
                "normalized_tuning_range": tuning_range / (mean_all + 1e-9),
            }
        )
    return pd.DataFrame(rows).sort_values("unit_id").reset_index(drop=True)
