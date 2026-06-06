"""Task rate-tuning light-dose control helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

VISUAL_FLASH_WIDTH_S = 0.015


def compute_light_exposure(
    windows_df: pd.DataFrame,
    stim_events: np.ndarray,
    flash_width_s: float = VISUAL_FLASH_WIDTH_S,
) -> pd.DataFrame:
    """Count visual flashes and derived light exposure in each trial window."""
    events = np.asarray(stim_events, dtype=float)
    rows = []
    for window in windows_df.itertuples(index=False):
        start = float(window.window_start_s)
        end = float(window.window_end_s)
        flash_count = int(np.count_nonzero((events >= start) & (events < end)))
        duration = float(window.window_duration_s)
        total_light_time = flash_count * flash_width_s
        rows.append(
            {
                "trial_idx": int(window.trial_idx),
                "stim_rate_vision": float(getattr(window, "stim_rate_vision", np.nan)),
                "window_start_s": start,
                "window_end_s": end,
                "window_duration_s": duration,
                "flash_count": flash_count,
                "total_light_time_s": total_light_time,
                "duty_cycle": total_light_time / duration if duration > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def add_light_exposure_to_responses(
    trial_responses: pd.DataFrame,
    light_exposure: pd.DataFrame,
) -> pd.DataFrame:
    """Merge trial light exposure into unit responses and compute spikes/flash."""
    exposure_columns = [
        "trial_idx",
        "flash_count",
        "total_light_time_s",
        "duty_cycle",
    ]
    out = trial_responses.merge(
        light_exposure[exposure_columns], on="trial_idx", how="left"
    )
    out["spikes_per_flash"] = np.where(
        out["flash_count"] > 0,
        out["spike_count"] / out["flash_count"],
        np.nan,
    )
    return out


def residualize_by_unit(
    trial_responses: pd.DataFrame,
    response_column: str,
    predictor_column: str,
) -> pd.DataFrame:
    """Regress one predictor out of each unit's response."""
    rows = []
    for _unit_id, unit_df in trial_responses.groupby("unit_id"):
        y = unit_df[response_column].to_numpy(dtype=float)
        predictor = unit_df[predictor_column].to_numpy(dtype=float)
        valid = np.isfinite(y) & np.isfinite(predictor)
        residual = np.full(y.shape, np.nan)
        if valid.sum() >= 2 and np.nanstd(predictor[valid]) > 0:
            x = np.column_stack([np.ones(valid.sum()), predictor[valid]])
            beta = np.linalg.pinv(x) @ y[valid]
            residual[valid] = y[valid] - x @ beta
        out = unit_df.copy()
        out["residual_response_sp_s"] = residual
        rows.append(out)
    return pd.concat(rows, ignore_index=True)
