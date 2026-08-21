"""Chipmunk trial metadata aligned to OBX event timestamps.

**Naming convention**

- ``fetch_*`` — query Chipmunk via labdata relations and merge with an existing
  ``align_ev`` dict (from ``io_digital_events``).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from labdata.schema import SpikeSorting


def fetch_trial_metadata(
    subject: str,
    session: str,
    align_ev: dict[str, np.ndarray],
) -> Optional[pd.DataFrame]:
    """Fetch Chipmunk trial metadata and align with OBX trial_start timestamps.

    Returns a DataFrame with trial-level metadata or None if Chipmunk data
    is unavailable.

    Trial-count mismatches are treated conservatively. A one-trial mismatch is
    tolerated as a likely trailing partial trial and is truncated with a
    warning. Larger mismatches raise instead of silently truncating.
    """
    try:
        from chipmunk import Chipmunk

        sess_dicts = (
            SpikeSorting()
            & f'subject_name = "{subject}"'
            & f'session_name = "{session}"'
        ).fetch("subject_name", "session_name", as_dict=True)

        trial_data = (
            (Chipmunk() & sess_dicts)
            * Chipmunk.Trial().proj(
                "response",
                "with_choice",
                "rewarded",
                "early_withdrawal",
                "t_start",
                "t_sync",
                "t_initiate",
                "t_stim",
                "t_gocue",
                "t_react",
                "t_response",
                "stim_duration",
            )
            * Chipmunk.TrialParameters().proj("stim_rate_vision", "category_boundary")
        ).fetch(format="frame")
        tdf: pd.DataFrame = trial_data.reset_index(
            level=["subject_name", "session_name", "dataset_name"], drop=True
        ).sort_index()

        trial_starts = align_ev["trial_start"]
        n_obx = len(trial_starts)
        n_chipmunk = len(tdf)
        n = min(n_obx, n_chipmunk)
        mismatch = abs(n_obx - n_chipmunk)
        if n == 0:
            raise ValueError(
                f"No aligned trials available for {subject} {session}: "
                f"OBX={n_obx}, Chipmunk={n_chipmunk}"
            )
        if mismatch:
            if mismatch > 1:
                raise ValueError(
                    f"Suspicious trial-count mismatch for {subject} {session}: "
                    f"OBX trial_start pulses={n_obx}, Chipmunk trials={n_chipmunk}. "
                    "Refusing to silently truncate."
                )
            print(
                f"Warning: {subject} {session} has a 1-trial OBX/Chipmunk mismatch "
                f"(OBX={n_obx}, Chipmunk={n_chipmunk}); truncating to {n}."
            )
        trial_df = tdf.iloc[:n].copy()
        trial_df["trial_start_ts"] = trial_starts[:n]
        trial_df["prev_rewarded"] = trial_df["rewarded"].shift(1)
        trial_df["prev_response"] = trial_df["response"].shift(1)
        trial_df["prev_stim_rate"] = trial_df["stim_rate_vision"].shift(1)
        trial_df["stim_category"] = pd.cut(
            trial_df["stim_rate_vision"] - trial_df["category_boundary"],
            bins=[-np.inf, -1e-9, 1e-9, np.inf],
            labels=["low_rate", "boundary", "high_rate"],
        )
        return trial_df
    except Exception as e:
        print(f"Could not load Chipmunk trial metadata: {e}")
        return None


def trial_start_from_row(row: pd.Series) -> float:
    """Trial start from `center_port_entries` or `cp_entry`; else NaN."""
    if "center_port_entries" in row.index:
        entries = row["center_port_entries"]
        if entries is None or len(entries) == 0:
            return np.nan
        return float(entries[0])
    if "cp_entry" in row.index:
        return float(row["cp_entry"]) if np.isfinite(row["cp_entry"]) else np.nan
    return np.nan


fetch_chipmunk_trials_aligned = fetch_trial_metadata
