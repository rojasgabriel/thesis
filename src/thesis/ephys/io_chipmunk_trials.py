"""Chipmunk trial metadata aligned to OBX event timestamps."""

from __future__ import annotations

import numpy as np
import pandas as pd


def fetch_trial_metadata(
    subject: str,
    session: str,
    align_ev: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Fetch Chipmunk trials and align them with OBX trial-start timestamps."""
    from chipmunk import Chipmunk

    trial_data = (
        Chipmunk.trial_query(subject_name=subject, session_name=session).proj(
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
            "stim_rate_vision",
            "category_boundary",
        )
    ).fetch(format="frame")
    trial_df = trial_data.reset_index(
        level=["subject_name", "session_name", "dataset_name"], drop=True
    ).sort_index()

    trial_starts = align_ev["trial_start"]
    n_obx = len(trial_starts)
    n_chipmunk = len(trial_df)
    n = min(n_obx, n_chipmunk)
    mismatch = abs(n_obx - n_chipmunk)
    if n == 0:
        raise ValueError(
            f"No aligned trials available for {subject} {session}: "
            f"OBX={n_obx}, Chipmunk={n_chipmunk}"
        )
    if mismatch > 1:
        raise ValueError(
            f"Suspicious trial-count mismatch for {subject} {session}: "
            f"OBX trial_start pulses={n_obx}, Chipmunk trials={n_chipmunk}. "
            "Refusing to silently truncate."
        )
    if mismatch == 1:
        print(
            f"Warning: {subject} {session} has a 1-trial OBX/Chipmunk mismatch "
            f"(OBX={n_obx}, Chipmunk={n_chipmunk}); truncating to {n}."
        )

    trial_df = trial_df.iloc[:n].copy()
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
