"""Chipmunk trial metadata aligned to OBX event timestamps.

**Naming convention**

- ``fetch_*`` — query Chipmunk via labdata relations and merge with an existing
  ``align_ev`` dict (from ``io_digital_events``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def align_response_port_entries(
    trial_starts: np.ndarray,
    responses: pd.Series,
    align_ev: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Select the response-port entry after center-port exit for each trial."""
    left_entries = np.asarray(align_ev["left_port"], dtype=float)
    right_entries = np.asarray(align_ev["right_port"], dtype=float)
    center_exits = np.asarray(align_ev.get("center_port_exit", []), dtype=float)
    response_ts = np.full(len(trial_starts), np.nan)
    response_ports = np.full(len(trial_starts), "", dtype=object)

    for trial_idx, response in enumerate(responses):
        if response == 1:
            entries = right_entries
            response_ports[trial_idx] = "right"
        elif response == -1:
            entries = left_entries
            response_ports[trial_idx] = "left"
        else:
            continue

        trial_start = trial_starts[trial_idx]
        trial_end = (
            trial_starts[trial_idx + 1] if trial_idx + 1 < len(trial_starts) else np.inf
        )
        trial_center_exits = center_exits[
            (center_exits > trial_start) & (center_exits < trial_end)
        ]
        if not trial_center_exits.size:
            continue

        center_exit = trial_center_exits[-1]
        trial_entries = entries[(entries > center_exit) & (entries < trial_end)]
        if trial_entries.size:
            response_ts[trial_idx] = trial_entries[0]

    return response_ts, response_ports


def align_trial_event_timestamps(
    trial_starts: np.ndarray,
    responses: pd.Series,
    align_ev: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Build per-trial event timestamps from the synchronized event arrays."""
    center_entries = np.asarray(align_ev["center_port"], dtype=float)
    center_exits = np.asarray(align_ev.get("center_port_exit", []), dtype=float)
    stims = np.asarray(
        align_ev.get("stim_ev_15ms", align_ev.get("stim_ev", [])), dtype=float
    )
    response_ts, response_ports = align_response_port_entries(
        trial_starts, responses, align_ev
    )

    rows: list[dict[str, object]] = []
    for trial_idx, trial_start in enumerate(trial_starts):
        trial_end = (
            trial_starts[trial_idx + 1] if trial_idx + 1 < len(trial_starts) else np.inf
        )
        trial_center_entries = center_entries[
            (center_entries > trial_start) & (center_entries < trial_end)
        ]
        trial_center_exits = center_exits[
            (center_exits > trial_start) & (center_exits < trial_end)
        ]
        trial_stims = stims[(stims > trial_start) & (stims < trial_end)]
        rows.append(
            {
                "trial_start_ts": trial_start,
                "center_port_ts": (
                    trial_center_entries[0] if trial_center_entries.size else np.nan
                ),
                "stim_ts": trial_stims,
                "first_stim_ts": trial_stims[0] if trial_stims.size else np.nan,
                "center_port_exit_ts": (
                    trial_center_exits[-1] if trial_center_exits.size else np.nan
                ),
                "response_ts": response_ts[trial_idx],
                "response_port": response_ports[trial_idx],
            }
        )
    return pd.DataFrame(rows)


def fetch_trial_metadata(
    subject: str,
    session: str,
    align_ev: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Fetch Chipmunk trial metadata and align with OBX trial_start timestamps.

    Returns a DataFrame with trial-level metadata. Raises if Chipmunk data is
    unavailable.

    Trial-count mismatches are treated conservatively. A one-trial mismatch is
    tolerated as a likely trailing partial trial and is truncated with a
    warning. Larger mismatches raise instead of silently truncating.
    """
    try:
        from chipmunk import Chipmunk
        from labdata.schema import SpikeSorting

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
        event_timestamps = align_trial_event_timestamps(
            trial_starts[:n], trial_df["response"], align_ev
        )
        trial_df = pd.concat(
            [
                trial_df.drop(
                    columns=[
                        "t_start",
                        "t_sync",
                        "t_initiate",
                        "t_stim",
                        "t_gocue",
                        "t_react",
                        "t_response",
                    ]
                ).reset_index(drop=True),
                event_timestamps,
            ],
            axis=1,
        )
        trial_df["prev_rewarded"] = trial_df["rewarded"].shift(1)
        trial_df["prev_response"] = trial_df["response"].shift(1)
        trial_df["prev_stim_rate"] = trial_df["stim_rate_vision"].shift(1)
        trial_df["stim_category"] = pd.cut(
            trial_df["stim_rate_vision"] - trial_df["category_boundary"],
            bins=[-np.inf, -1e-9, 1e-9, np.inf],
            labels=["low_rate", "boundary", "high_rate"],
        )
        return trial_df[
            [
                "stim_rate_vision",
                "response",
                "rewarded",
                "prev_stim_rate",
                "prev_response",
                "prev_rewarded",
                "stim_category",
                "early_withdrawal",
                "with_choice",
                "trial_start_ts",
                "center_port_ts",
                "stim_ts",
                "first_stim_ts",
                "center_port_exit_ts",
                "response_ts",
                "response_port",
            ]
        ]
    except Exception as e:
        raise RuntimeError(
            f"Could not load Chipmunk trial metadata for {subject} {session}: {e}"
        ) from e


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
