"""Trial-event timestamp alignment helpers for the Stringer RRR baseline.

Ported from the category-decoding branch's ``io_chipmunk_trials`` enrichment so
this baseline can use ``first_stim_ts`` / ``center_port_exit_ts`` / ``response_ts``
without changing the maintained ``fetch_trial_metadata`` return contract.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def align_response_port_entries(
    trial_starts: np.ndarray,
    responses: pd.Series,
    align_ev: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Pick the response-port entry after center-port withdrawal for each trial."""
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
        center_withdrawal = trial_center_exits[-1]
        trial_entries = entries[(entries > center_withdrawal) & (entries < trial_end)]
        if trial_entries.size:
            response_ts[trial_idx] = trial_entries[0]

    return response_ts, response_ports


def align_trial_event_timestamps(
    trial_starts: np.ndarray,
    responses: pd.Series,
    align_ev: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Align center-port, first-stim, exit, and response times onto trial rows.

    ``first_stim_ts`` prefers session ``first_stim_ev_15ms`` (first-of-train),
    matching other ephys analyses. Falls back to the first ``stim_ev_15ms``
    inside the trial window when no first-of-train event is found.
    """
    center_entries = np.asarray(align_ev["center_port"], dtype=float)
    center_exits = np.asarray(align_ev.get("center_port_exit", []), dtype=float)
    first_stims = np.asarray(align_ev.get("first_stim_ev_15ms", []), dtype=float)
    stims = np.asarray(
        align_ev.get("stim_ev_15ms", align_ev.get("stim_ev", [])), dtype=float
    )
    response_ts, response_ports = align_response_port_entries(
        trial_starts, responses, align_ev
    )

    rows = []
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
        trial_first = first_stims[
            (first_stims > trial_start) & (first_stims < trial_end)
        ]
        if trial_first.size:
            first_stim_ts = float(trial_first[0])
        else:
            trial_stims = stims[(stims > trial_start) & (stims < trial_end)]
            first_stim_ts = float(trial_stims[0]) if trial_stims.size else np.nan
        rows.append(
            {
                "trial_start_ts": float(trial_start),
                "center_port_ts": (
                    float(trial_center_entries[0])
                    if trial_center_entries.size
                    else np.nan
                ),
                "center_port_exit_ts": (
                    float(trial_center_exits[-1]) if trial_center_exits.size else np.nan
                ),
                "first_stim_ts": first_stim_ts,
                "response_ts": float(response_ts[trial_idx]),
                "response_port": response_ports[trial_idx],
            }
        )
    return pd.DataFrame(rows)


def enrich_trials_with_event_timestamps(
    trial_df: pd.DataFrame,
    align_ev: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Attach OBX-aligned event timestamps to Chipmunk trial metadata."""
    trial_starts = np.asarray(
        trial_df.get("trial_start_ts", align_ev["trial_start"][: len(trial_df)]),
        dtype=float,
    )
    if len(trial_starts) != len(trial_df):
        trial_starts = np.asarray(align_ev["trial_start"][: len(trial_df)], dtype=float)
    event_timestamps = align_trial_event_timestamps(
        trial_starts, trial_df["response"], align_ev
    )
    out = trial_df.reset_index(drop=True).copy()
    for col in event_timestamps.columns:
        out[col] = event_timestamps[col].to_numpy()
    return out
