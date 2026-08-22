"""Conditioned-stimulus classification and anchor extraction (locomotion pipeline).

**Naming convention**

- ``build_*`` — construct a derived table or structure from events + trials.
- ``extract_*`` — pull named anchor time arrays from an existing trial table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def load_trial_classification(subject: str, session: str) -> pd.DataFrame:
    """Load the conditioned-stim classification for one session."""
    from thesis.ephys.utils.io_chipmunk_trials import fetch_trial_metadata
    from thesis.ephys.utils.io_digital_events import fetch_session_events

    align_ev = fetch_session_events(subject, session)
    trial_df = fetch_trial_metadata(subject, session, align_ev)
    return build_trial_stim_classification(align_ev, trial_df).reset_index(drop=True)


def build_trial_stim_classification(
    align_ev: dict,
    trial_df,
) -> "pd.DataFrame":
    """Classify each 15 ms stim pulse as stationary or movement for every trial.

    Uses DB center-port exit events as the movement onset when available.
    If the session has no explicit center-port exit events, falls back to the
    bpod→OBX clock mapping derived from paired t_sync (bpod) and trial_start
    timestamps to convert t_react into OBX time. Stim pulses that fall between
    center-port entry and center-port exit are labelled stationary; pulses
    between center-port exit and response-port entry are labelled movement.

    Reentrances: when an animal pokes center port, leaves, and re-enters
    before the final exit (t_react), the "stable" fixation period is the
    interval between the LAST center port entry before t_react and t_react
    itself.  Earlier brief pokes are excluded from the stationary window.

    Args:
        align_ev: dict returned by fetch_session_events.  Must contain
            'trial_start', 'center_port', 'left_port', 'right_port',
            and 'stim_ev_15ms'. If present, 'center_port_exit' is used as
            the center-port exit source.
        trial_df: DataFrame returned by fetch_trial_metadata.  Must contain
            't_sync', 't_react', and 'response' columns.

    Returns:
        DataFrame with one row per trial that has both stationary and movement
        stimulus lists. Columns include trial timing, both stimulus lists, and
        the number of center-port entries.
    """
    stim_times = np.asarray(align_ev["stim_ev_15ms"])
    cp_entries = np.asarray(align_ev["center_port"])
    cp_exits = np.asarray(align_ev.get("center_port_exit", []), dtype=float)
    left_entries = np.asarray(align_ev["left_port"])
    right_entries = np.asarray(align_ev["right_port"])
    obx_trial_starts = np.asarray(align_ev["trial_start"])

    n = min(len(trial_df), len(obx_trial_starts))
    bpod_sync = trial_df["t_sync"].iloc[:n].to_numpy(dtype=float)
    obx_sync = obx_trial_starts[:n].astype(float)

    valid = np.isfinite(bpod_sync) & np.isfinite(obx_sync)
    bpod_sync = bpod_sync[valid]
    obx_sync = obx_sync[valid]

    t_react = trial_df["t_react"].iloc[:n].to_numpy(dtype=float)
    response = trial_df["response"].iloc[:n].to_numpy()

    cp_exit_obx = np.interp(t_react, bpod_sync, obx_sync)

    rows = []
    for i in range(n):
        if not np.isfinite(t_react[i]):
            continue

        trial_start = obx_trial_starts[i]
        trial_end = obx_trial_starts[i + 1] if i + 1 < len(obx_trial_starts) else np.inf

        trial_cp_exits = cp_exits[(cp_exits > trial_start) & (cp_exits < trial_end)]
        if trial_cp_exits.size:
            cp_exit = trial_cp_exits[np.argmin(np.abs(trial_cp_exits - cp_exit_obx[i]))]
        else:
            cp_exit = cp_exit_obx[i]

        cp_mask = (
            (cp_entries > trial_start)
            & (cp_entries < cp_exit)
            & (cp_entries < trial_end)
        )
        if not cp_mask.any():
            continue
        cp_entry = cp_entries[cp_mask][-1]

        if response[i] == 1:
            rp_pool = right_entries
        elif response[i] == -1:
            rp_pool = left_entries
        else:
            continue
        rp_mask = (rp_pool > cp_exit) & (rp_pool < trial_end)
        if not rp_mask.any():
            continue
        rp_entry = rp_pool[rp_mask][0]

        stat = stim_times[(stim_times >= cp_entry) & (stim_times < cp_exit)].tolist()
        move = stim_times[(stim_times >= cp_exit) & (stim_times <= rp_entry)].tolist()

        if not stat or not move:
            continue

        rows.append(
            dict(
                trial_idx=i,
                cp_entry=cp_entry,
                cp_exit_obx=cp_exit,
                rp_entry=rp_entry,
                stationary_stims=stat,
                movement_stims=move,
                n_cp_entries=int(cp_mask.sum()),
            )
        )

    return pd.DataFrame(rows)


def extract_paired_stim_anchors(
    trial_ts: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Return last-stationary and first-movement stimulus times."""
    has_stat = trial_ts["stationary_stims"].apply(lambda x: len(x) > 0)
    has_move = trial_ts["movement_stims"].apply(lambda x: len(x) > 0)
    paired_trials = trial_ts[has_stat & has_move]
    last_stationary = np.array(
        [stims[-1] for stims in paired_trials["stationary_stims"]],
        dtype=float,
    )
    first_movement = np.array(
        [stims[0] for stims in paired_trials["movement_stims"]],
        dtype=float,
    )
    return last_stationary, first_movement
