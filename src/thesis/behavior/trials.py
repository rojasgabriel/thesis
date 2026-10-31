"""Load the behavior trials of a mouse from the Bpod data (no recording needed).

Sessions: the rate-discrimination sessions that have an ephys recording
(`EphysRecording`), whether or not their units are analyzed, that are
visual-only sessions, with at least `MIN_RATES` visual rates among the used
trials (so some are near the boundary). No performance rule (Gabriel,
2026-10-31: the expert rules leave too few GRB006 sessions). `RECORDED` names the session of each mouse that the neural analyses
use (the decoder session).

Trials: rewarded on the visual rate. Completed (no early withdrawal) with a left or right choice, with a full
`STIM_S` train. The previous side and outcome
are taken before this filter (the trial just before, of any kind).

Times are from the Bpod clock: the flash times (`stim_events`) count from
`t_sync`; the first flash is time 0. The center exit is `t_react`, the go cue
`t_gocue`. Bpod state times can be rounded to 10 ms, so a time can differ from
the hardware time by that much.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from thesis.ephys.cache import cached
from thesis.ephys.trials import CATEGORY_BOUNDARY_HZ

RECORDED = {"GRB006": "20240821_121447", "GRB064": "20261008_143359"}
SUBJECTS = tuple(RECORDED)
MIN_RATES = 6
STIM_S = 1.0
FIELDS = (
    "setting_modalities",
    "setting_task_mode",
    "stim_duration",
    "stim_rate_vision",
    "category_boundary",
    "rewarded_modality",
    "early_withdrawal",
    "response",
    "rewarded",
    "t_sync",
    "t_react",
    "t_gocue",
    "stim_events",
)


@cached
def _ephys_sessions(subject: str) -> pd.DataFrame:
    import labdata  # noqa: F401
    from labdata.schema import EphysRecording

    names = (EphysRecording & {"subject_name": subject}).fetch("session_name")
    return pd.DataFrame({"session_name": sorted(set(names))})


@cached
def _bpod_table(subject: str) -> pd.DataFrame:
    import labdata  # noqa: F401  (loads the chipmunk plugin)
    from chipmunk import Chipmunk

    table = (
        Chipmunk.trial_query(subject_name=subject)
        .proj(*FIELDS)
        .fetch(format="frame")
        .reset_index()
    )
    table["stim_events"] = table["stim_events"].map(
        lambda e: [float(t) for t in e] if isinstance(e, list | tuple) else []
    )
    return table


def task_trials(subject: str, recorded: bool = False) -> pd.DataFrame:
    """Return the used trials of the sessions (or of the recorded session)."""
    table = _bpod_table(subject).sort_values(["session_name", "trial_num"])
    by_session = table.groupby("session_name")
    table["prev_response"] = by_session["response"].shift(1)
    table["prev_rewarded"] = by_session["rewarded"].shift(1)
    used = (
        (table["setting_task_mode"].astype(str) == "discrimination")
        & (table["rewarded_modality"].astype(str) == "visual")
        & (table["early_withdrawal"] == 0)
        & table["response"].isin((-1, 1))
        & (table["stim_duration"] >= STIM_S - 1e-6)
        & (table["stim_events"].map(len) > 0)
    )
    used &= table["session_name"].isin(_ephys_sessions(subject)["session_name"])
    used &= table["setting_modalities"].astype(str) == "visual"
    trials = table[used].copy()
    if not np.allclose(trials["category_boundary"], CATEGORY_BOUNDARY_HZ):
        raise ValueError(f"{subject}: a boundary is not {CATEGORY_BOUNDARY_HZ} Hz.")
    by_session = trials.groupby("session_name")
    rates = by_session["stim_rate_vision"].nunique()
    keep = rates.index[rates >= MIN_RATES]
    if recorded:
        keep = [RECORDED[subject]]
    trials = trials[trials["session_name"].isin(keep)].copy()
    first = trials["t_sync"] + trials["stim_events"].map(lambda e: e[0])
    trials["flashes_s"] = trials["stim_events"].map(lambda e: np.asarray(e) - e[0])
    trials["exit_s"] = trials["t_react"] - first
    trials["go_cue_s"] = trials["t_gocue"] - first
    trials["right"] = (trials["response"] == 1).astype(int)
    trials["rate_hz"] = trials["stim_rate_vision"].astype(float)
    # Fast trains are rewarded on the right (the strength axis assumes it).
    wrong = trials[
        (trials["rewarded"] == 1) & (trials["rate_hz"] != CATEGORY_BOUNDARY_HZ)
    ]
    if np.any((wrong["rate_hz"] > CATEGORY_BOUNDARY_HZ) != (wrong["response"] == 1)):
        raise ValueError(f"{subject}: a rewarded fast train is not on the right.")
    return trials.reset_index(drop=True)


def session_columns(sessions: pd.Series) -> np.ndarray:
    """Return one indicator column per session (each session its own bias)."""
    return pd.get_dummies(sessions).to_numpy(float)


def hierarchical_samples(sessions: pd.Series, n: int, seed: int = 0):
    """Yield (rows, session labels): sessions with replacement, then their trials."""
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(sessions.to_numpy() == s) for s in sessions.unique()]
    for _ in range(n):
        rows, labels = [], []
        for label, pick in enumerate(rng.integers(0, len(groups), len(groups))):
            members = groups[pick]
            rows.append(rng.choice(members, len(members)))
            labels.append(np.full(len(members), label))
        yield np.concatenate(rows), pd.Series(np.concatenate(labels))
