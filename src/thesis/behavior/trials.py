"""Load the behavior trials of a mouse from the Bpod data (no recording needed).

Sessions: the rate-discrimination sessions that have an ephys recording
(`EphysRecording`), whether or not their units are analyzed, that are visual
or visual + audio sessions, with at least `MIN_RATES` visual rates among the
used trials (so some are near the boundary). No performance rule (Gabriel,
2026-10-31: the expert rules and visual-only sessions leave too few GRB006
sessions). `RECORDED` names the session of each mouse that the neural analyses
use (the decoder session).

A session enters `RECORDED` only with good cortical units and expert behavior
(Gabriel, 2026-11-15): a required wait (`required_wait_time`) of at least
0.5 s, actual center-port times (center exit minus center entry, all
initiated trials) like those of GRB006 and GRB064, and few movement-artifact
units (`uv run movement-artifacts <subject> <session>`: stable units silent
in the port, under 0.1 of their recording rate, with a broad spike, fw3m over
0.25 ms; 0 in GRB006 20240821, GRB064 20261008 and GRB061 20261113).

GRB050 never learned the task and is out. GRB068 20261113 has expert waits
(0.58 s required, 0.63 s median center-port time) but is out: 45 of its 100
stable units are movement artifacts. Its other sessions are out too: 20261109
fails the wait rule and has 12 stable units; in 20261110 and 20261112 the
recording ended before the probe or behavior did, and they are not synced
(Gabriel, 2026-11-15: keep it clean). GRB064 stays on 20261008: of its
sessions that pass the wait rule, none has clearly more clean units or
trials. GRB061 20261113 fails the wait rule (0.37 s required, 0.48 s median
against 0.57 s for GRB006) and stays in provisionally, to see how its data
look until better sessions come (Gabriel, 2026-11-15). It has no ephys session
with 6 rates; it uses 4 (`MIN_RATES_BY_SUBJECT`).

Behavior-only mice (`PRE_IMPLANT`): GRB061 and GRB068 have no neural data in
the analyses yet, but they learned the same task. Their sessions are the expert
sessions before the implant: discrimination sessions before their first ephys
recording, with a median required wait (`required_wait_time`) of at least
`EXPERT_WAIT_S` and at least `MIN_RATES` rates (Gabriel, 2026-11-23: show their
expert sessions in the behavior figures). `BEHAVIOR_SUBJECTS` lists all mice of
the behavior figures.

Trials: visual (flashes only) and multisensory (flashes and clicks at the same
rate; `stim_events` holds the flash times), in `MODALITIES`; auditory-only
trials are left out. The column `modality` says which. Completed (no early withdrawal) with a left or right choice, with a full
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

RECORDED = {
    "GRB006": "20240821_121447",
    "GRB064": "20261008_143359",
    "GRB061": "20261113_133142",
}
SUBJECTS = tuple(RECORDED)
PRE_IMPLANT = ("GRB061", "GRB068")
BEHAVIOR_SUBJECTS = ("GRB006", "GRB064", *PRE_IMPLANT)
EXPERT_WAIT_S = 0.5
# The one list of the analyzed sessions; the neural analyses import it.
SESSIONS = tuple(f"{subject}_{name}" for subject, name in RECORDED.items())
MIN_RATES = 6
MIN_RATES_BY_SUBJECT = {"GRB061": 4}
MODALITIES = ("visual", "visual+audio")
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
    table["stim_events"] = table["stim_events"].map(_flash_times)
    # A left join: old sessions have no required wait, and a projection with
    # it would drop them.
    waits = (
        Chipmunk.trial_query(subject_name=subject)
        .proj("required_wait_time")
        .fetch(format="frame")
        .reset_index()
    )
    keys = ["subject_name", "session_name", "trial_num"]
    return table.merge(waits[[*keys, "required_wait_time"]], on=keys, how="left")


def _flash_times(events) -> list[float]:
    """Return the flash times: a flat list, or row 0 of the (2, n) array
    (row 1 is the clicks, NaN on visual trials)."""
    if events is None:
        return []
    events = np.atleast_2d(np.asarray(events, dtype=float))[0]
    return events[np.isfinite(events)].tolist()


def task_trials(subject: str, recorded: bool = False) -> pd.DataFrame:
    """Return the used trials of the sessions (or of the recorded session)."""
    table = _bpod_table(subject).sort_values(["session_name", "trial_num"])
    by_session = table.groupby("session_name")
    table["prev_response"] = by_session["response"].shift(1)
    table["prev_rewarded"] = by_session["rewarded"].shift(1)
    used = (
        (table["setting_task_mode"].astype(str) == "discrimination")
        & table["rewarded_modality"].astype(str).isin(MODALITIES)
        & (table["early_withdrawal"] == 0)
        & table["response"].isin((-1, 1))
        & (table["stim_duration"] >= STIM_S - 1e-6)
        & (table["stim_events"].map(len) > 0)
    )
    ephys = _ephys_sessions(subject)["session_name"]
    if subject in PRE_IMPLANT and not recorded:
        # Expert sessions before the implant (see the module docstring).
        used &= table["session_name"] < ephys.min()
        waits = table[used].groupby("session_name")["required_wait_time"].median()
        used &= table["session_name"].isin(waits.index[waits >= EXPERT_WAIT_S])
    else:
        used &= table["session_name"].isin(ephys)
    used &= table["setting_modalities"].astype(str).isin(MODALITIES)
    trials = table[used].copy()
    if not np.allclose(trials["category_boundary"], CATEGORY_BOUNDARY_HZ):
        raise ValueError(f"{subject}: a boundary is not {CATEGORY_BOUNDARY_HZ} Hz.")
    by_session = trials.groupby("session_name")
    rates = by_session["stim_rate_vision"].nunique()
    least = MIN_RATES if subject in PRE_IMPLANT and not recorded else None
    keep = rates.index[rates >= (least or MIN_RATES_BY_SUBJECT.get(subject, MIN_RATES))]
    if recorded:
        keep = [RECORDED[subject]]
    trials = trials[trials["session_name"].isin(keep)].copy()
    first = trials["t_sync"] + trials["stim_events"].map(lambda e: e[0])
    trials["flashes_s"] = trials["stim_events"].map(lambda e: np.asarray(e) - e[0])
    trials["exit_s"] = trials["t_react"] - first
    trials["go_cue_s"] = trials["t_gocue"] - first
    trials["right"] = (trials["response"] == 1).astype(int)
    trials["rate_hz"] = trials["stim_rate_vision"].astype(float)
    trials["modality"] = trials["rewarded_modality"].astype(str)
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
