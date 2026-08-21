"""Session-level orchestration: events + Chipmunk + conditioned-stim classification.

**Naming convention**

- ``load_*`` — compose several subsystems into one bundle for downstream analyses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from thesis.ephys.utils.analysis_conditioned_stim import build_trial_stim_classification
from thesis.ephys.utils.io_chipmunk_trials import fetch_trial_metadata
from thesis.ephys.utils.io_digital_events import fetch_session_events


def load_session_behavior(
    subject: str, session: str
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Load events, Chipmunk trials, enriched table, stim classification, first 15 ms stims."""
    align_ev = fetch_session_events(subject, session)
    trial_df = fetch_trial_metadata(subject, session, align_ev)
    if trial_df is None:
        raise RuntimeError(f"Could not load trial metadata for {subject} {session}")

    trial_ts = build_trial_stim_classification(align_ev, trial_df).reset_index(
        drop=True
    )
    first_stim_times = np.asarray(align_ev["first_stim_ev_15ms"], dtype=float)
    return trial_df, trial_ts, first_stim_times
