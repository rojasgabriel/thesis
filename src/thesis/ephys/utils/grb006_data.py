from __future__ import annotations

import numpy as np
import pandas as pd

GRB006_SUBJECT = "GRB006"
GRB006_SESSION = "20240821_121447"


def load_grb006_first_stim() -> np.ndarray:
    from thesis.ephys.utils.io_behavior import load_session_behavior

    _, _, first_stim = load_session_behavior(GRB006_SUBJECT, GRB006_SESSION)
    return first_stim[np.isfinite(first_stim)]


def fetch_spike_times(
    subject: str, session: str, unit_criteria_id: int = 1
) -> tuple[list[int], list[np.ndarray]]:
    from thesis.ephys.utils.io_session_units import fetch_good_units

    st_per_unit = fetch_good_units(subject, session, unit_criteria_id)
    return list(st_per_unit.keys()), list(st_per_unit.values())


def fetch_grb006_spike_times(
    unit_criteria_id: int = 1,
) -> tuple[list[int], list[np.ndarray]]:
    return fetch_spike_times(GRB006_SUBJECT, GRB006_SESSION, unit_criteria_id)


def load_grb006_aligned_trial_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    from thesis.ephys.utils.io_behavior import load_session_behavior

    trial_df, trial_ts, _ = load_session_behavior(GRB006_SUBJECT, GRB006_SESSION)
    return trial_df, trial_ts


def load_grb006_session_inputs(
    unit_criteria_id: int = 1,
) -> tuple[list[int], list[np.ndarray], pd.DataFrame, pd.DataFrame]:
    trial_df, trial_ts = load_grb006_aligned_trial_data()
    unit_ids, spike_times = fetch_grb006_spike_times(unit_criteria_id)
    return unit_ids, spike_times, trial_df, trial_ts
