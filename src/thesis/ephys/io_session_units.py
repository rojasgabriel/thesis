"""Good-unit spike data and metrics from labdata.

**Naming convention**

- ``fetch_*`` — read from labdata / join tables.
- ``fetch_good_unit_metrics_table`` — DataFrame with normalized spike times and durations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from labdata.schema import SpikeSorting, UnitCount


def fetch_good_unit_metrics_table(
    subject: str,
    session: str,
    unit_criteria_id: int = 1,
) -> pd.DataFrame:
    """Return good-unit spike and metric rows, sorted by depth.

    `unit_criteria_id=1` is the project's standard quality criterion set
    (amplitude / SNR / contamination thresholds defined upstream in labdata).
    Don't change without reason — most downstream analyses assume criterion 1.

    Adds ``spike_times_s`` and ``spike_duration_ms`` columns. The upstream
    waveform metric already stores spike duration in milliseconds.
    """
    sess_query = (
        SpikeSorting() & f'subject_name = "{subject}"' & f'session_name = "{session}"'
    ).proj()

    good_unit_ids = (
        sess_query
        * (UnitCount.Unit & f"unit_criteria_id = {unit_criteria_id}" & "passes = 1")
    ).fetch("subject_name", "session_name", "unit_id", as_dict=True)

    good_units = pd.DataFrame(
        (SpikeSorting.Unit() & good_unit_ids).get_spike_times(include_metrics=True)
    ).rename(columns={"spike_times": "spike_times_s"})
    good_units["spike_duration_ms"] = good_units["spike_duration"].astype(float)
    return good_units.sort_values("depth", ascending=True)


def fetch_good_units(
    subject: str,
    session: str,
    unit_criteria_id: int = 1,
) -> dict[int, np.ndarray]:
    """Fetch spike times (in seconds) for units passing quality criteria.

    Returns a dict mapping unit_id → spike_times_seconds, sorted by depth.
    """
    good_units = fetch_good_unit_metrics_table(subject, session, unit_criteria_id)
    return dict(zip(good_units["unit_id"], good_units["spike_times_s"], strict=True))
