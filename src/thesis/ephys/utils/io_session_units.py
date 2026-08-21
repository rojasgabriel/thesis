"""Good-unit spike data from labdata (quality table + spike times in seconds).

**Naming convention**

- ``fetch_*`` — read from labdata / join tables.
- ``fetch_good_unit_metrics_table`` — DataFrame with spike times in **samples** plus probe ``srate``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from labdata.schema import EphysRecording, SpikeSorting, UnitCount, UnitMetrics


def fetch_good_unit_metrics_table(
    subject: str,
    session: str,
    unit_criteria_id: int = 1,
) -> tuple[pd.DataFrame, float]:
    """Return good-unit rows with spike_times and depth, sorted by depth.

    `unit_criteria_id=1` is the project's standard quality criterion set
    (amplitude / SNR / contamination thresholds defined upstream in labdata).
    Don't change without reason — most downstream analyses assume criterion 1.

    Spike times in the frame are in **samples**; divide by ``srate`` for seconds.
    """
    sess_query = (
        SpikeSorting() & f'subject_name = "{subject}"' & f'session_name = "{session}"'
    ).proj()

    good_unit_ids = (
        sess_query
        * (UnitCount.Unit & f"unit_criteria_id = {unit_criteria_id}" & "passes = 1")
    ).fetch("subject_name", "session_name", "unit_id", as_dict=True)

    good_units = pd.DataFrame(
        ((SpikeSorting.Unit & good_unit_ids) * UnitMetrics).fetch(
            "unit_id", "spike_times", "depth", as_dict=True
        )
    )

    srate = float(
        (EphysRecording.ProbeSetting() & sess_query).fetch("sampling_rate")[0]
    )
    good_units = good_units.sort_values("depth", ascending=True)
    return good_units, srate


def fetch_good_units(
    subject: str,
    session: str,
    unit_criteria_id: int = 1,
) -> dict[int, np.ndarray]:
    """Fetch spike times (in seconds) for units passing quality criteria.

    Returns a dict mapping unit_id → spike_times_seconds, sorted by depth.
    """
    good_units, srate = fetch_good_unit_metrics_table(
        subject, session, unit_criteria_id
    )
    st_per_unit = {
        row["unit_id"]: row["spike_times"] / srate for _, row in good_units.iterrows()
    }
    return st_per_unit


def fetch_good_units_with_depth(
    subject: str,
    session: str,
    unit_criteria_id: int = 1,
) -> tuple[dict[int, np.ndarray], dict[int, float]]:
    """Fetch good units and aligned depth metadata sorted by depth."""
    good_units, srate = fetch_good_unit_metrics_table(
        subject, session, unit_criteria_id
    )
    st_per_unit = {
        row["unit_id"]: row["spike_times"] / srate for _, row in good_units.iterrows()
    }
    depth_per_unit = {
        int(row["unit_id"]): float(row["depth"]) for _, row in good_units.iterrows()
    }
    return st_per_unit, depth_per_unit
