"""Unit spike data and metrics from labdata.

**Naming convention**

- ``fetch_*`` — read from labdata / join tables.
- ``fetch_good_unit_metrics_table`` — DataFrame with normalized spike times and durations.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

# Silence the setuptools pkg_resources deprecation notice
warnings.filterwarnings("ignore", category=UserWarning, module="datajoint.plugin")

from labdata.schema import SpikeSorting, UnitCount  # noqa: E402


def fetch_good_unit_metrics_table(
    subject: str,
    session: str,
    unit_criteria_id: int = 1,
    stability_param_id: int | None = None,
) -> pd.DataFrame:
    """Return good-unit spike and metric rows, sorted by depth.

    `unit_criteria_id=1` is the project's standard quality criterion set
    (amplitude / SNR / contamination thresholds defined upstream in labdata).
    Don't change without reason — most downstream analyses assume criterion 1.

    Adds ``spike_times_s`` and ``spike_duration_ms`` columns. The upstream
    waveform metric already stores spike duration in milliseconds.

    If ``stability_param_id`` is set, this populates missing UnitStability
    rows before applying the stability filter.
    """
    sess_query = (
        SpikeSorting() & f'subject_name = "{subject}"' & f'session_name = "{session}"'
    ).proj()

    unit_key_fields = (
        "subject_name",
        "session_name",
        "dataset_name",
        "probe_num",
        "parameter_set_num",
        "unit_id",
    )
    good_unit_keys = (
        sess_query
        * (UnitCount.Unit & f"unit_criteria_id = {unit_criteria_id}" & "passes = 1")
    ).fetch(*unit_key_fields, as_dict=True)

    unit_query = SpikeSorting.Unit() & good_unit_keys
    if stability_param_id is not None:
        from labdata_plugin.schema import UnitStability

        stability_key = {
            "subject_name": subject,
            "session_name": session,
            "unit_criteria_id": unit_criteria_id,
            "unit_stability_param_id": stability_param_id,
        }
        UnitStability().populate(stability_key)
        stable_unit_keys = (UnitStability.Unit & stability_key & "passes = 1").fetch(
            *unit_key_fields, as_dict=True
        )
        unit_query &= stable_unit_keys

    good_units = pd.DataFrame(unit_query.get_spike_times(include_metrics=True)).rename(
        columns={"spike_times": "spike_times_s"}
    )
    if good_units.empty:
        raise ValueError(f"No units pass the selected filters for {subject} {session}")
    good_units["spike_duration_ms"] = good_units["spike_duration"].astype(float)
    return good_units.sort_values("depth", ascending=True)


def fetch_good_units(
    subject: str,
    session: str,
    unit_criteria_id: int = 1,
    stability_param_id: int | None = None,
) -> dict[int, np.ndarray]:
    """Fetch spike times (in seconds) for units passing quality criteria.

    Returns a dict mapping unit_id → spike_times_seconds, sorted by depth.
    """
    good_units = fetch_good_unit_metrics_table(
        subject, session, unit_criteria_id, stability_param_id
    )
    return dict(zip(good_units["unit_id"], good_units["spike_times_s"], strict=True))


def fetch_stimulus_excited_unit_ids(
    subject: str,
    session: str,
    unit_criteria_id: int = 1,
    responsiveness_param_id: int = 0,
) -> set[int]:
    """Populate responsiveness and return unit IDs classified as excited."""
    from labdata_plugin.schema import StimulusResponsiveness

    key = {
        "subject_name": subject,
        "session_name": session,
        "unit_criteria_id": unit_criteria_id,
        "responsiveness_param_id": responsiveness_param_id,
    }
    StimulusResponsiveness().populate(key)
    return {
        int(unit_id)
        for unit_id in (
            StimulusResponsiveness.Unit & key & 'response_type = "excited"'
        ).fetch("unit_id")
    }
