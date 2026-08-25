"""Unit spike data, metrics, and stored classifications from labdata."""

from __future__ import annotations

import warnings

import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning, module="datajoint.plugin")


def fetch_unit_table(
    subject: str,
    session: str,
    unit_criteria_id: int = 1,
    stability_param_id: int | None = None,
) -> pd.DataFrame:
    """Return units passing the requested quality filters, sorted by depth.

    `unit_criteria_id=1` is the project's standard quality criterion set
    (amplitude / SNR / contamination thresholds defined upstream in labdata).
    Don't change without reason — most downstream analyses assume criterion 1.

    Spike times are in seconds. The upstream waveform metric stores spike
    duration in milliseconds.
    """
    from labdata.schema import SpikeSorting, UnitCount

    session_query = {
        "subject_name": subject,
        "session_name": session,
    }
    quality_query = {
        **session_query,
        "unit_criteria_id": unit_criteria_id,
        "passes": 1,
    }
    passing_units = SpikeSorting.Unit & (UnitCount.Unit & quality_query)
    if stability_param_id is not None:
        from labdata_plugin.schema import UnitStability

        stability_query = {
            **session_query,
            "unit_criteria_id": unit_criteria_id,
            "stability_param_id": stability_param_id,
        }
        if len(UnitStability & stability_query) == 0:
            raise ValueError(
                f"UnitStability has not been populated for {stability_query}"
            )
        passing_units &= UnitStability.Unit & stability_query & {"passes": 1}

    unit_table = pd.DataFrame(
        passing_units.get_spike_times(include_metrics=True)
    ).rename(
        columns={
            "spike_times": "spike_times_s",
            "spike_duration": "spike_duration_ms",
        }
    )
    if unit_table.empty:
        raise ValueError(f"No units pass the selected filters for {subject} {session}")
    duplicate_unit_ids = unit_table.loc[
        unit_table["unit_id"].duplicated(keep=False), "unit_id"
    ].unique()
    if duplicate_unit_ids.size:
        raise ValueError(
            "Unit IDs are not unique across this session: "
            f"{duplicate_unit_ids.tolist()}"
        )
    return unit_table.sort_values("depth").reset_index(drop=True)
