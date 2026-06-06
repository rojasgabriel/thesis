"""Shared loading and plotting helpers for task rate-tuning scripts."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import types

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if "ephys" not in sys.modules:
    package = types.ModuleType("ephys")
    package.__path__ = [str(REPO_ROOT)]
    sys.modules["ephys"] = package
sys.path.insert(0, str(REPO_ROOT))

from ephys.src.utils.analysis_rate_tuning import (  # noqa: E402
    aggregate_tuning_curves,
    build_task_stimulus_windows,
    compute_trial_responses,
    summarize_units,
)
from ephys.src.utils.analysis_rate_tuning_light import (  # noqa: E402
    add_light_exposure_to_responses,
    compute_light_exposure,
)
from ephys.src.utils.analysis_rate_tuning_models import add_trial_predictors  # noqa: E402
from ephys.src.utils.io_chipmunk_trials import fetch_trial_metadata  # noqa: E402
from ephys.src.utils.io_digital_events import fetch_session_events  # noqa: E402
from ephys.src.utils.io_session_units import fetch_good_units  # noqa: E402

FIGURE_ROOT = Path(
    os.environ.get("EPHYS_FIGURE_ROOT", "/Users/gabriel/lib/ephys/figures")
)
FIGURE_DIR = FIGURE_ROOT / "task_rate_tuning"
SUBJECT_SESSIONS = [("GRB006", "20240821_121447")]
UNIT_CRITERIA_ID = 1
RANDOM_SEED = 0
TIMECOURSE_BIN_EDGES_S = np.arange(0.0, 1.0 + 0.1, 0.1)


def load_session_tables(subject: str, session: str) -> tuple:
    print(f"\nLoading {subject} {session}")
    align_ev = fetch_session_events(subject, session)
    trial_df = fetch_trial_metadata(subject, session, align_ev)
    if trial_df is None:
        raise RuntimeError(f"Could not load Chipmunk trials for {subject} {session}.")
    spike_times_by_unit = fetch_good_units(
        subject,
        session,
        unit_criteria_id=UNIT_CRITERIA_ID,
    )
    windows = build_task_stimulus_windows(align_ev, trial_df)
    if windows.empty:
        raise RuntimeError(f"No valid task stimulus windows for {subject} {session}.")

    trial_responses = compute_trial_responses(windows, spike_times_by_unit)
    light_exposure = compute_light_exposure(windows, align_ev["stim_ev_15ms"])
    trial_responses = add_light_exposure_to_responses(trial_responses, light_exposure)
    trial_responses = add_trial_predictors(trial_responses)
    tuning_curves = aggregate_tuning_curves(trial_responses)
    unit_summary = summarize_units(tuning_curves)

    for table in (
        windows,
        trial_responses,
        tuning_curves,
        unit_summary,
        light_exposure,
    ):
        table.insert(0, "session", session)
        table.insert(0, "subject", subject)

    print(f"  Units: {len(spike_times_by_unit)}")
    print(f"  Valid trials: {len(windows)}")
    print(
        "  Rates: "
        + ", ".join(
            str(int(rate)) for rate in sorted(windows["stim_rate_vision"].unique())
        )
    )
    return (
        align_ev,
        spike_times_by_unit,
        windows,
        trial_responses,
        tuning_curves,
        unit_summary,
        light_exposure,
    )


def load_all_sessions() -> tuple[pd.DataFrame, ...]:
    all_windows = []
    all_trial_responses = []
    all_tuning_curves = []
    all_unit_summaries = []
    all_light_exposures = []
    session_payloads = []
    for subject, session in SUBJECT_SESSIONS:
        payload = load_session_tables(subject, session)
        (
            align_ev,
            spike_times_by_unit,
            windows,
            trial_responses,
            tuning_curves,
            unit_summary,
            light_exposure,
        ) = payload
        session_payloads.append(
            (subject, session, align_ev, spike_times_by_unit, windows)
        )
        all_windows.append(windows)
        all_trial_responses.append(trial_responses)
        all_tuning_curves.append(tuning_curves)
        all_unit_summaries.append(unit_summary)
        all_light_exposures.append(light_exposure)
    return (
        pd.concat(all_windows, ignore_index=True),
        pd.concat(all_trial_responses, ignore_index=True),
        pd.concat(all_tuning_curves, ignore_index=True),
        pd.concat(all_unit_summaries, ignore_index=True),
        pd.concat(all_light_exposures, ignore_index=True),
        session_payloads,
    )


def pivot_tuning(
    tuning_curves: pd.DataFrame,
    value_column: str = "mean_sp_s",
    rates: np.ndarray | None = None,
) -> pd.DataFrame:
    pivot = tuning_curves.pivot_table(
        index=["subject", "session", "unit_id"],
        columns="stim_rate_vision",
        values=value_column,
    )
    if rates is None:
        rates = np.asarray(sorted(tuning_curves["stim_rate_vision"].unique()))
    return pivot.reindex(columns=rates)
