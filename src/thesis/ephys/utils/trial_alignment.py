from __future__ import annotations

import numpy as np
import pandas as pd


def enrich_chipmunk_trial_table(trial_table: pd.DataFrame) -> pd.DataFrame:
    trial_table = trial_table.reset_index(drop=True).copy()
    trial_table["prev_response"] = trial_table["response"].shift(1)
    trial_table["prev_rewarded"] = trial_table["rewarded"].shift(1)
    trial_table["prev_stim_rate"] = trial_table["stim_rate_vision"].shift(1)
    return trial_table


def local_trial_row_signature(
    local_trial_row: pd.Series,
) -> tuple[int | None, int, int]:
    stimulus_rate = local_trial_row.get("trial_rate")
    stimulus_rate_key = int(stimulus_rate) if np.isfinite(stimulus_rate) else None
    outcome_code = int(local_trial_row.get("trial_outcome"))
    response_side = local_trial_row.get("response_side")
    if np.isfinite(response_side):
        choice_code = 1 if int(response_side) == 1 else -1
    else:
        choice_code = 0
    return stimulus_rate_key, choice_code, outcome_code


def chipmunk_trial_row_signature(
    chipmunk_trial_row: pd.Series,
) -> tuple[int | None, int, int]:
    stimulus_rate = chipmunk_trial_row.get("stim_rate_vision")
    stimulus_rate_key = int(stimulus_rate) if np.isfinite(stimulus_rate) else None
    choice_code = int(chipmunk_trial_row.get("response", 0))
    if chipmunk_trial_row.get("rewarded", 0) == 1:
        outcome_code = 1
    elif chipmunk_trial_row.get("with_choice", 0) == 1:
        outcome_code = 0
    else:
        outcome_code = 2
    return stimulus_rate_key, choice_code, outcome_code


def map_local_trial_rows_to_chipmunk_trials(
    local_trial_table: pd.DataFrame, chipmunk_trial_table: pd.DataFrame
) -> np.ndarray:
    matched_trial_indices = []
    search_start_index = 0
    chipmunk_signatures = [
        chipmunk_trial_row_signature(trial_row)
        for _, trial_row in chipmunk_trial_table.iterrows()
    ]
    for _, local_trial_row in local_trial_table.iterrows():
        target_signature = local_trial_row_signature(local_trial_row)
        matched_index = None
        for chipmunk_index in range(search_start_index, len(chipmunk_signatures)):
            if chipmunk_signatures[chipmunk_index] == target_signature:
                matched_index = chipmunk_index
                break
        if matched_index is None:
            relaxed_signature = (target_signature[0], target_signature[2])
            for chipmunk_index in range(search_start_index, len(chipmunk_signatures)):
                probe_signature = chipmunk_signatures[chipmunk_index]
                if (probe_signature[0], probe_signature[2]) == relaxed_signature:
                    matched_index = chipmunk_index
                    break
        if matched_index is None:
            raise RuntimeError(
                "Could not align local trial rows to Chipmunk trial rows "
                f"for {target_signature} starting at full trial index "
                f"{search_start_index}."
            )
        matched_trial_indices.append(matched_index)
        search_start_index = matched_index + 1
    return np.asarray(matched_trial_indices, dtype=int)


def fetch_chipmunk_trial_table(subject: str, session: str) -> pd.DataFrame:
    from labdata.schema import DecisionTask  # noqa: F401
    from chipmunk import Chipmunk

    restriction = f"subject_name = '{subject}' AND session_name = '{session}'"
    chipmunk_trial_table = pd.DataFrame(
        (Chipmunk * Chipmunk.Trial * Chipmunk.TrialParameters & restriction).fetch(
            order_by="trial_num"
        )
    )
    if chipmunk_trial_table.empty:
        raise RuntimeError(f"Could not load Chipmunk trials for {subject} {session}.")
    return enrich_chipmunk_trial_table(chipmunk_trial_table)
