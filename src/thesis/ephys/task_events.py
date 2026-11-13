"""Task-only visual pulses for stimulus-response classification and task analyses.

Compatibility: requires nonempty, contiguous zero-based Chipmunk task metadata
and hardware trial-start pulses that pass trials._trial_start_indices. That
mapping uses saved Bpod pulse numbers to skip interrupted trials. Legacy
unnumbered streams are supported only within that mapping's count tolerances;
they cannot identify arbitrary middle gaps. The legacy missing-final-task-pulse
warning/tolerance is retained. No pooled-event fallback is permitted.
Historical equivalence applies only when every old selected flash lies inside
an aligned saved task window. Pre-task flashes and flashes in an unsaved final
trial are intentionally excluded, even in sessions without passive replay.

Validation before using results: verify the Chipmunk task/replay rows, numbered
Bpod sync values and hardware start pulses for each session with this code.
StimulusResponse rows created by older code have no population provenance field:
validate historical task-only equivalence, or explicitly regenerate affected
master/Unit rows under separate authorization. populate skips existing rows;
the browser's populate call does not refresh them. Do not mix those rows with
new task-only PETHs or interpret them as corrected classifications.

Use a fresh THESIS_CACHE_DIR (or explicitly invalidate affected files) after
validating source data and stored classifications. In particular, cached
build_trial_table/build_replay_table files predate corrected pulse mapping;
fetch_unit_table files can retain spike times from old StreamSync mappings. Regenerate
analysis outputs after this validation: double_peaks, flash_weight_check,
stim_responses_by_locomotion, glm/peaks, leak_test, flash_movement, and flash_by_gap.
The last also depends on raw_video_trust_classification.json. Saved results/glm
JSON files and their _before_* backups are outside THESIS_CACHE_DIR; a fresh
cache does not refresh them. Preserve historical evidence and regenerate current
outputs deliberately. This helper is uncached and never repopulates rows.

Imports of trials reuse only the pulse mapping, not the cached trial builder;
there is no dependency through unit selection or StimulusResponse. The general
fetch_session_events API continues to include task and replay events.
"""

import numpy as np
import pandas as pd


def fetch_task_stimulus_pulses(subject: str, session: str) -> pd.DataFrame:
    """Return original pulse rows in saved task windows, preserving train flags.

    Each window ends at the next actual hardware pulse, including an unsaved
    trial's pulse. The final window is open-ended only without a later pulse.
    Replay and unsaved-trial windows are excluded, without regrouping trains.
    """
    from chipmunk import Chipmunk

    from thesis.ephys.events import fetch_session_events
    from thesis.ephys.trials import _replay_trials, _trial_start_indices

    trial_numbers = np.asarray(
        Chipmunk.trial_query(subject_name=subject, session_name=session).fetch(
            "trial_num", order_by="trial_num"
        )
    )
    if not trial_numbers.size or not np.array_equal(
        trial_numbers, np.arange(len(trial_numbers))
    ):
        raise ValueError(
            f"{subject} {session}: task-only stimulus selection requires "
            "nonempty contiguous zero-based Chipmunk task trial metadata"
        )
    session_events, pulses = fetch_session_events(subject, session)
    starts = np.asarray(session_events["trial_start"], dtype=float)
    task_indices = _trial_start_indices(
        starts,
        len(trial_numbers),
        len(_replay_trials(subject, session)),
        subject,
        session,
    )[: len(trial_numbers)]
    if not task_indices.size:
        raise ValueError(f"{subject} {session}: no aligned saved task starts")
    bounds = np.r_[starts, np.inf]
    selected = np.zeros(len(pulses), dtype=bool)
    for index in task_indices:
        selected |= (pulses["timestamp"] >= bounds[index]) & (
            pulses["timestamp"] < bounds[index + 1]
        )
    return pulses.loc[selected].copy()
