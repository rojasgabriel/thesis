from __future__ import annotations

import numpy as np
import datajoint as dj
from labdata.schema import DecisionTask, Session, Subject, get_user_schema


rojasbowe_schema = get_user_schema()


@rojasbowe_schema
class BehaviorSessionSet(dj.Manual):
    definition = """
    session_set_id                      : varchar(64)
    ---
    session_set_name                    : varchar(64)
    session_set_description = NULL      : varchar(512)
    performance_threshold = NULL        : float
    min_trials_with_choice = 0           : int
    goal_wait_time_min = NULL            : float
    goal_wait_time_max = NULL            : float
    kernel_timebins = 10                 : int
    kernel_cv_splits = 10                : int
    kernel_random_state = 0              : int
    analysis_version                    : varchar(32)
    """

    class Session(dj.Part):
        definition = """
        -> master
        -> Session
        ---
        include_reason = NULL            : varchar(256)
        """

    class TrialSet(dj.Part):
        definition = """
        -> master
        -> DecisionTask.TrialSet
        ---
        include_reason = NULL            : varchar(256)
        """

    class SubjectTrialSet(dj.Part):
        definition = """
        -> master
        -> Subject
        trialset_description             : varchar(54)
        ---
        n_sessions                       : int
        """


@rojasbowe_schema
class LearningSessionMetrics(dj.Computed):
    definition = """
    -> BehaviorSessionSet.TrialSet
    ---
    n_trials                            : int
    n_with_choice                       : int
    n_correct                           : int
    performance = NULL                  : float
    performance_easy = NULL             : float
    mean_initiation_time = NULL         : float
    mean_reaction_time = NULL           : float
    stim_values                         : longblob
    response_values                     : longblob
    correct_values                      : longblob
    intensity_values                    : longblob
    """

    def make(self, key):
        from behavior_analyses.learning import summarize_trialset

        row = (DecisionTask.TrialSet() & key).fetch1()
        summary = summarize_trialset(row)
        self.insert1({**key, **summary})


@rojasbowe_schema
class PsychometricSessionFit(dj.Computed):
    definition = """
    -> BehaviorSessionSet.TrialSet
    ---
    stims                               : longblob
    p_side                              : longblob  # legacy p(right)
    p_right                             : longblob
    p_side_ci                           : longblob
    p_right_ci                          : longblob
    n_side                              : longblob  # legacy n(right)
    n_right                             : longblob
    n_obs                               : longblob
    bias                                : float
    sensitivity                         : float
    guess_rate                          : float
    lapse_rate                          : float
    goodness_of_fit                     : float
    fit_params                          : longblob
    """

    def make(self, key):
        from behavior_analyses.psychometrics import fit_psychometric_labdata

        row = (DecisionTask.TrialSet() & key).fetch1()
        fit = fit_psychometric_labdata(row["intensity_values"], row["response_values"])
        if fit is None:
            return
        self.insert1({**key, **fit})


@rojasbowe_schema
class PsychometricSubjectFit(dj.Computed):
    definition = """
    -> BehaviorSessionSet.SubjectTrialSet
    ---
    n_sessions                          : int
    n_trials                            : int
    stims                               : longblob
    p_side                              : longblob  # legacy p(right)
    p_right                             : longblob
    p_side_ci                           : longblob
    p_right_ci                          : longblob
    n_side                              : longblob  # legacy n(right)
    n_right                             : longblob
    n_obs                               : longblob
    bias                                : float
    sensitivity                         : float
    guess_rate                          : float
    lapse_rate                          : float
    goodness_of_fit                     : float
    fit_params                          : longblob
    """

    def make(self, key):
        from behavior_analyses.psychometrics import fit_psychometric_labdata

        rows = _fetch_trialset_rows_for_subject(key)
        if not rows:
            return
        intensity_values = np.concatenate(
            [np.asarray(row["intensity_values"], dtype=float) for row in rows]
        )
        response_values = np.concatenate(
            [np.asarray(row["response_values"], dtype=float) for row in rows]
        )
        fit = fit_psychometric_labdata(intensity_values, response_values)
        if fit is None:
            return
        self.insert1(
            {
                **key,
                "n_sessions": len(rows),
                "n_trials": int(response_values.size),
                **fit,
            }
        )


@rojasbowe_schema
class PsychophysicalKernel(dj.Computed):
    definition = """
    -> BehaviorSessionSet.SubjectTrialSet
    ---
    n_sessions                          : int
    n_trials                            : int
    timebins                            : int
    cv_splits                           : int
    random_state                        : int
    weights                             : longblob
    weights_mean                        : longblob
    weights_error                       : longblob
    scores                              : longblob
    score_mean                          : float
    bias                                : longblob
    bias_mean                           : float
    """

    def make(self, key):
        from behavior_analyses.io import get_chipmunk_table
        from behavior_analyses.kernels import fit_psychophysical_kernel

        set_params = (BehaviorSessionSet() & key).fetch1(
            "kernel_timebins", "kernel_cv_splits", "kernel_random_state"
        )
        timebins, cv_splits, random_state = [int(value) for value in set_params]
        session_rows = (BehaviorSessionSet.TrialSet() & key).fetch("KEY")
        if not session_rows:
            return

        Chipmunk = get_chipmunk_table()
        relation = (
            Chipmunk()
            * Chipmunk.Trial()
            * Chipmunk.TrialParameters()
            & session_rows
            & {"rewarded_modality": key["trialset_description"]}
        )
        stim_events, response_values = relation.fetch("stim_events", "response")
        result = fit_psychophysical_kernel(
            stim_events,
            response_values,
            timebins=timebins,
            cv_splits=cv_splits,
            random_state=random_state,
        )
        if result["weights"].size == 0:
            return
        self.insert1(
            {
                **key,
                "n_sessions": len(
                    {row["session_name"] for row in session_rows if "session_name" in row}
                ),
                "n_trials": int(result["choice_right"].size),
                "timebins": timebins,
                "cv_splits": cv_splits,
                "random_state": random_state,
                "weights": result["weights"],
                "weights_mean": np.mean(result["weights"], axis=0),
                "weights_error": np.mean(result["error"], axis=0),
                "scores": result["scores"],
                "score_mean": float(np.mean(result["scores"])),
                "bias": result["bias"],
                "bias_mean": float(np.mean(result["bias"])),
            }
        )


def _fetch_trialset_rows_for_subject(key):
    trialset_keys = (BehaviorSessionSet.TrialSet() & key).fetch("KEY")
    if not trialset_keys:
        return []
    return list((DecisionTask.TrialSet() & trialset_keys).fetch(as_dict=True))
