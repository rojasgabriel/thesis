from __future__ import annotations

import numpy as np
import datajoint as dj
from labdata.schema import (
    DecisionTask,
    Subject,  # noqa: F401 - referenced by DataJoint definitions
    get_user_schema,
)


rojasbowe_schema = get_user_schema()

TRIALSET_KEY_FIELDS = (
    "subject_name",
    "session_name",
    "dataset_name",
    "trialset_description",
)


@rojasbowe_schema
class BehaviorAnalysisSet(dj.Manual):
    """A curated set of upstream DecisionTask trial sets."""

    definition = """
    analysis_set_id                     : varchar(64)
    ---
    analysis_set_name                   : varchar(64)
    analysis_set_description = NULL     : varchar(512)
    performance_threshold = NULL        : float
    min_trials_with_choice = 0          : int
    selection_version                   : varchar(32)
    """

    class TrialSet(dj.Part):
        definition = """
        -> master
        -> DecisionTask.TrialSet
        ---
        include_reason = NULL            : varchar(256)
        """


@rojasbowe_schema
class PsychometricFitConfig(dj.Lookup):
    """Versioned eligibility settings for psychometric fits."""

    definition = """
    psychometric_fit_config_id           : varchar(48)
    ---
    min_choices                          : int
    min_stim_values                      : int
    analysis_version                     : varchar(32)
    """
    contents = [("v1", 100, 6, "v1")]  # noqa: RUF012


@rojasbowe_schema
class PsychometricSessionFit(dj.Computed):
    """One psychometric fit per upstream trial set and fit configuration."""

    definition = """
    -> DecisionTask.TrialSet
    -> PsychometricFitConfig
    ---
    fit_status                           : enum('fit', 'skipped')
    fit_message = NULL                   : varchar(256)
    n_choices_fit                        : int
    stims = NULL                         : longblob  # boundary-centered stimulus rate (Hz)
    p_right = NULL                       : longblob
    p_right_ci = NULL                    : longblob
    n_right = NULL                       : longblob
    n_obs = NULL                         : longblob
    bias = NULL                          : float
    sensitivity = NULL                   : float
    guess_rate = NULL                    : float
    lapse_rate = NULL                    : float
    goodness_of_fit = NULL               : float
    """

    @property
    def key_source(self):
        selected = DecisionTask.TrialSet() & BehaviorAnalysisSet.TrialSet()
        return selected * PsychometricFitConfig()

    def make(self, key):
        row = (DecisionTask.TrialSet() & _trialset_key(key)).fetch1()
        config = (PsychometricFitConfig() & key).fetch1()
        self.insert1({**key, **_psychometric_fit_payload(row, config)})


@rojasbowe_schema
class PsychometricSubjectFit(dj.Computed):
    """One pooled psychometric fit per analysis set, subject, condition, and config."""

    definition = """
    -> BehaviorAnalysisSet
    -> Subject
    trialset_description                 : varchar(54)
    -> PsychometricFitConfig
    ---
    fit_status                           : enum('fit', 'skipped')
    fit_message = NULL                   : varchar(256)
    n_choices_fit                        : int
    stims = NULL                         : longblob  # boundary-centered stimulus rate (Hz)
    p_right = NULL                       : longblob
    p_right_ci = NULL                    : longblob
    n_right = NULL                       : longblob
    n_obs = NULL                         : longblob
    bias = NULL                          : float
    sensitivity = NULL                   : float
    guess_rate = NULL                    : float
    lapse_rate = NULL                    : float
    goodness_of_fit = NULL               : float
    """

    @property
    def key_source(self):
        subject_conditions = (
            dj.U("analysis_set_id", "subject_name", "trialset_description")
            & BehaviorAnalysisSet.TrialSet()
        )
        return subject_conditions * PsychometricFitConfig()

    def make(self, key):
        rows = _fetch_trialset_rows_for_subject(key)
        intensity_values = np.concatenate(
            [np.asarray(row["intensity_values"], dtype=float) for row in rows]
        )
        response_values = np.concatenate(
            [np.asarray(row["response_values"], dtype=float) for row in rows]
        )
        config = (PsychometricFitConfig() & key).fetch1()
        self.insert1(
            {
                **key,
                **_psychometric_fit_payload(
                    {
                        "intensity_values": intensity_values,
                        "response_values": response_values,
                    },
                    config,
                ),
            }
        )


@rojasbowe_schema
class PsychophysicalKernelFitConfig(dj.Lookup):
    """Versioned settings for pooled psychophysical-kernel fits."""

    definition = """
    kernel_fit_config_id                 : varchar(48)
    ---
    timebins                             : int
    cv_splits                            : int
    random_state                         : int
    max_rate_hz                          : float  # calibration rate
    regularization_c                     : float
    kernel_method = 'legacy_variable'    : enum('legacy_variable', 'fixed_window')
    bin_width_s = NULL                   : float
    evidence_encoding = NULL             : enum('max_rate', 'trial_rate')
    min_trials_per_bin = NULL            : int
    observation_window = NULL            : enum('center_exit', 'response')
    analysis_version                     : varchar(32)
    """
    contents = [  # noqa: RUF012
        (
            "v1_10bin_10fold",
            10,
            10,
            0,
            20.0,
            1.0,
            "legacy_variable",
            None,
            None,
            None,
            None,
            "v1",
        ),
        (
            "v2_100ms_10bin_center_rate",
            10,
            10,
            0,
            20.0,
            1.0,
            "fixed_window",
            0.1,
            "trial_rate",
            50,
            "center_exit",
            "v2",
        ),
        (
            "v2_100ms_10bin_response_rate",
            10,
            10,
            0,
            20.0,
            1.0,
            "fixed_window",
            0.1,
            "trial_rate",
            50,
            "response",
            "v2",
        ),
    ]


@rojasbowe_schema
class PsychophysicalKernel(dj.Computed):
    """One pooled kernel per analysis set, subject, condition, and config."""

    definition = """
    -> BehaviorAnalysisSet
    -> Subject
    trialset_description                 : varchar(54)
    -> PsychophysicalKernelFitConfig
    ---
    fit_status                           : enum('fit', 'skipped')
    fit_message = NULL                   : varchar(256)
    timing_source = 'bpod'               : enum('nidaq', 'bpod', 'mixed')
    n_trials_fit                         : int
    n_bins_fit = NULL                    : int
    n_observed_per_bin = NULL            : longblob
    bin_centers_s = NULL                 : longblob  # time from first flash
    weights = NULL                       : longblob  # cv fold x stimulus time bin
    weights_mean = NULL                  : longblob
    weights_error = NULL                 : longblob
    scores = NULL                        : longblob  # held-out accuracy by fold
    score_mean = NULL                    : float
    majority_accuracy = NULL             : float
    score_above_majority = NULL          : float
    bias = NULL                          : longblob  # intercept by fold
    bias_mean = NULL                     : float
    interpretation = NULL                : varchar(32)
    """

    @property
    def key_source(self):
        subject_conditions = (
            dj.U("analysis_set_id", "subject_name", "trialset_description")
            & BehaviorAnalysisSet.TrialSet()
        )
        return subject_conditions * PsychophysicalKernelFitConfig()

    def make(self, key):
        config = (PsychophysicalKernelFitConfig() & key).fetch1()
        trialset_keys = _selected_trialset_keys(key)
        if config["kernel_method"] == "legacy_variable":
            payload = _legacy_kernel_payload(key, config, trialset_keys)
        else:
            payload = _fixed_kernel_payload(key, config, trialset_keys)
        self.insert1(
            {
                **key,
                **payload,
            }
        )


def _trialset_key(key):
    return {field: key[field] for field in TRIALSET_KEY_FIELDS}


def _selected_trialset_keys(key):
    selection_key = {
        field: key[field]
        for field in ("analysis_set_id", "subject_name", "trialset_description")
    }
    return list(
        (BehaviorAnalysisSet.TrialSet() & selection_key).fetch(
            *TRIALSET_KEY_FIELDS, as_dict=True
        )
    )


def _fetch_trialset_rows_for_subject(key):
    trialset_keys = _selected_trialset_keys(key)
    return list((DecisionTask.TrialSet() & trialset_keys).fetch(as_dict=True))


def _legacy_kernel_payload(key, config, trialset_keys):
    from behavior_analyses.io import get_chipmunk_table
    from behavior_analyses.kernels import (
        fit_psychophysical_kernel,
        interpret_kernel_weights,
    )

    Chipmunk = get_chipmunk_table()
    relation = (
        Chipmunk() * Chipmunk.Trial() * Chipmunk.TrialParameters()
        & trialset_keys
        & {"rewarded_modality": key["trialset_description"]}
    )
    stim_events, response_values = relation.fetch("stim_events", "response")
    result = fit_psychophysical_kernel(
        stim_events,
        response_values,
        timebins=int(config["timebins"]),
        cv_splits=int(config["cv_splits"]),
        random_state=int(config["random_state"]),
        max_rate_hz=float(config["max_rate_hz"]),
        regularization_c=float(config["regularization_c"]),
    )
    n_trials_fit = int(result["choice_right"].size)
    if result["weights"].size == 0:
        return {
            "fit_status": "skipped",
            "fit_message": "insufficient trials or response classes for CV",
            "timing_source": "bpod",
            "n_trials_fit": n_trials_fit,
            "n_bins_fit": 0,
        }

    n_observed = np.full(int(config["timebins"]), n_trials_fit, dtype=int)
    score_mean = float(np.mean(result["scores"]))
    majority_accuracy = float(
        max(np.mean(result["choice_right"]), 1.0 - np.mean(result["choice_right"]))
    )
    weights_mean = np.mean(result["weights"], axis=0)
    return {
        "fit_status": "fit",
        "timing_source": "bpod",
        "n_trials_fit": n_trials_fit,
        "n_bins_fit": int(config["timebins"]),
        "n_observed_per_bin": n_observed,
        "weights": result["weights"],
        "weights_mean": weights_mean,
        "weights_error": np.mean(result["error"], axis=0),
        "scores": result["scores"],
        "score_mean": score_mean,
        "majority_accuracy": majority_accuracy,
        "score_above_majority": score_mean - majority_accuracy,
        "bias": result["bias"],
        "bias_mean": float(np.mean(result["bias"])),
        "interpretation": interpret_kernel_weights(
            weights_mean,
            n_observed,
            min_trials_per_bin=int(config["cv_splits"]),
        ),
    }


def _fixed_kernel_payload(key, config, trialset_keys):
    from behavior_analyses.kernel_timing import fetch_pooled_kernel_inputs
    from behavior_analyses.kernels import (
        build_fixed_residual_rate_matrix,
        fit_fixed_psychophysical_kernel,
        interpret_kernel_weights,
    )

    inputs = fetch_pooled_kernel_inputs(
        trialset_keys,
        key["trialset_description"],
        observation_window=str(config["observation_window"]),
    )
    trial_rate_hz = (
        inputs["trial_rate_hz"] if config["evidence_encoding"] == "trial_rate" else None
    )
    residual, choices, n_observed, bin_centers, expected_counts = (
        build_fixed_residual_rate_matrix(
            inputs["stim_times_per_trial"],
            inputs["first_stim_times"],
            inputs["observation_end_times"],
            inputs["response_values"],
            timebins=int(config["timebins"]),
            bin_width_s=float(config["bin_width_s"]),
            max_rate_hz=float(config["max_rate_hz"]),
            trial_rate_hz=trial_rate_hz,
        )
    )
    result = fit_fixed_psychophysical_kernel(
        residual,
        choices,
        n_observed_per_bin=n_observed,
        cv_splits=int(config["cv_splits"]),
        random_state=int(config["random_state"]),
        min_trials_per_bin=int(config["min_trials_per_bin"]),
        regularization_c=float(config["regularization_c"]),
        expected_counts=(
            expected_counts if config["evidence_encoding"] == "trial_rate" else None
        ),
    )
    base = {
        "timing_source": inputs["timing_source"],
        "n_trials_fit": int(result["n_trials_fit"]),
        "n_bins_fit": int(result["n_bins_fit"]),
        "n_observed_per_bin": n_observed,
        "bin_centers_s": bin_centers,
    }
    if not result["fit_converged"]:
        return {
            **base,
            "fit_status": "skipped",
            "fit_message": "insufficient trials or response classes for CV",
        }

    return {
        **base,
        "fit_status": "fit",
        "weights": result["weights"],
        "weights_mean": result["weights_mean"],
        "weights_error": result["weights_error"],
        "scores": result["scores"],
        "score_mean": result["score_mean"],
        "majority_accuracy": result["majority_accuracy"],
        "score_above_majority": result["score_above_majority"],
        "bias": result["bias"],
        "bias_mean": result["bias_mean"],
        "interpretation": interpret_kernel_weights(
            result["weights_mean"],
            n_observed,
            min_trials_per_bin=int(config["min_trials_per_bin"]),
        ),
    }


def _psychometric_fit_payload(row, config):
    from behavior_analyses.psychometrics import fit_psychometric_labdata

    intensity_values = np.asarray(row["intensity_values"], dtype=float)
    response_values = np.asarray(row["response_values"], dtype=float)
    valid_choice = np.isfinite(intensity_values) & np.isin(response_values, [-1, 1])
    n_choices_fit = int(np.sum(valid_choice))
    fit = fit_psychometric_labdata(
        intensity_values,
        response_values,
        min_choices=int(config["min_choices"]),
        min_required_stim_values=int(config["min_stim_values"]),
    )
    if fit is None:
        return {
            "fit_status": "skipped",
            "fit_message": "insufficient choices, stimulus values, or fit convergence",
            "n_choices_fit": n_choices_fit,
        }
    return {
        "fit_status": "fit",
        "n_choices_fit": n_choices_fit,
        **{
            field: fit[field]
            for field in (
                "stims",
                "p_right",
                "p_right_ci",
                "n_right",
                "n_obs",
                "bias",
                "sensitivity",
                "guess_rate",
                "lapse_rate",
                "goodness_of_fit",
            )
        },
    }
