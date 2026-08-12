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
    """Settings for pooled psychophysical-kernel fits."""

    definition = """
    kernel_fit_config_id                 : int
    ---
    timebins                             : int
    binning_method                       : enum('fixed_width')
    bin_width_s                          : float
    observation_window                   : enum('center_exit', 'response')
    evidence_model                       : enum('trial_rate_residual')  # Odoemene Eq. 5
    min_trials_per_bin                   : int
    cv_splits                            : int
    random_state                         : int
    regularization_c                     : float
    """
    contents = [  # noqa: RUF012
        (
            0,
            10,
            "fixed_width",
            0.1,
            "center_exit",
            "trial_rate_residual",
            50,
            10,
            0,
            1.0,
        ),
        (
            1,
            10,
            "fixed_width",
            0.1,
            "response",
            "trial_rate_residual",
            50,
            10,
            0,
            1.0,
        ),
    ]


@rojasbowe_schema
class PsychophysicalKernel(dj.Computed):
    """One pooled kernel per analysis set, subject, condition, config, and clock."""

    definition = """
    -> BehaviorAnalysisSet
    -> Subject
    trialset_description                 : varchar(54)
    -> PsychophysicalKernelFitConfig
    timing_source                        : enum('nidq', 'bpod')
    ---
    fit_status                           : enum('fit', 'skipped')
    fit_message = NULL                   : varchar(256)
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
            .aggr(BehaviorAnalysisSet.TrialSet(), n_trialsets="count(*)")
            .proj()
        )
        base = subject_conditions * PsychophysicalKernelFitConfig()
        key_fields = (
            "analysis_set_id",
            "subject_name",
            "trialset_description",
            "kernel_fit_config_id",
        )
        nidq_keys = []
        for row in base.fetch(as_dict=True):
            trialset_keys = _selected_trialset_keys(row)
            from behavior_analyses.kernel_timing import available_timing_sources

            if "nidq" in available_timing_sources(trialset_keys):
                nidq_keys.append({field: row[field] for field in key_fields})

        key_relation = dj.U(*key_fields, "timing_source")
        bpod = key_relation & base.proj(*key_fields, timing_source="'bpod'")
        if not nidq_keys:
            return bpod
        nidq = key_relation & (base & nidq_keys).proj(
            *key_fields, timing_source="'nidq'"
        )
        return bpod + nidq

    def make(self, key):
        config = (PsychophysicalKernelFitConfig() & key).fetch1()
        trialset_keys = _selected_trialset_keys(key)
        payload = _kernel_payload(key, config, trialset_keys)
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


def _kernel_payload(key, config, trialset_keys):
    from behavior_analyses.kernel_timing import fetch_pooled_kernel_inputs
    from behavior_analyses.kernels import (
        build_residual_rate_matrix,
        fit_psychophysical_kernel,
        interpret_kernel_weights,
    )

    inputs = fetch_pooled_kernel_inputs(
        trialset_keys,
        key["trialset_description"],
        observation_window=str(config["observation_window"]),
        timing_source=str(key["timing_source"]),
    )
    residual, choices, n_observed, bin_centers, expected_counts = (
        build_residual_rate_matrix(
            inputs["stim_times_per_trial"],
            inputs["first_stim_times"],
            inputs["observation_end_times"],
            inputs["response_values"],
            timebins=int(config["timebins"]),
            bin_width_s=float(config["bin_width_s"]),
            trial_rate_hz=inputs["trial_rate_hz"],
        )
    )
    result = fit_psychophysical_kernel(
        residual,
        choices,
        n_observed_per_bin=n_observed,
        cv_splits=int(config["cv_splits"]),
        random_state=int(config["random_state"]),
        min_trials_per_bin=int(config["min_trials_per_bin"]),
        regularization_c=float(config["regularization_c"]),
        expected_counts=expected_counts,
    )
    base = {
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
