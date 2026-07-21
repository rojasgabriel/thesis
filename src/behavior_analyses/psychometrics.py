from __future__ import annotations

import numpy as np
from fit_psychometric import cumulative_gaussian, fit_psychometric

MIN_CHOICES = 100
MIN_STIM_VALUES = 6


def fit_psychometric_labdata(
    stim_values,
    response_values,
    *,
    min_choices: int = MIN_CHOICES,
    min_required_stim_values: int = MIN_STIM_VALUES,
):
    """Fit psychometrics using the external fitter and labdata response coding."""
    stim_values = np.asarray(stim_values, dtype=float)
    response_values = np.asarray(response_values, dtype=float)
    valid_choice = np.isfinite(stim_values) & np.isin(response_values, [-1, 1])
    stim_values = stim_values[valid_choice]
    response_values = response_values[valid_choice]
    if response_values.size < min_choices:
        return None

    choice_right = (response_values == 1).astype(float)
    fit_result = fit_psychometric(
        stim_values,
        choice_right,
        min_required_stim_values=min_required_stim_values,
    )
    if fit_result["fit_params"] is None:
        return None

    params = np.asarray(fit_result["fit_params"], dtype=float)
    predicted = fit_result["function"](*params, np.asarray(fit_result["stims"]))
    goodness_of_fit = _r_squared(np.asarray(fit_result["p_side"]), predicted)
    return {
        "stims": np.asarray(fit_result["stims"], dtype=float),
        "p_side": np.asarray(fit_result["p_side"], dtype=float),
        "p_right": np.asarray(fit_result["p_side"], dtype=float),
        "p_side_ci": np.asarray(fit_result["p_side_ci"], dtype=float),
        "p_right_ci": np.asarray(fit_result["p_side_ci"], dtype=float),
        "n_side": np.asarray(fit_result["n_side"], dtype=float),
        "n_right": np.asarray(fit_result["n_side"], dtype=float),
        "n_obs": np.asarray(fit_result["n_obs"], dtype=float),
        "bias": float(params[0]),
        "sensitivity": float(params[1]),
        "guess_rate": float(params[2]),
        "lapse_rate": float(params[3]),
        "goodness_of_fit": goodness_of_fit,
        "fit_params": params,
    }


def _r_squared(observed, predicted) -> float:
    ss_res = np.sum((observed - predicted) ** 2)
    ss_tot = np.sum((observed - np.mean(observed)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1 - (ss_res / ss_tot))
