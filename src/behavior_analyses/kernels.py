from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


def build_residual_rate_matrix(
    stim_times_per_trial: Sequence[np.ndarray],
    first_stim_times: Sequence[float],
    observation_end_times: Sequence[float],
    response_values: Sequence[Any],
    trial_rate_hz: Sequence[float],
    *,
    timebins: int = 10,
    bin_width_s: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build Odoemene Eq. 5 residual counts in fixed-width time bins."""
    if timebins < 1:
        raise ValueError("timebins must be >= 1")
    if bin_width_s <= 0:
        raise ValueError("bin_width_s must be > 0")

    responses = np.asarray(response_values)
    choice_mask = np.isin(responses, (-1, 1))
    choices = (responses[choice_mask] == 1).astype(int)
    stim_list = [
        np.asarray(times, dtype=float)
        for times, keep in zip(stim_times_per_trial, choice_mask)
        if keep
    ]
    first_stim = np.asarray(first_stim_times, dtype=float)[choice_mask]
    observation_end = np.asarray(observation_end_times, dtype=float)[choice_mask]

    expected_rates = np.asarray(trial_rate_hz, dtype=float)
    if expected_rates.shape != choice_mask.shape:
        raise ValueError("trial_rate_hz must match response_values")
    if np.any(~np.isfinite(expected_rates)) or np.any(expected_rates < 0):
        raise ValueError("trial_rate_hz must contain finite nonnegative rates")
    expected_rates = expected_rates[choice_mask]

    bin_edges = np.arange(timebins + 1, dtype=float) * bin_width_s
    bin_centers_s = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    residual = np.full((len(stim_list), timebins), np.nan, dtype=float)
    expected_counts = np.full_like(residual, np.nan)

    for trial_index, (stims, first, observation_end, expected_rate) in enumerate(
        zip(stim_list, first_stim, observation_end, expected_rates)
    ):
        if (
            not np.isfinite(first)
            or not np.isfinite(observation_end)
            or observation_end <= first
        ):
            continue

        duration = float(observation_end - first)
        relative_stims = stims[np.isfinite(stims)] - first
        relative_stims = relative_stims[
            (relative_stims >= 0.0) & (relative_stims < duration)
        ]
        for bin_index in range(timebins):
            bin_start = bin_edges[bin_index]
            if bin_start >= duration:
                continue
            observed_end = min(bin_edges[bin_index + 1], duration)
            observed_duration = observed_end - bin_start
            count = np.sum(
                (relative_stims >= bin_start) & (relative_stims < observed_end)
            )
            expected = expected_rate * observed_duration
            residual[trial_index, bin_index] = float(count - expected)
            expected_counts[trial_index, bin_index] = expected

    n_observed_per_bin = np.sum(np.isfinite(residual), axis=0).astype(int)
    return residual, choices, n_observed_per_bin, bin_centers_s, expected_counts


def fit_psychophysical_kernel(
    residual: np.ndarray,
    choice_right: np.ndarray,
    *,
    expected_counts: np.ndarray,
    n_observed_per_bin: np.ndarray | None = None,
    cv_splits: int = 10,
    random_state: int = 0,
    min_trials_per_bin: int = 50,
    regularization_c: float = 1.0,
) -> dict[str, Any]:
    """Fit the Odoemene Eq. 5 kernel over the longest complete-case prefix."""
    residual = np.asarray(residual, dtype=float)
    choice_right = np.asarray(choice_right, dtype=int)
    n_bins = residual.shape[1] if residual.ndim == 2 else 0
    if n_observed_per_bin is None:
        n_observed_per_bin = np.sum(np.isfinite(residual), axis=0).astype(int)
    else:
        n_observed_per_bin = np.asarray(n_observed_per_bin, dtype=int)

    empty = {
        "choice_right": choice_right,
        "weights": np.empty((0, n_bins)),
        "weights_mean": np.full(n_bins, np.nan),
        "weights_error": np.full(n_bins, np.nan),
        "scores": np.empty((0,)),
        "score_mean": np.nan,
        "majority_accuracy": np.nan,
        "score_above_majority": np.nan,
        "bias": np.empty((0,)),
        "bias_mean": np.nan,
        "n_observed_per_bin": n_observed_per_bin,
        "n_trials_fit": 0,
        "n_bins_fit": 0,
        "fit_converged": False,
    }
    if residual.size == 0 or choice_right.size == 0 or np.unique(choice_right).size < 2:
        return empty

    n_bins_fit = _complete_case_prefix(
        residual,
        n_observed_per_bin,
        min_trials_per_bin=min_trials_per_bin,
    )
    if n_bins_fit < 1:
        return empty

    complete = np.all(np.isfinite(residual[:, :n_bins_fit]), axis=1)
    x = residual[complete, :n_bins_fit]
    y = choice_right[complete]
    if x.shape[0] < max(cv_splits, min_trials_per_bin) or np.unique(y).size < 2:
        return empty
    n_splits = int(min(cv_splits, np.min(np.bincount(y))))
    if n_splits < 2:
        return empty

    design, coefficient_to_weights = _kernel_design(
        residual,
        x,
        complete,
        n_bins_fit,
        expected_counts,
    )
    splitter = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    weights = []
    errors = []
    scores = []
    biases = []
    for train_index, test_index in splitter.split(design, y):
        x_train, x_test = design[train_index], design[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model = LogisticRegression(
            solver="liblinear",
            C=regularization_c,
            fit_intercept=True,
        ).fit(x_train, y_train)
        predict_prob = model.predict_proba(x_train)
        variance = np.prod(predict_prob, axis=1)
        covariance = np.linalg.pinv(np.dot(x_train.T * variance, x_train))

        weight_full = np.full(n_bins, np.nan, dtype=float)
        error_full = np.full(n_bins, np.nan, dtype=float)
        weight_full[:n_bins_fit] = coefficient_to_weights @ model.coef_[0]
        weight_covariance = (
            coefficient_to_weights @ covariance @ coefficient_to_weights.T
        )
        error_full[:n_bins_fit] = np.sqrt(np.diag(weight_covariance))
        weights.append(weight_full)
        errors.append(error_full)
        scores.append(model.score(x_test, y_test))
        biases.append(float(model.intercept_[0]))

    weights = np.asarray(weights, dtype=float)
    errors = np.asarray(errors, dtype=float)
    scores = np.asarray(scores, dtype=float)
    biases = np.asarray(biases, dtype=float)
    weights_mean = np.full(n_bins, np.nan)
    weights_error = np.full(n_bins, np.nan)
    weights_mean[:n_bins_fit] = np.mean(weights[:, :n_bins_fit], axis=0)
    weights_error[:n_bins_fit] = np.mean(errors[:, :n_bins_fit], axis=0)
    score_mean = float(np.mean(scores))
    majority_accuracy = float(max(np.mean(y), 1.0 - np.mean(y)))
    return {
        "choice_right": choice_right,
        "weights": weights,
        "weights_mean": weights_mean,
        "weights_error": weights_error,
        "scores": scores,
        "score_mean": score_mean,
        "majority_accuracy": majority_accuracy,
        "score_above_majority": score_mean - majority_accuracy,
        "bias": biases,
        "bias_mean": float(np.mean(biases)),
        "n_observed_per_bin": n_observed_per_bin,
        "n_trials_fit": int(x.shape[0]),
        "n_bins_fit": int(n_bins_fit),
        "fit_converged": True,
    }


def interpret_kernel_weights(
    weights_mean: np.ndarray,
    n_observed_per_bin: np.ndarray,
    *,
    min_trials_per_bin: int = 50,
    ratio_threshold: float = 1.5,
) -> str:
    """Return a descriptive early, late, flat, or failed kernel label."""
    weights_mean = np.asarray(weights_mean, dtype=float)
    n_observed_per_bin = np.asarray(n_observed_per_bin, dtype=int)
    usable = np.isfinite(weights_mean) & (n_observed_per_bin >= min_trials_per_bin)
    if int(np.sum(usable)) < 3:
        return "failed_fit"

    values = np.abs(weights_mean[usable])
    midpoint = max(1, values.size // 2)
    early = float(np.mean(values[:midpoint]))
    late = float(np.mean(values[midpoint:]))
    if early == 0:
        ratio = np.inf if late > 0 else 1.0
    else:
        ratio = late / early
    if ratio > ratio_threshold:
        return "late_integrator"
    if ratio < 1.0 / ratio_threshold:
        return "early_integrator"
    return "flat_indeterminate"


def _complete_case_prefix(
    residual: np.ndarray,
    n_observed_per_bin: np.ndarray,
    *,
    min_trials_per_bin: int,
) -> int:
    prefix = 0
    for bin_index in range(residual.shape[1]):
        complete = np.all(np.isfinite(residual[:, : bin_index + 1]), axis=1)
        if (
            int(n_observed_per_bin[bin_index]) < min_trials_per_bin
            or int(np.sum(complete)) < min_trials_per_bin
        ):
            break
        prefix = bin_index + 1
    return prefix


def _kernel_design(
    residual: np.ndarray,
    complete_residual: np.ndarray,
    complete: np.ndarray,
    n_bins_fit: int,
    expected_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    expected_counts = np.asarray(expected_counts, dtype=float)
    if expected_counts.shape != residual.shape:
        raise ValueError("expected_counts must match residual")
    expected_fit = expected_counts[complete, :n_bins_fit]
    if np.any(~np.isfinite(expected_fit)):
        raise ValueError("expected_counts must be finite for fitted bins")

    mean_rate = np.sum(expected_fit, axis=1)
    if n_bins_fit == 1:
        return mean_rate[:, None], np.ones((1, 1))

    basis_source = np.column_stack(
        [
            np.eye(n_bins_fit)[:, index] - np.eye(n_bins_fit)[:, -1]
            for index in range(n_bins_fit - 1)
        ]
    )
    basis = np.linalg.qr(basis_source, mode="reduced")[0]
    design = np.column_stack((complete_residual @ basis, mean_rate))
    coefficient_to_weights = np.column_stack((basis, np.ones(n_bins_fit)))
    return design, coefficient_to_weights
