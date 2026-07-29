"""Psychophysical kernels for fluctuating visual flash-rate decisions.

Kernel = time-resolved weights from logistic regression of choice on
momentary evidence (reverse correlation). Early vs late weight profiles
distinguish impulsive / primacy-like vs flat / late strategies.

References:
 - Odoemene, Pisupati, Nguyen & Churchland 2018 doi:10.1523/JNEUROSCI.3478-17.2018
   (flash-rate task; logistic kernel; residualize rate when pooling)
 - Huk & Shadlen 2005 doi:10.1523/JNEUROSCI.4684-04.2005
 - Katz, Yates, Pillow & Huk 2016 doi:10.1038/ncomms13623
 - Yates, Park, Katz, Pillow & Huk 2017 doi:10.1038/nn.4611
 - Okazawa, She, Purcell & Kiani 2018 doi:10.1038/s41467-018-05797-y
   (kernels mix sensory weights with decision dynamics)

Local provenance: ``behavior_analyses/psychophysical_kernels/``.
Deviations from the notebook: fixed bins from first flash (not
per-trial linspace), NaN for unobserved late bins, observation window
``center_exit`` vs ``response``, expected count = max_rate * duration.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


def code_choice_right(
    response_values: Sequence[Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Map Chipmunk response to right=1 / left=0; drop no-choice trials.

    Matches current Chipmunk / ephys convention (``response == 1`` is right),
    as in Odoemene et al. 2018 high-rate → right contingencies.
    """
    responses = np.asarray(response_values)
    mask = np.isin(responses, (-1, 1))
    coded = (responses[mask] == 1).astype(int)
    return coded, mask


def build_residual_rate_matrix(
    stim_times_per_trial: Sequence[np.ndarray],
    first_stim_times: Sequence[float],
    observation_end_times: Sequence[float],
    response_values: Sequence[Any],
    *,
    timebins: int = 10,
    bin_width_s: float = 0.1,
    max_rate_hz: float = 20.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Trial × bin residual flash rate, with NaN after observation ends.

    Evidence in bin ``b`` (Odoemene et al. 2018 Eqs. 4–5 style):

        X_{t,b} = n_flashes_{t,b} - max_rate_hz * observed_duration_{t,b}

    Residualizing removes the mean-rate confound when trials span multiple
    generative rates. Bins the animal never reached are ``NaN``, not zero.

    Returns residual, choice_right, n_observed_per_bin, bin_centers_s.
    """
    if timebins < 1:
        raise ValueError("timebins must be >= 1")
    if bin_width_s <= 0:
        raise ValueError("bin_width_s must be > 0")

    choices, choice_mask = code_choice_right(response_values)
    stim_list = [
        np.asarray(times, dtype=float)
        for times, keep in zip(stim_times_per_trial, choice_mask)
        if keep
    ]
    first_stim = np.asarray(first_stim_times, dtype=float)[choice_mask]
    observation_end = np.asarray(observation_end_times, dtype=float)[choice_mask]

    bin_edges = np.arange(timebins + 1, dtype=float) * bin_width_s
    bin_centers_s = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    residual = np.full((len(stim_list), timebins), np.nan, dtype=float)

    for trial_idx, (stims, t0, t_end) in enumerate(
        zip(stim_list, first_stim, observation_end)
    ):
        if not np.isfinite(t0) or not np.isfinite(t_end) or t_end <= t0:
            continue
        wait_duration = float(t_end - t0)
        relative = np.asarray(stims, dtype=float) - t0
        relative = relative[(relative >= 0.0) & (relative < wait_duration)]
        for bin_idx in range(timebins):
            bin_start = bin_edges[bin_idx]
            bin_end = bin_edges[bin_idx + 1]
            if bin_start >= wait_duration:
                continue
            observed_end = min(bin_end, wait_duration)
            observed_duration = observed_end - bin_start
            if observed_duration <= 0:
                continue
            count = int(np.sum((relative >= bin_start) & (relative < observed_end)))
            # expected count under max_rate_hz over the observed fraction of the bin
            expected = max_rate_hz * observed_duration
            residual[trial_idx, bin_idx] = float(count - expected)

    n_observed_per_bin = np.sum(np.isfinite(residual), axis=0).astype(int)
    return residual, choices, n_observed_per_bin, bin_centers_s


def select_complete_case_prefix(
    residual: np.ndarray,
    n_observed_per_bin: np.ndarray,
    *,
    min_trials_per_bin: int,
) -> int:
    """Longest leading bin count usable for complete-case multivariate fit."""
    if residual.size == 0:
        return 0
    n_bins = residual.shape[1]
    prefix = 0
    for bin_idx in range(n_bins):
        if int(n_observed_per_bin[bin_idx]) < min_trials_per_bin:
            break
        complete = np.all(np.isfinite(residual[:, : bin_idx + 1]), axis=1)
        if int(np.sum(complete)) < min_trials_per_bin:
            break
        prefix = bin_idx + 1
    return prefix


def fit_psychophysical_kernel(
    residual: np.ndarray,
    choice_right: np.ndarray,
    *,
    n_observed_per_bin: np.ndarray | None = None,
    cv_splits: int = 10,
    random_state: int = 0,
    min_trials_per_bin: int = 50,
    regularization_C: float = 1.0,
) -> dict[str, Any]:
    """L2 logistic psychophysical kernel on residual-rate bins.

    Fits:

        logit P(right) = β0 + X w

    where ``w`` is the kernel (weight per time bin). Same logistic reverse-
    correlation approach as Odoemene et al. 2018 / Huk & Shadlen 2005.
    Incomplete late bins are handled by a complete-case prefix (never
    zero-filled). Coefficients beyond the fitted prefix stay ``NaN``.

    Stored metrics: CV weights, mean±error kernel, holdout scores, bias,
    n_observed_per_bin, n_trials_fit / n_bins_fit.
    """
    residual = np.asarray(residual, dtype=float)
    y = np.asarray(choice_right, dtype=int)
    n_bins = residual.shape[1] if residual.ndim == 2 else 0
    if n_observed_per_bin is None:
        n_observed_per_bin = np.sum(np.isfinite(residual), axis=0).astype(int)
    else:
        n_observed_per_bin = np.asarray(n_observed_per_bin, dtype=int)

    empty = {
        "design_matrix": residual,
        "choice_right": y,
        "weights": np.empty((0, n_bins)),
        "weights_mean": np.full(n_bins, np.nan),
        "weights_error": np.full(n_bins, np.nan),
        "scores": np.empty((0,)),
        "score_mean": np.nan,
        "bias": np.empty((0,)),
        "bias_mean": np.nan,
        "n_observed_per_bin": n_observed_per_bin,
        "n_trials_fit": 0,
        "n_bins_fit": 0,
        "fit_converged": False,
    }
    if residual.size == 0 or y.size == 0 or np.unique(y).size < 2:
        return empty

    n_bins_fit = select_complete_case_prefix(
        residual,
        n_observed_per_bin,
        min_trials_per_bin=min_trials_per_bin,
    )
    if n_bins_fit < 1:
        return empty

    complete = np.all(np.isfinite(residual[:, :n_bins_fit]), axis=1)
    x = residual[complete][:, :n_bins_fit]
    y_fit = y[complete]
    if x.shape[0] < max(cv_splits, min_trials_per_bin) or np.unique(y_fit).size < 2:
        return empty

    n_splits = int(min(cv_splits, np.min(np.bincount(y_fit))))
    if n_splits < 2:
        return empty

    splitter = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    weights = []
    scores = []
    biases = []
    errors = []
    for train_index, test_index in splitter.split(x, y_fit):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y_fit[train_index], y_fit[test_index]
        model = LogisticRegression(
            penalty="l2",
            solver="liblinear",
            C=regularization_C,
            fit_intercept=True,
        ).fit(x_train, y_train)
        predict_prob = model.predict_proba(x_train)
        variance = np.prod(predict_prob, axis=1)
        covariance = np.linalg.pinv(np.dot(x_train.T * variance, x_train))
        weight_full = np.full(n_bins, np.nan, dtype=float)
        error_full = np.full(n_bins, np.nan, dtype=float)
        weight_full[:n_bins_fit] = model.coef_[0]
        error_full[:n_bins_fit] = np.sqrt(np.diag(covariance))
        weights.append(weight_full)
        errors.append(error_full)
        scores.append(model.score(x_test, y_test))
        biases.append(float(model.intercept_[0]))

    weights_arr = np.asarray(weights, dtype=float)
    errors_arr = np.asarray(errors, dtype=float)
    scores_arr = np.asarray(scores, dtype=float)
    bias_arr = np.asarray(biases, dtype=float)
    return {
        "design_matrix": residual,
        "choice_right": y,
        "weights": weights_arr,
        "weights_mean": np.nanmean(weights_arr, axis=0),
        "weights_error": np.nanmean(errors_arr, axis=0),
        "scores": scores_arr,
        "score_mean": float(np.mean(scores_arr)),
        "bias": bias_arr,
        "bias_mean": float(np.mean(bias_arr)),
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
    """Coarse early/late/flat label from |kernel| halves.

    Matches the early-vs-late comparison in Odoemene et al. 2018 Fig. 2A/F.
    Labels are descriptive only — Okazawa et al. 2018: kernel shape also
    reflects decision dynamics, not pure sensory weighting.

    Returns ``early_integrator``, ``late_integrator``, ``flat_indeterminate``,
    or ``failed_fit``.
    """
    weights_mean = np.asarray(weights_mean, dtype=float)
    n_observed_per_bin = np.asarray(n_observed_per_bin, dtype=int)
    usable = np.isfinite(weights_mean) & (n_observed_per_bin >= min_trials_per_bin)
    if int(np.sum(usable)) < 3:
        return "failed_fit"
    values = np.abs(weights_mean[usable])
    mid = max(1, values.size // 2)
    early = float(np.mean(values[:mid]))
    late = float(np.mean(values[mid:]))
    if late > ratio_threshold * early:
        return "late_integrator"
    if early > ratio_threshold * late:
        return "early_integrator"
    return "flat_indeterminate"


OBSERVATION_WINDOWS = ("center_exit", "response")


def extract_trial_kernel_inputs(
    align_ev: Mapping[str, np.ndarray],
    trial_df,
    *,
    stim_key: str = "stim_ev",
    observation_window: str = "center_exit",
) -> dict[str, np.ndarray | list]:
    """Per-trial NIDAQ flash times and observation cutoffs for the kernel.

    ``observation_window``:
    - ``center_exit``: flashes until center-port exit (fixation only)
    - ``response``: flashes until chosen response-port poke

    Timing uses NIDAQ digital events (Bpod clocks can lag ~10–15 ms).
    """
    import pandas as pd

    if observation_window not in OBSERVATION_WINDOWS:
        raise ValueError(
            f"observation_window must be one of {OBSERVATION_WINDOWS}, "
            f"got {observation_window!r}"
        )

    stim_times = np.asarray(align_ev[stim_key], dtype=float)
    cp_entries = np.asarray(align_ev["center_port"], dtype=float)
    cp_exits = np.asarray(align_ev.get("center_port_exit", []), dtype=float)
    left_entries = np.asarray(align_ev["left_port"], dtype=float)
    right_entries = np.asarray(align_ev["right_port"], dtype=float)
    obx_trial_starts = np.asarray(align_ev["trial_start"], dtype=float)

    n = min(len(trial_df), len(obx_trial_starts))
    bpod_sync = trial_df["t_sync"].iloc[:n].to_numpy(dtype=float)
    obx_sync = obx_trial_starts[:n].astype(float)
    valid_sync = np.isfinite(bpod_sync) & np.isfinite(obx_sync)
    if int(np.sum(valid_sync)) < 2:
        raise ValueError(
            "Insufficient finite Bpod/NIDAQ sync points to interpolate trial timing."
        )
    bpod_sync = bpod_sync[valid_sync]
    obx_sync = obx_sync[valid_sync]
    t_react = trial_df["t_react"].iloc[:n].to_numpy(dtype=float)
    response = trial_df["response"].iloc[:n].to_numpy()
    cp_exit_obx = np.interp(t_react, bpod_sync, obx_sync)

    stim_times_per_trial: list[np.ndarray] = []
    first_stim_times: list[float] = []
    observation_end_times: list[float] = []
    response_values: list[int] = []
    wait_times: list[float] = []
    response_times: list[float] = []

    for i in range(n):
        if not np.isfinite(t_react[i]):
            continue
        if response[i] not in (-1, 1):
            continue

        trial_start = obx_trial_starts[i]
        trial_end = obx_trial_starts[i + 1] if i + 1 < len(obx_trial_starts) else np.inf
        trial_cp_exits = cp_exits[(cp_exits > trial_start) & (cp_exits < trial_end)]
        if trial_cp_exits.size:
            cp_exit = float(
                trial_cp_exits[np.argmin(np.abs(trial_cp_exits - cp_exit_obx[i]))]
            )
        else:
            cp_exit = float(cp_exit_obx[i])

        cp_mask = (
            (cp_entries > trial_start)
            & (cp_entries < cp_exit)
            & (cp_entries < trial_end)
        )
        if not cp_mask.any():
            continue
        cp_entry = float(cp_entries[cp_mask][-1])

        if int(response[i]) == 1:
            rp_pool = right_entries
        else:
            rp_pool = left_entries
        rp_mask = (rp_pool > cp_exit) & (rp_pool < trial_end)
        if not rp_mask.any():
            continue
        rp_entry = float(rp_pool[rp_mask][0])

        if observation_window == "center_exit":
            observation_end = cp_exit
        else:
            observation_end = rp_entry

        trial_stims = stim_times[
            (stim_times >= cp_entry) & (stim_times < observation_end)
        ]
        if trial_stims.size < 1:
            continue

        stim_times_per_trial.append(trial_stims)
        first_stim_times.append(float(trial_stims[0]))
        observation_end_times.append(observation_end)
        response_values.append(int(response[i]))
        wait_times.append(cp_exit - cp_entry)
        response_times.append(rp_entry - cp_exit)

    return {
        "stim_times_per_trial": stim_times_per_trial,
        "first_stim_times": np.asarray(first_stim_times, dtype=float),
        "observation_end_times": np.asarray(observation_end_times, dtype=float),
        "response_values": np.asarray(response_values, dtype=int),
        "wait_times": np.asarray(wait_times, dtype=float),
        "response_times": np.asarray(response_times, dtype=float),
        "observation_window": observation_window,
        "n_trials": len(stim_times_per_trial),
        "trial_table": pd.DataFrame(
            {
                "response": response_values,
                "wait_time": wait_times,
                "response_time": response_times,
                "first_stim": first_stim_times,
                "observation_end": observation_end_times,
            }
        ),
    }


def compute_session_psychophysical_kernel(
    align_ev: Mapping[str, np.ndarray],
    trial_df,
    *,
    timebins: int = 10,
    bin_width_s: float = 0.1,
    max_rate_hz: float = 20.0,
    cv_splits: int = 10,
    random_state: int = 0,
    min_trials_per_bin: int = 50,
    regularization_C: float = 1.0,
    stim_key: str = "stim_ev",
    observation_window: str = "center_exit",
) -> dict[str, Any]:
    """Session kernel: extract flashes → residual matrix → logistic fit."""
    inputs = extract_trial_kernel_inputs(
        align_ev,
        trial_df,
        stim_key=stim_key,
        observation_window=observation_window,
    )
    residual, choice_right, n_observed, bin_centers = build_residual_rate_matrix(
        inputs["stim_times_per_trial"],
        inputs["first_stim_times"],
        inputs["observation_end_times"],
        inputs["response_values"],
        timebins=timebins,
        bin_width_s=bin_width_s,
        max_rate_hz=max_rate_hz,
    )
    fit = fit_psychophysical_kernel(
        residual,
        choice_right,
        n_observed_per_bin=n_observed,
        cv_splits=cv_splits,
        random_state=random_state,
        min_trials_per_bin=min_trials_per_bin,
        regularization_C=regularization_C,
    )
    interpretation = interpret_kernel_weights(
        fit["weights_mean"],
        n_observed,
        min_trials_per_bin=min_trials_per_bin,
    )
    wait_times = np.asarray(inputs["wait_times"], dtype=float)
    response_times = np.asarray(inputs["response_times"], dtype=float)
    return {
        **fit,
        "bin_centers_s": bin_centers,
        "interpretation": interpretation,
        "observation_window": observation_window,
        "wait_time_mean": float(np.mean(wait_times)) if wait_times.size else np.nan,
        "wait_time_std": float(np.std(wait_times)) if wait_times.size else np.nan,
        "response_time_mean": float(np.mean(response_times))
        if response_times.size
        else np.nan,
        "response_time_std": float(np.std(response_times))
        if response_times.size
        else np.nan,
        "wait_times": wait_times,
        "response_times": response_times,
        "n_trials": int(choice_right.size),
    }
