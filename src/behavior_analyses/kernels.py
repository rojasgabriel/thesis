from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


def build_residual_rate_matrix(
    stim_events,
    response_values,
    *,
    timebins: int = 10,
    max_rate_hz: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    choices = []
    for events, response in zip(stim_events, response_values):
        if response not in (-1, 1):
            continue
        events = np.asarray(events, dtype=float)
        events = events[np.isfinite(events)]
        if events.size < 2:
            continue
        bins = np.linspace(events[0], events[-1], num=timebins + 1)
        specific_rate = max_rate_hz / len(bins)
        instantaneous_rate, _ = np.histogram(events, bins=bins)
        rows.append(instantaneous_rate - specific_rate)
        choices.append(response == 1)
    if not rows:
        return np.empty((0, timebins)), np.empty((0,), dtype=int)
    return np.asarray(rows, dtype=float), np.asarray(choices, dtype=int)


def fit_psychophysical_kernel(
    stim_events,
    response_values,
    *,
    timebins: int = 10,
    cv_splits: int = 10,
    random_state: int = 0,
    max_rate_hz: float = 20.0,
) -> dict:
    x, y = build_residual_rate_matrix(
        stim_events, response_values, timebins=timebins, max_rate_hz=max_rate_hz
    )
    if x.shape[0] < cv_splits or np.unique(y).size < 2:
        return {
            "design_matrix": x,
            "choice_right": y,
            "weights": np.empty((0, timebins)),
            "scores": np.empty((0,)),
            "bias": np.empty((0,)),
            "error": np.empty((0, timebins)),
        }

    splitter = StratifiedKFold(
        n_splits=cv_splits, shuffle=True, random_state=random_state
    )
    weights = []
    scores = []
    biases = []
    errors = []
    for train_index, test_index in splitter.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model = LogisticRegression(
            penalty="l2", solver="liblinear", C=1, fit_intercept=True
        ).fit(x_train, y_train)
        predict_prob = model.predict_proba(x_train)
        variance = np.prod(predict_prob, axis=1)
        covariance = np.linalg.pinv(np.dot(x_train.T * variance, x_train))
        errors.append(np.sqrt(np.diag(covariance)))
        weights.append(model.coef_[0])
        scores.append(model.score(x_test, y_test))
        biases.append(model.intercept_[0])

    return {
        "design_matrix": x,
        "choice_right": y,
        "weights": np.asarray(weights, dtype=float),
        "scores": np.asarray(scores, dtype=float),
        "bias": np.asarray(biases, dtype=float),
        "error": np.asarray(errors, dtype=float),
    }

