"""Choice-balanced logistic-regression decoding for task rate tuning."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ephys.src.utils.analysis_rate_tuning_choice import balanced_choice_trial_ids


def trial_response_matrix(
    trial_responses: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return trial-by-unit response matrix and one-row-per-trial metadata."""
    matrix = trial_responses.pivot_table(
        index="trial_idx",
        columns="unit_id",
        values="response_sp_s",
    )
    metadata_columns = [
        "trial_idx",
        "stim_rate_vision",
        "stim_category",
        "response_side",
    ]
    if "category_boundary" in trial_responses.columns:
        metadata_columns.append("category_boundary")
    metadata = (
        trial_responses[metadata_columns]
        .drop_duplicates("trial_idx")
        .set_index("trial_idx")
        .loc[matrix.index]
        .reset_index()
    )
    return matrix, metadata


def _fit_predict_logistic(x: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    labels = normalize_labels(y)
    class_counts = np.bincount(pd.factorize(labels)[0])
    n_splits = min(5, int(class_counts.min()))
    if n_splits < 2:
        return np.full(labels.shape, np.nan)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    predictions = np.empty(labels.shape, dtype=object)
    for train_idx, test_idx in cv.split(x, labels):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                solver="lbfgs",
                C=1.0,
                max_iter=1000,
            ),
        )
        model.fit(x[train_idx], labels[train_idx])
        predictions[test_idx] = model.predict(x[test_idx])
    return predictions


def normalize_labels(labels: np.ndarray) -> np.ndarray:
    """Return labels with numeric object arrays converted back to numeric dtype."""
    return pd.Series(labels).infer_objects(copy=False).to_numpy()


def _shuffle_labels_within_choice(
    metadata: pd.DataFrame,
    labels: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(labels, dtype=object).copy()
    choices = metadata["response_side"].to_numpy()
    for choice in np.unique(choices):
        mask = choices == choice
        shuffled[mask] = rng.permutation(shuffled[mask])
    return shuffled


def decode_target(
    trial_responses: pd.DataFrame,
    *,
    target: str,
    n_resamples: int = 200,
    seed: int = 0,
) -> pd.DataFrame:
    """Decode stimulus category or exact rate with choice-balanced trials."""
    matrix, metadata = trial_response_matrix(trial_responses)
    if target == "category":
        keep = metadata["stim_category"].isin(["low_rate", "high_rate"])
        label_column = "stim_category"
    elif target == "rate":
        keep = np.isfinite(metadata["stim_rate_vision"])
        label_column = "stim_rate_vision"
    else:
        raise ValueError(f"Unknown decoding target: {target}")
    matrix = matrix.loc[keep.to_numpy()]
    metadata = metadata.loc[keep.to_numpy()].reset_index(drop=True)
    matrix.index = metadata["trial_idx"]

    rows = []
    for resample_idx in range(n_resamples):
        trial_ids = balanced_choice_trial_ids(
            metadata[["trial_idx", label_column, "response_side"]],
            class_column=label_column,
            seed=seed + resample_idx,
        )
        if trial_ids.size == 0:
            continue
        sample_metadata = metadata[metadata["trial_idx"].isin(trial_ids)].copy()
        class_counts = sample_metadata[label_column].value_counts()
        valid_classes = class_counts[class_counts >= 4].index
        sample_metadata = sample_metadata[
            sample_metadata[label_column].isin(valid_classes)
        ].copy()
        if sample_metadata[label_column].nunique() < 2:
            continue
        x = matrix.loc[sample_metadata["trial_idx"]].to_numpy(dtype=float)
        y = sample_metadata[label_column].to_numpy()
        predictions = _fit_predict_logistic(x, y, seed + resample_idx)
        valid_predictions = pd.notna(predictions)
        if not np.any(valid_predictions):
            continue
        y_valid = normalize_labels(y[valid_predictions])
        pred_valid = normalize_labels(predictions[valid_predictions])
        row = {
            "target": target,
            "resample_idx": resample_idx,
            "condition": "observed",
            "n_trials": int(len(y_valid)),
            "n_classes": int(np.unique(y_valid).size),
            "balanced_accuracy": float(balanced_accuracy_score(y_valid, pred_valid)),
            "rate_correlation": np.nan,
        }
        if target == "rate" and np.unique(pred_valid).size > 1:
            row["rate_correlation"] = float(
                pearsonr(y_valid.astype(float), pred_valid.astype(float)).statistic
            )
        rows.append(row)

        shuffled_y = _shuffle_labels_within_choice(
            sample_metadata,
            y,
            seed=seed + 10_000 + resample_idx,
        )
        shuffled_predictions = _fit_predict_logistic(
            x,
            shuffled_y,
            seed + 20_000 + resample_idx,
        )
        valid_shuffle = pd.notna(shuffled_predictions)
        if np.any(valid_shuffle):
            shuffled_valid = normalize_labels(shuffled_y[valid_shuffle])
            shuffled_pred_valid = normalize_labels(shuffled_predictions[valid_shuffle])
            shuffle_row = {
                "target": target,
                "resample_idx": resample_idx,
                "condition": "choice_side_shuffle",
                "n_trials": int(len(shuffled_valid)),
                "n_classes": int(np.unique(shuffled_valid).size),
                "balanced_accuracy": float(
                    balanced_accuracy_score(shuffled_valid, shuffled_pred_valid)
                ),
                "rate_correlation": np.nan,
            }
            if target == "rate" and np.unique(shuffled_pred_valid).size > 1:
                shuffle_row["rate_correlation"] = float(
                    pearsonr(
                        shuffled_valid.astype(float),
                        shuffled_pred_valid.astype(float),
                    ).statistic
                )
            rows.append(shuffle_row)
    return pd.DataFrame(rows)
