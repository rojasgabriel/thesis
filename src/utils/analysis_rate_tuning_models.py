"""Task rate-tuning evidence, category, and choice model helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ephys.src.utils.analysis_rate_tuning import CATEGORY_BOUNDARY_HZ


def add_trial_predictors(table: pd.DataFrame) -> pd.DataFrame:
    """Add task predictors used by rate-tuning follow-up analyses."""
    out = table.copy()
    boundary = out.get("category_boundary", CATEGORY_BOUNDARY_HZ)
    out["signed_evidence"] = out["stim_rate_vision"].astype(float) - boundary
    out["stim_category"] = np.select(
        [out["signed_evidence"] < 0, out["signed_evidence"] > 0],
        ["low_rate", "high_rate"],
        default="boundary",
    )
    if "rewarded" in out.columns and out["rewarded"].notna().any():
        out["correct"] = out["rewarded"].astype(float)
    else:
        out["correct"] = np.select(
            [
                (out["signed_evidence"] < 0) & (out["response_side"] == -1),
                (out["signed_evidence"] > 0) & (out["response_side"] == 1),
            ],
            [1.0, 1.0],
            default=np.nan,
        )
    return out


def _cv_r2(y: np.ndarray, x: np.ndarray, n_splits: int = 5) -> float:
    valid = np.isfinite(y) & np.isfinite(x).all(axis=1)
    y = y[valid]
    x = x[valid]
    if y.size < 4 or np.nanstd(y) == 0:
        return np.nan

    n_splits = min(n_splits, y.size)
    fold_ids = np.arange(y.size) % n_splits
    ss_res = 0.0
    ss_tot = 0.0
    for fold in range(n_splits):
        train = fold_ids != fold
        test = ~train
        if train.sum() == 0 or test.sum() == 0:
            continue
        beta = np.linalg.pinv(x[train]) @ y[train]
        prediction = x[test] @ beta
        ss_res += float(np.sum((y[test] - prediction) ** 2))
        ss_tot += float(np.sum((y[test] - np.mean(y[train])) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def fit_encoding_models(
    trial_responses: pd.DataFrame,
    n_splits: int = 5,
) -> pd.DataFrame:
    """Fit simple cross-validated per-unit task-predictor models."""
    responses = add_trial_predictors(trial_responses)
    rows = []
    for unit_id, unit_df in responses.groupby("unit_id"):
        y = unit_df["response_sp_s"].to_numpy(dtype=float)
        signed = unit_df["signed_evidence"].to_numpy(dtype=float)
        category_high = (unit_df["stim_category"] == "high_rate").to_numpy(dtype=float)
        category_boundary = (unit_df["stim_category"] == "boundary").to_numpy(
            dtype=float
        )
        choice = unit_df["response_side"].to_numpy(dtype=float)
        intercept = np.ones_like(y)
        model_matrices = {
            "baseline": np.column_stack([intercept]),
            "signed_evidence": np.column_stack([intercept, signed]),
            "category": np.column_stack([intercept, category_high, category_boundary]),
            "choice": np.column_stack([intercept, choice]),
            "combined": np.column_stack(
                [intercept, signed, category_high, category_boundary, choice]
            ),
        }
        scores = {
            name: _cv_r2(y, matrix, n_splits=n_splits)
            for name, matrix in model_matrices.items()
        }
        baseline_score = scores["baseline"]
        for model_name, score in scores.items():
            rows.append(
                {
                    "unit_id": int(unit_id),
                    "model": model_name,
                    "cv_r2": score,
                    "delta_cv_r2": score - baseline_score,
                }
            )
    return pd.DataFrame(rows)


def _shuffle_columns(
    matrix: np.ndarray,
    columns: list[int],
    rng: np.random.Generator,
) -> np.ndarray:
    shuffled = matrix.copy()
    order = rng.permutation(shuffled.shape[0])
    shuffled[:, columns] = shuffled[order][:, columns]
    return shuffled


def fit_choice_light_encoding_models(
    trial_responses: pd.DataFrame,
    n_splits: int = 5,
    n_shuffles: int = 5,
    seed: int = 0,
) -> pd.DataFrame:
    """Estimate unique variance from full-model regressor shuffles.

    Each unit is fit with a full design matrix containing stimulus, choice, and
    light-time predictors. Unique delta CV R2 is the full-model CV R2 minus the
    CV R2 after shuffling one predictor block across trials.
    """
    responses = add_trial_predictors(trial_responses)
    rows = []
    for unit_id, unit_df in responses.groupby("unit_id"):
        y = unit_df["response_sp_s"].to_numpy(dtype=float)
        signed = unit_df["signed_evidence"].to_numpy(dtype=float)
        category_high = (unit_df["stim_category"] == "high_rate").to_numpy(dtype=float)
        category_boundary = (unit_df["stim_category"] == "boundary").to_numpy(
            dtype=float
        )
        choice = unit_df["response_side"].to_numpy(dtype=float)
        light_time = unit_df.get(
            "total_light_time_s",
            pd.Series(np.zeros(len(unit_df)), index=unit_df.index),
        ).to_numpy(dtype=float)
        intercept = np.ones_like(y)
        full_matrix = np.column_stack(
            [
                intercept,
                signed,
                category_high,
                category_boundary,
                choice,
                light_time,
            ]
        )
        full_score = _cv_r2(y, full_matrix, n_splits=n_splits)
        block_columns = {
            "stimulus": [1, 2, 3],
            "choice": [4],
            "light": [5],
        }
        rng = np.random.default_rng(seed + int(unit_id))
        for regressor, columns in block_columns.items():
            shuffle_scores = [
                _cv_r2(
                    y,
                    _shuffle_columns(full_matrix, columns, rng),
                    n_splits=n_splits,
                )
                for _ in range(n_shuffles)
            ]
            shuffle_scores = np.asarray(shuffle_scores, dtype=float)
            rows.append(
                {
                    "unit_id": int(unit_id),
                    "regressor": regressor,
                    "full_cv_r2": full_score,
                    "shuffled_cv_r2_median": float(np.nanmedian(shuffle_scores)),
                    "shuffled_cv_r2_p025": float(np.nanpercentile(shuffle_scores, 2.5)),
                    "shuffled_cv_r2_p975": float(
                        np.nanpercentile(shuffle_scores, 97.5)
                    ),
                    "unique_delta_cv_r2": full_score
                    - float(np.nanmedian(shuffle_scores)),
                    "n_shuffles": int(n_shuffles),
                }
            )
    return pd.DataFrame(rows)
