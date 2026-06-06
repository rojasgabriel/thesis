"""Task rate-tuning timecourse helpers."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from ephys.src.utils.analysis_rate_tuning_models import add_trial_predictors


def compute_timecourse_responses(
    windows_df: pd.DataFrame,
    spike_times_by_unit: Mapping[int, np.ndarray],
    bin_edges_s: np.ndarray,
) -> pd.DataFrame:
    """Count spikes in fixed bins after first flash, masking after response."""
    rows = []
    for unit_id, spike_times in spike_times_by_unit.items():
        spikes = np.asarray(spike_times, dtype=float)
        for window in windows_df.itertuples(index=False):
            window_duration = float(window.window_duration_s)
            for bin_start, bin_end in zip(
                bin_edges_s[:-1], bin_edges_s[1:], strict=True
            ):
                if bin_end > window_duration:
                    continue
                start = float(window.window_start_s) + float(bin_start)
                end = float(window.window_start_s) + float(bin_end)
                spike_count = int(np.count_nonzero((spikes >= start) & (spikes < end)))
                rows.append(
                    {
                        "unit_id": int(unit_id),
                        "trial_idx": int(window.trial_idx),
                        "stim_rate_vision": float(window.stim_rate_vision),
                        "response_side": int(window.response_side),
                        "category_boundary": float(window.category_boundary),
                        "bin_start_s": float(bin_start),
                        "bin_end_s": float(bin_end),
                        "spike_count": spike_count,
                        "response_sp_s": spike_count / (bin_end - bin_start),
                    }
                )
    return add_trial_predictors(pd.DataFrame(rows))


def _nearest_centroid_accuracy(
    matrix: pd.DataFrame,
    labels: pd.Series,
    n_splits: int = 5,
) -> float:
    labels = labels.astype(int)
    valid_rows = matrix.notna().all(axis=1) & labels.notna()
    x = matrix.loc[valid_rows].to_numpy(dtype=float)
    y = labels.loc[valid_rows].to_numpy(dtype=int)
    if x.shape[0] < 4 or np.unique(y).size < 2:
        return np.nan

    n_splits = min(n_splits, x.shape[0])
    fold_ids = np.arange(x.shape[0]) % n_splits
    predictions = []
    truth = []
    for fold in range(n_splits):
        train = fold_ids != fold
        test = ~train
        if np.unique(y[train]).size < 2:
            continue
        mean = x[train].mean(axis=0)
        std = x[train].std(axis=0)
        std[std == 0] = 1.0
        z_train = (x[train] - mean) / std
        z_test = (x[test] - mean) / std
        centroids = {
            label: z_train[y[train] == label].mean(axis=0)
            for label in np.unique(y[train])
        }
        for sample, label in zip(z_test, y[test], strict=True):
            predicted = min(
                centroids,
                key=lambda candidate: np.linalg.norm(sample - centroids[candidate]),
            )
            predictions.append(predicted)
            truth.append(label)
    return float(np.mean(np.asarray(predictions) == np.asarray(truth)))


def summarize_timecourse_encoding(
    timecourse_responses: pd.DataFrame,
    n_splits: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize rate correlation, category decoding, and unit coefficients."""
    summary_rows = []
    coefficient_rows = []
    for (bin_start, bin_end), bin_df in timecourse_responses.groupby(
        ["bin_start_s", "bin_end_s"]
    ):
        unit_z = bin_df.groupby("unit_id")["response_sp_s"].transform(
            lambda values: (values - values.mean()) / values.std(ddof=1)
        )
        valid = np.isfinite(unit_z) & np.isfinite(bin_df["signed_evidence"])
        rate_correlation = (
            float(
                np.corrcoef(bin_df.loc[valid, "signed_evidence"], unit_z[valid])[0, 1]
            )
            if valid.sum() >= 2
            else np.nan
        )
        trial_matrix = bin_df.pivot_table(
            index="trial_idx",
            columns="unit_id",
            values="response_sp_s",
        )
        trial_labels = (
            bin_df[["trial_idx", "stim_category"]]
            .drop_duplicates("trial_idx")
            .set_index("trial_idx")
            .loc[trial_matrix.index, "stim_category"]
        )
        binary_labels = trial_labels.map({"low_rate": 0, "high_rate": 1})
        category_accuracy = _nearest_centroid_accuracy(
            trial_matrix.loc[binary_labels.notna()],
            binary_labels.dropna(),
            n_splits=n_splits,
        )
        summary_rows.append(
            {
                "bin_start_s": float(bin_start),
                "bin_end_s": float(bin_end),
                "rate_correlation": rate_correlation,
                "category_accuracy": category_accuracy,
                "n_trials": int(trial_matrix.shape[0]),
            }
        )

        for unit_id, unit_df in bin_df.groupby("unit_id"):
            x = unit_df["signed_evidence"].to_numpy(dtype=float)
            y = unit_df["response_sp_s"].to_numpy(dtype=float)
            valid_unit = np.isfinite(x) & np.isfinite(y)
            coefficient = (
                float(np.polyfit(x[valid_unit], y[valid_unit], deg=1)[0])
                if valid_unit.sum() >= 2 and np.nanstd(x[valid_unit]) > 0
                else np.nan
            )
            coefficient_rows.append(
                {
                    "unit_id": int(unit_id),
                    "bin_start_s": float(bin_start),
                    "bin_end_s": float(bin_end),
                    "signed_rate_coefficient": coefficient,
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(coefficient_rows)
