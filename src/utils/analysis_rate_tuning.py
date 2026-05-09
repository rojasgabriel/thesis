"""Task stimulus-period rate tuning helpers.

**Naming convention**

- ``build_*`` — construct analysis tables from source events and trials.
- ``compute_*`` — deterministic array transforms.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


STIM_RATE_MIN_HZ = 4
STIM_RATE_MAX_HZ = 20
CATEGORY_BOUNDARY_HZ = 12
VISUAL_FLASH_WIDTH_S = 0.015


def first_event_in_window(
    events: np.ndarray,
    start: float,
    end: float,
    include_start: bool = True,
) -> float:
    """Return the first event in a half-open time window, else NaN."""
    event_array = np.asarray(events, dtype=float)
    finite_events = event_array[np.isfinite(event_array)]
    if include_start:
        mask = (finite_events >= start) & (finite_events < end)
    else:
        mask = (finite_events > start) & (finite_events < end)
    if not mask.any():
        return np.nan
    return float(finite_events[mask][0])


def response_events_for_choice(align_ev: Mapping[str, np.ndarray], response: float):
    """Return side-port events matching the Chipmunk response code."""
    if response == -1:
        return align_ev["left_port"]
    if response == 1:
        return align_ev["right_port"]
    return np.array([], dtype=float)


def build_task_stimulus_windows(
    align_ev: Mapping[str, np.ndarray],
    trial_df: pd.DataFrame,
    min_rate_hz: int = STIM_RATE_MIN_HZ,
    max_rate_hz: int = STIM_RATE_MAX_HZ,
) -> pd.DataFrame:
    """Build first-flash-to-response trial windows for rate tuning."""
    required_columns = {
        "trial_start_ts",
        "stim_rate_vision",
        "response",
        "with_choice",
    }
    missing_columns = required_columns.difference(trial_df.columns)
    if missing_columns:
        raise ValueError(f"trial_df is missing columns: {sorted(missing_columns)}")

    trial_starts = trial_df["trial_start_ts"].to_numpy(dtype=float)
    first_stim_events = np.asarray(align_ev["first_stim_ev_15ms"], dtype=float)

    rows = []
    for position, (trial_idx, trial) in enumerate(trial_df.iterrows()):
        stim_rate = float(trial["stim_rate_vision"])
        if int(trial["with_choice"]) != 1:
            continue
        if not min_rate_hz <= stim_rate <= max_rate_hz:
            continue

        trial_start = float(trial["trial_start_ts"])
        if not np.isfinite(trial_start):
            continue
        trial_end = (
            float(trial_starts[position + 1])
            if position + 1 < len(trial_starts)
            else np.inf
        )

        first_flash = first_event_in_window(first_stim_events, trial_start, trial_end)
        if not np.isfinite(first_flash):
            continue

        response_events = response_events_for_choice(align_ev, trial["response"])
        response_onset = first_event_in_window(
            response_events,
            first_flash,
            trial_end,
            include_start=False,
        )
        if not np.isfinite(response_onset) or response_onset <= first_flash:
            continue

        rows.append(
            {
                "trial_idx": int(trial_idx),
                "stim_rate_vision": stim_rate,
                "response_side": int(trial["response"]),
                "with_choice": int(trial["with_choice"]),
                "trial_start_s": trial_start,
                "trial_end_s": trial_end,
                "window_start_s": first_flash,
                "window_end_s": response_onset,
                "window_duration_s": response_onset - first_flash,
                "category_boundary": float(
                    trial.get("category_boundary", CATEGORY_BOUNDARY_HZ)
                ),
                "rewarded": float(trial.get("rewarded", np.nan)),
                "early_withdrawal": float(trial.get("early_withdrawal", np.nan)),
            }
        )

    return pd.DataFrame(rows)


def compute_trial_responses(
    windows_df: pd.DataFrame,
    spike_times_by_unit: Mapping[int, np.ndarray],
) -> pd.DataFrame:
    """Compute spike counts and rates for every unit and valid trial window."""
    rows = []
    for unit_id, spike_times in spike_times_by_unit.items():
        spikes = np.asarray(spike_times, dtype=float)
        for window in windows_df.itertuples(index=False):
            start = float(window.window_start_s)
            end = float(window.window_end_s)
            spike_count = int(np.count_nonzero((spikes >= start) & (spikes < end)))
            duration = float(window.window_duration_s)
            rows.append(
                {
                    "unit_id": int(unit_id),
                    "trial_idx": int(window.trial_idx),
                    "stim_rate_vision": float(window.stim_rate_vision),
                    "response_side": int(window.response_side),
                    "with_choice": int(window.with_choice),
                    "category_boundary": float(
                        getattr(window, "category_boundary", CATEGORY_BOUNDARY_HZ)
                    ),
                    "rewarded": float(getattr(window, "rewarded", np.nan)),
                    "early_withdrawal": float(
                        getattr(window, "early_withdrawal", np.nan)
                    ),
                    "window_start_s": start,
                    "window_end_s": end,
                    "window_duration_s": duration,
                    "spike_count": spike_count,
                    "response_sp_s": spike_count / duration,
                }
            )

    return pd.DataFrame(rows)


def aggregate_tuning_curves(trial_responses: pd.DataFrame) -> pd.DataFrame:
    """Aggregate unit-by-trial responses into unit-by-rate tuning curves."""
    if trial_responses.empty:
        return pd.DataFrame(
            columns=[
                "unit_id",
                "stim_rate_vision",
                "n_trials",
                "mean_sp_s",
                "sem_sp_s",
                "median_sp_s",
                "window_duration_mean_s",
            ]
        )

    tuning = (
        trial_responses.groupby(["unit_id", "stim_rate_vision"], as_index=False)
        .agg(
            n_trials=("response_sp_s", "size"),
            mean_sp_s=("response_sp_s", "mean"),
            sem_sp_s=("response_sp_s", "sem"),
            median_sp_s=("response_sp_s", "median"),
            window_duration_mean_s=("window_duration_s", "mean"),
        )
        .sort_values(["unit_id", "stim_rate_vision"])
        .reset_index(drop=True)
    )
    tuning["sem_sp_s"] = tuning["sem_sp_s"].fillna(0.0)
    return tuning


def summarize_units(tuning_curves: pd.DataFrame) -> pd.DataFrame:
    """Summarize each unit's descriptive tuning curve shape."""
    if tuning_curves.empty:
        return pd.DataFrame(
            columns=[
                "unit_id",
                "mean_sp_s_all_rates",
                "min_rate_sp_s",
                "max_rate_sp_s",
                "preferred_stim_rate",
                "tuning_range_sp_s",
                "frequency_selectivity_index",
                "normalized_tuning_range",
            ]
        )

    rows = []
    for unit_id, unit_df in tuning_curves.groupby("unit_id"):
        unit_df = unit_df.sort_values("stim_rate_vision")
        max_idx = unit_df["mean_sp_s"].idxmax()
        min_rate = float(unit_df["mean_sp_s"].min())
        max_rate = float(unit_df["mean_sp_s"].max())
        tuning_range = max_rate - min_rate
        fsi_denominator = max_rate + min_rate
        frequency_selectivity_index = (
            tuning_range / fsi_denominator if fsi_denominator > 0 else np.nan
        )
        mean_all = float(unit_df["mean_sp_s"].mean())
        rows.append(
            {
                "unit_id": int(unit_id),
                "mean_sp_s_all_rates": mean_all,
                "min_rate_sp_s": min_rate,
                "max_rate_sp_s": max_rate,
                "preferred_stim_rate": float(unit_df.loc[max_idx, "stim_rate_vision"]),
                "tuning_range_sp_s": tuning_range,
                "frequency_selectivity_index": frequency_selectivity_index,
                "normalized_tuning_range": tuning_range / (mean_all + 1e-9),
            }
        )
    return pd.DataFrame(rows).sort_values("unit_id").reset_index(drop=True)


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


def compute_light_exposure(
    windows_df: pd.DataFrame,
    stim_events: np.ndarray,
    flash_width_s: float = VISUAL_FLASH_WIDTH_S,
) -> pd.DataFrame:
    """Count visual flashes and derived light exposure in each trial window."""
    events = np.asarray(stim_events, dtype=float)
    rows = []
    for window in windows_df.itertuples(index=False):
        start = float(window.window_start_s)
        end = float(window.window_end_s)
        flash_count = int(np.count_nonzero((events >= start) & (events < end)))
        duration = float(window.window_duration_s)
        total_light_time = flash_count * flash_width_s
        rows.append(
            {
                "trial_idx": int(window.trial_idx),
                "stim_rate_vision": float(getattr(window, "stim_rate_vision", np.nan)),
                "window_start_s": start,
                "window_end_s": end,
                "window_duration_s": duration,
                "flash_count": flash_count,
                "total_light_time_s": total_light_time,
                "duty_cycle": total_light_time / duration if duration > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def add_light_exposure_to_responses(
    trial_responses: pd.DataFrame,
    light_exposure: pd.DataFrame,
) -> pd.DataFrame:
    """Merge trial light exposure into unit responses and compute spikes/flash."""
    exposure_columns = [
        "trial_idx",
        "flash_count",
        "total_light_time_s",
        "duty_cycle",
    ]
    out = trial_responses.merge(
        light_exposure[exposure_columns], on="trial_idx", how="left"
    )
    out["spikes_per_flash"] = np.where(
        out["flash_count"] > 0,
        out["spike_count"] / out["flash_count"],
        np.nan,
    )
    return out


def shuffle_fsi_null(
    trial_responses: pd.DataFrame,
    observed_unit_summary: pd.DataFrame,
    n_shuffles: int = 1000,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shuffle trial rate labels and recompute population/unit FSI nulls."""
    rng = np.random.default_rng(seed)
    base = trial_responses.copy()
    trial_rates = (
        base[["trial_idx", "stim_rate_vision"]]
        .drop_duplicates("trial_idx")
        .sort_values("trial_idx")
        .reset_index(drop=True)
    )
    rate_values = trial_rates["stim_rate_vision"].to_numpy(dtype=float)
    trial_ids = trial_rates["trial_idx"].to_numpy()

    population_rows = []
    unit_rows = []
    for shuffle_idx in range(n_shuffles):
        shuffled_rates = rng.permutation(rate_values)
        rate_map = dict(zip(trial_ids, shuffled_rates, strict=True))
        shuffled = base.copy()
        shuffled["stim_rate_vision"] = shuffled["trial_idx"].map(rate_map)
        shuffled_summary = summarize_units(aggregate_tuning_curves(shuffled))
        fsi = shuffled_summary["frequency_selectivity_index"].dropna()
        population_rows.append(
            {
                "shuffle_idx": shuffle_idx,
                "median_fsi": float(fsi.median()),
                "mean_fsi": float(fsi.mean()),
            }
        )
        for row in shuffled_summary.itertuples(index=False):
            unit_rows.append(
                {
                    "shuffle_idx": shuffle_idx,
                    "unit_id": int(row.unit_id),
                    "frequency_selectivity_index": float(
                        row.frequency_selectivity_index
                    ),
                }
            )

    population_null = pd.DataFrame(population_rows)
    unit_null = pd.DataFrame(unit_rows)
    unit_p95 = (
        unit_null.groupby("unit_id")["frequency_selectivity_index"]
        .quantile(0.95)
        .rename("shuffle_fsi_p95")
        .reset_index()
    )
    unit_summary = observed_unit_summary[
        ["unit_id", "frequency_selectivity_index"]
    ].rename(columns={"frequency_selectivity_index": "observed_fsi"})
    unit_summary = unit_summary.merge(unit_p95, on="unit_id", how="left")
    unit_summary["exceeds_shuffle_p95"] = (
        unit_summary["observed_fsi"] > unit_summary["shuffle_fsi_p95"]
    )
    return population_null, unit_summary


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


def residualize_by_unit(
    trial_responses: pd.DataFrame,
    response_column: str,
    predictor_column: str,
) -> pd.DataFrame:
    """Regress one predictor out of each unit's response."""
    rows = []
    for unit_id, unit_df in trial_responses.groupby("unit_id"):
        y = unit_df[response_column].to_numpy(dtype=float)
        predictor = unit_df[predictor_column].to_numpy(dtype=float)
        valid = np.isfinite(y) & np.isfinite(predictor)
        residual = np.full(y.shape, np.nan)
        if valid.sum() >= 2 and np.nanstd(predictor[valid]) > 0:
            x = np.column_stack([np.ones(valid.sum()), predictor[valid]])
            beta = np.linalg.pinv(x) @ y[valid]
            residual[valid] = y[valid] - x @ beta
        out = unit_df.copy()
        out["residual_response_sp_s"] = residual
        rows.append(out)
    return pd.concat(rows, ignore_index=True)


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
