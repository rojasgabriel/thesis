"""Choice-balancing helpers for task rate-tuning controls."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ephys.src.utils.analysis_rate_tuning import (
    aggregate_tuning_curves,
    summarize_units,
)


def balanced_choice_trial_ids(
    trial_table: pd.DataFrame,
    *,
    class_column: str,
    choice_column: str = "response_side",
    seed: int = 0,
) -> np.ndarray:
    """Return trial ids after equalizing left/right choices within each class."""
    rng = np.random.default_rng(seed)
    sampled_trial_ids = []
    for _class_value, class_df in trial_table.groupby(class_column, sort=True):
        choice_groups = {
            int(choice): group["trial_idx"].drop_duplicates().to_numpy(dtype=int)
            for choice, group in class_df.groupby(choice_column)
            if choice in (-1, 1)
        }
        if -1 not in choice_groups or 1 not in choice_groups:
            continue
        n_per_choice = min(len(choice_groups[-1]), len(choice_groups[1]))
        if n_per_choice == 0:
            continue
        for choice in (-1, 1):
            sampled = rng.choice(
                choice_groups[choice], size=n_per_choice, replace=False
            )
            sampled_trial_ids.extend(sampled.tolist())
    return np.asarray(sorted(sampled_trial_ids), dtype=int)


def balanced_choice_trial_responses(
    trial_responses: pd.DataFrame,
    *,
    class_column: str,
    seed: int = 0,
) -> pd.DataFrame:
    """Downsample trial responses to equal left/right choices within each class."""
    trial_table = trial_responses[
        ["trial_idx", class_column, "response_side"]
    ].drop_duplicates("trial_idx")
    trial_ids = balanced_choice_trial_ids(
        trial_table,
        class_column=class_column,
        seed=seed,
    )
    return trial_responses[trial_responses["trial_idx"].isin(trial_ids)].copy()


def repeated_choice_balanced_fsi(
    trial_responses: pd.DataFrame,
    *,
    class_column: str = "stim_rate_vision",
    n_resamples: int = 200,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recompute unit FSI after repeated within-rate choice balancing."""
    population_rows = []
    unit_rows = []
    for resample_idx in range(n_resamples):
        balanced = balanced_choice_trial_responses(
            trial_responses,
            class_column=class_column,
            seed=seed + resample_idx,
        )
        if balanced.empty:
            continue
        unit_summary = summarize_units(aggregate_tuning_curves(balanced))
        if unit_summary.empty:
            continue
        fsi = unit_summary["frequency_selectivity_index"].dropna()
        population_rows.append(
            {
                "resample_idx": resample_idx,
                "n_trials": int(balanced["trial_idx"].nunique()),
                "n_units": int(unit_summary["unit_id"].nunique()),
                "median_fsi": float(fsi.median()),
                "mean_fsi": float(fsi.mean()),
            }
        )
        unit_summary = unit_summary[["unit_id", "frequency_selectivity_index"]].copy()
        unit_summary.insert(0, "resample_idx", resample_idx)
        unit_rows.append(unit_summary)
    return pd.DataFrame(population_rows), pd.concat(unit_rows, ignore_index=True)


def summarize_resampled_units(unit_resamples: pd.DataFrame) -> pd.DataFrame:
    """Summarize repeated choice-balanced FSI per unit."""
    return (
        unit_resamples.groupby("unit_id", as_index=False)
        .agg(
            balanced_fsi_median=("frequency_selectivity_index", "median"),
            balanced_fsi_p025=(
                "frequency_selectivity_index",
                lambda values: float(np.nanpercentile(values, 2.5)),
            ),
            balanced_fsi_p975=(
                "frequency_selectivity_index",
                lambda values: float(np.nanpercentile(values, 97.5)),
            ),
            n_resamples=("frequency_selectivity_index", "size"),
        )
        .reset_index(drop=True)
    )
