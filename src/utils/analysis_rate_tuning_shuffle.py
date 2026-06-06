"""Task rate-tuning shuffle-null helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ephys.src.utils.analysis_rate_tuning import (
    aggregate_tuning_curves,
    summarize_units,
)


def shuffle_fsi_null(
    trial_responses: pd.DataFrame,
    observed_unit_summary: pd.DataFrame,
    n_shuffles: int = 1000,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shuffle trial rate labels and recompute population/unit FSI nulls."""
    population_null, unit_null = compute_shuffle_fsi_values(
        trial_responses,
        n_shuffles=n_shuffles,
        seed=seed,
    )
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


def compute_shuffle_fsi_values(
    trial_responses: pd.DataFrame,
    n_shuffles: int = 1000,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return population and unit FSI values from shuffled rate labels."""
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
    return population_null, unit_null
