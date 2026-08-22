"""Per-unit baseline vs response selectivity from aligned PETHs.

**Naming convention**

- ``compute_*`` — statistical summaries on array inputs (no labdata I/O).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests


def compute_unit_selectivity(
    peth: np.ndarray,
    bin_edges: np.ndarray,
    unit_ids: Sequence,
    base_window: tuple[float, float] = (-0.1, 0.0),
    resp_window: tuple[float, float] = (0.04, 0.10),
    test: str = "wilcoxon",
    correction: str = "fdr_bh",
    alpha: float = 0.05,
    min_delta_abs: Optional[float] = None,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Compute baseline vs response selectivity per unit.

    The test compares **per-trial mean rates** in the response window to
    per-trial mean rates in the baseline window (paired). This is NOT a
    per-bin test — time structure within the window is collapsed.

    Statistical significance is NOT the same as biological magnitude. With
    enough trials, a mean difference of 1 sp/s can pass at q<0.05. If you
    need a magnitude floor, pass `min_delta_abs` — units with
    |delta| < min_delta_abs will NOT be marked excited/suppressed even if
    the Wilcoxon is significant.

    Args:
        peth: array (n_units, n_trials, n_timepoints) of firing rates (sp/s)
        bin_edges: array (n_timepoints + 1,) of bin edges in seconds relative to event
        unit_ids: sequence of unit IDs matching peth's first axis
        base_window: (start, end) seconds for baseline
        resp_window: (start, end) seconds for response
        test: 'wilcoxon' (default) or 'ttest'
        correction: multiple-comparisons method passed to
            statsmodels.stats.multitest.multipletests, e.g. 'fdr_bh'
            (Benjamini-Hochberg, default) or 'bonferroni'.
        alpha: significance threshold applied to corrected p-values
        min_delta_abs: optional minimum |mean response - mean baseline|
            threshold (sp/s). Units below this are excluded from the
            excited/suppressed masks even if statistically significant.
            Default None = no magnitude filter.

    Returns:
        results_df: DataFrame with per-unit stats
        masks: dict with boolean masks (excited, suppressed, selective)
    """
    n_units, n_trials, _n_time = peth.shape
    if len(unit_ids) != n_units:
        raise ValueError(f"len(unit_ids)={len(unit_ids)} != peth n_units={n_units}")
    t_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    base_mask = (t_centers >= base_window[0]) & (t_centers < base_window[1])
    resp_mask = (t_centers >= resp_window[0]) & (t_centers < resp_window[1])

    if not base_mask.any() or not resp_mask.any():
        raise ValueError("Baseline/response windows do not overlap available bins.")

    base_rates = peth[:, :, base_mask].mean(axis=2)
    resp_rates = peth[:, :, resp_mask].mean(axis=2)

    pvals = np.ones(n_units, dtype=float)
    deltas = np.zeros(n_units, dtype=float)
    d_cohen = np.zeros(n_units, dtype=float)
    mean_base = base_rates.mean(axis=1)
    mean_resp = resp_rates.mean(axis=1)
    si = (mean_resp - mean_base) / (mean_resp + mean_base + 1e-9)

    for u in range(n_units):
        x = resp_rates[u]
        y = base_rates[u]
        diff = x - y

        deltas[u] = diff.mean()
        sd = diff.std(ddof=1)
        d_cohen[u] = deltas[u] / sd if sd > 0 else 0.0

        if np.allclose(diff, 0):
            pvals[u] = 1.0
            continue

        if test == "ttest":
            _, pvals[u] = stats.ttest_rel(
                x, y, alternative="two-sided", nan_policy="omit"
            )
        elif test == "wilcoxon":
            try:
                _, pvals[u] = stats.wilcoxon(
                    x, y, zero_method="wilcox", alternative="two-sided"
                )
            except ValueError:
                pvals[u] = 1.0
        else:
            raise ValueError("test must be 'wilcoxon' or 'ttest'")

    _, qvals, _, _ = multipletests(pvals, alpha=alpha, method=correction)

    excited = (qvals < alpha) & (deltas > 0)
    suppressed = (qvals < alpha) & (deltas < 0)
    if min_delta_abs is not None:
        magnitude_mask = np.abs(deltas) >= min_delta_abs
        excited &= magnitude_mask
        suppressed &= magnitude_mask
    selective_any = excited | suppressed

    results_df = pd.DataFrame(
        {
            "unit": list(unit_ids),
            "mean_base": mean_base,
            "mean_resp": mean_resp,
            "delta": deltas,
            "cohen_d": d_cohen,
            "si": si,
            "p": pvals,
            "q": qvals,
            "excited": excited,
            "suppressed": suppressed,
            "selective": selective_any,
        }
    )

    return results_df, {
        "excited": excited,
        "suppressed": suppressed,
        "selective": selective_any,
    }
