"""Small frequentist summaries (means + t-based CIs).

**Naming convention**

- ``mean_*`` — scalar summaries over a 1D sample.
"""

from __future__ import annotations

import numpy as np
import scipy.stats as stats


def mean_and_t_ci(
    values: np.ndarray,
    *,
    log_scale: bool,
    ci_level: float,
    drop_nonfinite: bool,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    if drop_nonfinite:
        values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("mean_and_t_ci requires at least one value.")
    if values.size == 1:
        mean_value = float(values[0])
        return mean_value, mean_value, mean_value

    scale_values = np.log(values) if log_scale else values
    mean_value = float(np.mean(scale_values))
    lower, upper = stats.t.interval(
        ci_level,
        df=values.size - 1,
        loc=mean_value,
        scale=stats.sem(scale_values),
    )
    if log_scale:
        mean_value, lower, upper = np.exp([mean_value, lower, upper])
    return float(mean_value), float(lower), float(upper)
