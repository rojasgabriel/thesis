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

    if log_scale:
        log_values = np.log(values)
        mean_log = float(np.mean(log_values))
        dof = values.size - 1
        t_crit = float(stats.t.ppf((1.0 + ci_level) / 2.0, dof))
        sem_log = float(np.std(log_values, ddof=1)) / np.sqrt(values.size)
        lower = float(np.exp(mean_log - t_crit * sem_log))
        upper = float(np.exp(mean_log + t_crit * sem_log))
        mean_value = float(np.exp(mean_log))
        return mean_value, lower, upper

    mean_value = float(np.mean(values))
    dof = values.size - 1
    t_crit = float(stats.t.ppf((1.0 + ci_level) / 2.0, dof))
    sem_value = float(np.std(values, ddof=1)) / np.sqrt(values.size)
    lower = mean_value - t_crit * sem_value
    upper = mean_value + t_crit * sem_value
    return mean_value, lower, upper
