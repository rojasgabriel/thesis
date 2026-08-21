"""Population PETH: spike times + alignment → binned rates (sp/s).

**Naming convention**

- ``compute_*`` — deterministic array transforms / thin wrappers around external libs.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from spks.event_aligned import population_peth


def compute_population_peth(
    spike_times_per_unit: Sequence[np.ndarray],
    alignment_times: np.ndarray,
    pre_seconds: float = 0.1,
    post_seconds: float = 0.15,
    binwidth_ms: int = 10,
    t_rise: Optional[float] = None,
    t_decay: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute population PETH aligned to event times.

    This is a thin wrapper around `spks.event_aligned.population_peth`.

    `spks.population_peth` has shipped in two common conventions:
    - returning **counts per bin**, which must be converted to **sp/s**
      by dividing by bin width
    - returning **rates (sp/s)** already

    This wrapper auto-detects which convention is in use and ensures the
    returned `peth` is always in **sp/s**. DO NOT re-divide downstream.
    A runtime assertion catches accidental re-scaling (values > 1000 sp/s
    are implausible for V1).

    WARNING on kernel behaviour: the alpha-function kernel built with
    `t_decay=0.025s` (a common default elsewhere) merges temporal features
    30-45 ms apart. For double-peak detection or any fine temporal analysis,
    pass `t_rise=None, t_decay=None` to disable smoothing entirely.

    Args:
        spike_times_per_unit: list/sequence of arrays, one per unit (seconds).
        alignment_times: array of event times to align to (seconds).
        pre_seconds: time before event.
        post_seconds: time after event.
        binwidth_ms: bin width in milliseconds.
        t_rise: alpha-function rise time (seconds). Pass None to disable.
        t_decay: alpha-function decay time (seconds). Pass None to disable.
            Both t_rise and t_decay must be None OR both non-None; passing
            only one is treated as disabled.

    Returns:
        peth: array (n_units, n_trials, n_timebins) in **sp/s**
        bin_edges: array (n_timebins + 1,) in seconds relative to event
        bin_centers: array (n_timebins,) in seconds relative to event
    """
    kernel = None
    if t_rise is not None and t_decay is not None:
        from spks.utils import alpha_function

        decay_bins = t_decay / (binwidth_ms / 1000)
        kernel = alpha_function(
            int(decay_bins * 15),
            t_rise=t_rise,
            t_decay=decay_bins,
            srate=1.0 / (binwidth_ms / 1000),
        )

    peth_counts, bin_edges, _event_index = population_peth(
        all_spike_times=spike_times_per_unit,
        alignment_times=alignment_times,
        pre_seconds=pre_seconds,
        post_seconds=post_seconds,
        binwidth_ms=binwidth_ms,
        kernel=kernel,
    )

    dt = binwidth_ms / 1000.0
    peth_if_counts = peth_counts / dt

    # Heuristic: some spks versions return sp/s already (with kernel=None,
    # values are typically quantized in steps of ~1/dt, e.g. 100 sp/s for
    # 10 ms bins). Detect that case to avoid double-scaling.
    pos = peth_counts[peth_counts > 0]
    min_pos = float(pos.min()) if pos.size else 0.0
    looks_like_rates = min_pos >= 0.9 * (1.0 / dt)
    peth = peth_counts if looks_like_rates else peth_if_counts

    # Sanity guard: values > 1000 sp/s are implausible for V1 and usually
    # indicate a bad input scale or an extra downstream rescaling.
    if peth.size > 0 and peth.max() > 1000:
        raise AssertionError(
            f"peth.max()={peth.max():.1f} sp/s — implausibly high. "
            "Did you accidentally pass counts instead of spike times, or "
            "re-scale the PETH downstream?"
        )
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    return peth, bin_edges, bin_centers
