"""Canonical parameters for double-peak V1 unit classification.

Last reviewed: 2026-04-29.

Pipeline:
  1. PETH (sp/s) with kernel disabled — peaks 30–45 ms apart need fine
     temporal resolution; the default spks kernel (t_decay=0.025s) merges them.
  2. Selectivity (excited): FDR-corrected Wilcoxon comparing per-trial
     baseline vs response means.
  3. Peak detection: prominence-based; both peaks must fall within
     PEAK_KWARGS["search_window"].
  4. Height filter: both peaks must clear MIN_PEAK_HEIGHT_ABS sp/s above
     baseline.

A unit is "double-peak" if selectivity passes (excited=True) AND step 3
returns n_peaks==2 AND step 4 passes.

DO NOT define these dicts inline in scripts. Import from here so all
analyses move together.
"""

from thesis.ephys.config.typing_params import (
    PeakCountParams,
    PopulationPethParams,
    UnitSelectivityParams,
)

PETH_KWARGS: PopulationPethParams = dict(
    pre_seconds=0.1,
    post_seconds=0.15,
    binwidth_ms=10,
)

BASELINE_WINDOW = (-0.04, 0.0)  # TODO: extend to -0.05
RESPONSE_WINDOW = (0.03, 0.12)
# Peak search: start at stimulus onset (0.0 s). Response selectivity stays
# anchored to the canonical post-latency window above.
SEARCH_WINDOW = (0.0, 0.12)

SELECTIVITY_KWARGS: UnitSelectivityParams = dict(
    base_window=BASELINE_WINDOW,
    resp_window=RESPONSE_WINDOW,
    test="wilcoxon",
    correction="fdr_bh",
    alpha=0.05,
)

PEAK_KWARGS: PeakCountParams = dict(
    search_window=SEARCH_WINDOW,
    baseline_window=BASELINE_WINDOW,
    min_prominence_frac=0.25,
    min_distance_ms=20.0,
    binwidth_ms=10.0,
)

MIN_PEAK_HEIGHT_ABS = 5.0  # sp/s — both peaks must clear this above baseline
