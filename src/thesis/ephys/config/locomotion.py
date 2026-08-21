"""Canonical parameters for locomotion analyses.

Last reviewed: 2026-04-29.

Pipeline:
  1. PETH (sp/s) with the same unsmoothed event-aligned bins used by the
     double-peak analyses.
  2. Baseline subtraction uses the shared stationary baseline window.

DO NOT define these dicts or windows inline in scripts. Import from here so
all locomotion analyses move together.
"""

from thesis.ephys.config.typing_params import PopulationPethParams

PETH_KWARGS: PopulationPethParams = dict(
    pre_seconds=0.1,
    post_seconds=0.15,
    binwidth_ms=10,
    t_rise=None,
    t_decay=None,
)

BASELINE_WINDOW = (-0.04, 0.0)  # TODO: extend to -0.05
RESP_WINDOW = (0.03, 0.12)
