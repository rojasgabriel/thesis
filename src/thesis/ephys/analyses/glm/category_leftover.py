"""What carries the category after the flash terms are divided out?

Question: without the flash terms, the decoder still finds the category after
the center exit (`decoding_without_terms.py`). Is that leftover a category
signal that is not visual, or a miss of the flash model at high flash rates?

Comparison, per unit, in the `WINDOW_S` after the center exit: the log of the
mean rate of high-rate trials minus that of low-rate trials, within each
choice, averaged over the two choices (so the choice cannot make the
difference). Three versions of the same trials and units, from
`decoding_without_terms.session_rates`: observed; without the flash terms
(the rest); without the other terms. The flash part is observed minus rest;
the other part is observed minus the version without the other terms. Each
trial uses the fold model that held it out.

- Sessions, units and trials: those of `decoding_without_terms.py` (GLM
  sessions, GLM units with a final fit, boundary trials left out). A unit with
  no spike in one group has no log ratio and is left out (the count is in the
  report).
- Sampling unit: the unit; each mouse alone; medians over units; two-sided
  Wilcoxon test of the rest against 0.
- Limit: the original script of 2026-10-29 was lost, and its exact
  definition is not known; this definition (log of the group mean) gives
  other values for GRB006 (rest +0.20, not +0.07), with the same sign.

    THESIS_CACHE_DIR=figures/cache nice -n 10 uv run python -m \
        thesis.ephys.analyses.glm.category_leftover
"""

from __future__ import annotations

import json

import numpy as np
from scipy.stats import wilcoxon

from thesis.ephys.analyses.glm.decoding_without_terms import OUT, session_rates
from thesis.ephys.analyses.glm.flash_kernels import SESSIONS


def high_minus_low(rates: np.ndarray, category: np.ndarray, choice: np.ndarray):
    """Return log(high / low) of each unit, the mean over the choices."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.mean(
            [
                np.log(rates[(category == 1) & (choice == c)].mean(0))
                - np.log(rates[(category == 0) & (choice == c)].mean(0))
                for c in np.unique(choice)
            ],
            0,
        )


def main() -> None:
    report = {}
    for session in SESSIONS:
        data = session_rates(session)
        values = {
            version: high_minus_low(
                data["rates"][("after exit", version)], data["category"], data["choice"]
            )
            for version in (
                "observed",
                "without the flash terms",
                "without the other terms",
            )
        }
        ok = np.all([np.isfinite(v) for v in values.values()], 0)
        observed, rest, other = (v[ok] for v in values.values())
        report[session] = {
            "units": int(ok.sum()),
            "units_left_out": int((~ok).sum()),
            "median_observed": float(np.median(observed)),
            "median_flash_part": float(np.median(observed - rest)),
            "median_rest": float(np.median(rest)),
            "fraction_rest_above_0": float(np.mean(rest > 0)),
            "rest_wilcoxon_p": float(wilcoxon(rest).pvalue),
            "median_other_part": float(np.median(observed - other)),
        }
    path = OUT / "category_leftover.json"
    path.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
