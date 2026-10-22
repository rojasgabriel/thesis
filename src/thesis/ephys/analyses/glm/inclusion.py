"""The two tests that decide if a regressor is in the V1 GLM.

Question: does the population of units need a regressor? Two tests, both on
held-out trials and on all units. A regressor is in the model when it passes
both, after a Holm correction for the number of regressors that are tested
together.

- Gain: is the cross-validated D² higher with the regressor than without it?
  The gain of a unit is the mean over its folds of the D² with the regressor
  minus the D² without it (or with it shuffled). One-sided Wilcoxon
  signed-rank test of the gains over units against zero.
- Units: do more units reliably need the regressor than chance gives? A unit
  needs it when its fold gains are above zero (one-sided t-test over folds,
  p < `ALPHA`). A regressor with no effect gives `ALPHA` of the units. One-sided
  binomial test of the count against `ALPHA`.

Reported with them, as a measure and not as a test: the fraction of units that
are reliably worse with the regressor (the same t-test in the other
direction). A real regressor cannot make a unit predict worse except by noise,
so this fraction is the measured rate of false positives of the unit test.

Limits: the folds of a unit share training trials, and the units share the
trials, so the tests are not fully independent. The fraction that is worse
shows how much that matters.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import binomtest, ttest_1samp, wilcoxon

ALPHA = 0.05


def regressor_tests(fold_gains: list[np.ndarray]) -> dict:
    """Return the gain test and the unit-count test for one regressor.

    `fold_gains` has one array for each unit: the gain of the regressor in each
    of its folds.
    """
    gains = np.asarray([np.mean(values) for values in fold_gains])
    better = np.asarray(
        [ttest_1samp(v, 0, alternative="greater").pvalue < ALPHA for v in fold_gains]
    )
    worse = np.asarray(
        [ttest_1samp(v, 0, alternative="less").pvalue < ALPHA for v in fold_gains]
    )
    return {
        "units": len(gains),
        "median_gain": float(np.median(gains)),
        "mean_gain": float(gains.mean()),
        "gain_wilcoxon_p": float(wilcoxon(gains, alternative="greater").pvalue),
        "units_that_need_it": int(better.sum()),
        "fraction_units_that_need_it": float(better.mean()),
        "units_binomial_p": float(
            binomtest(
                int(better.sum()), len(gains), ALPHA, alternative="greater"
            ).pvalue
        ),
        "fraction_units_worse": float(worse.mean()),
    }


def holm(p_values: dict[str, float]) -> dict[str, float]:
    """Return the Holm-adjusted p value of each name."""
    order = sorted(p_values, key=p_values.get)
    adjusted, largest = {}, 0.0
    for rank, name in enumerate(order):
        largest = max(largest, min(1.0, (len(order) - rank) * p_values[name]))
        adjusted[name] = largest
    return adjusted


def decide(tests: dict[str, dict]) -> dict[str, dict]:
    """Add the Holm-adjusted p values and the decision to the tests of a set."""
    for key in ("gain_wilcoxon_p", "units_binomial_p"):
        adjusted = holm({name: values[key] for name, values in tests.items()})
        for name, value in adjusted.items():
            tests[name][f"{key}_holm"] = value
    for values in tests.values():
        values["passes_both_tests"] = bool(
            values["gain_wilcoxon_p_holm"] < ALPHA
            and values["units_binomial_p_holm"] < ALPHA
        )
    return tests


def minimum_units(units: int, tested: int = 1) -> int:
    """Return the smallest count of units that is above chance for `tested` regressors."""
    return next(
        count
        for count in range(units + 1)
        if binomtest(count, units, ALPHA, alternative="greater").pvalue < ALPHA / tested
    )


if __name__ == "__main__":  # a check of the logic on made-up gains
    rng = np.random.default_rng(0)
    null = [rng.normal(0, 1, 10) for _ in range(167)]
    real = [rng.normal(0.6 * (index < 50), 1, 10) for index in range(167)]
    out = decide({"null": regressor_tests(null), "real": regressor_tests(real)})
    assert not out["null"]["passes_both_tests"] and out["real"]["passes_both_tests"]
    assert 0.0 < out["null"]["fraction_units_that_need_it"] < 0.12
    assert holm({"a": 0.01, "b": 0.04, "c": 0.03}) == {"a": 0.03, "c": 0.06, "b": 0.06}
    assert minimum_units(167) == 14 and minimum_units(167, 12) == 18
    print("inclusion checks passed")
