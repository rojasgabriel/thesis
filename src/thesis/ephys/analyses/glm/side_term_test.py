"""Test a term for the side of the response after the center exit.

Question: after the animal leaves the center port, about half of the V1 units
fire more for a response to one side, with or without the stimulus
(`side_preference.py`). The model has no term for the side. Does one predict
the spikes better?

Comparison: the design in use against the same design with one more column,
+1 for a response to the right port and -1 for the left port from the center
exit on (`dmat.SIDE_COLUMN`). Both designs are scored on the same units and
the same trials, so the comparison is paired over units.

- Validation D² of each unit with and without the term (two-sided Wilcoxon
  signed-rank test), which is the measure of the other design choices.
- For each unit, the held-out D² of the 10 cross-validation folds with the
  term against the same folds without it (paired t-test, one-sided,
  p < `RELIABLE_ALPHA`), as for the regressors of the main figure; a term with
  no effect gives about 5%.
- The weight of the column for each unit, as a change in log rate for a
  response to the right against the left (two times the weight of the ±1
  column), from the final refit.

Units (`--units`, the 20 selection units by default), trials, alignment (first
flash), baseline, and exclusions are those of `fit.py`. The sampling unit is
the unit. The fit with the term goes to a directory next to the session
directory (suffix `ROOT_SUFFIX`) and is not run again when it is there. The
script writes `side_term_test.json` in the session directory.

    uv run python -m thesis.ephys.analyses.glm.side_term_test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel, wilcoxon

from thesis.ephys.analyses.glm import cli, dmat
from thesis.ephys.analyses.glm.fit import load_unit_records
from thesis.ephys.analyses.glm.flash_fixation_test import (
    _fold_scores,
    _validation_scores,
)
from thesis.ephys.analyses.glm.hold_effect import _fit_variant
from thesis.ephys.analyses.glm.unique_dev import RELIABLE_ALPHA

ROOT_SUFFIX = "_side_term"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=cli.PoissonGLM().root)
    parser.add_argument("--units", type=cli._unit_set, default="sample")
    args = parser.parse_args()
    root = _fit_variant(args.root, args.units, ROOT_SUFFIX, {"SIDE_COLUMN": True})
    scores = [_validation_scores(r, args.units) for r in (args.root, root)]
    folds = [_fold_scores(r, args.units) for r in (args.root, root)]
    unit_ids = sorted(scores[0])
    if unit_ids != sorted(scores[1]):
        raise ValueError("The two designs were not fit on the same units.")
    gain = np.asarray([scores[1][unit] - scores[0][unit] for unit in unit_ids])
    paired = [u for u in unit_ids if len(folds[0][u]) == len(folds[1][u])]
    fold_gain = np.asarray([(folds[1][u] - folds[0][u]).mean() for u in paired])
    fold_p = np.asarray([ttest_rel(folds[1][u], folds[0][u]).pvalue for u in paired])
    reliable = fold_p / 2 < RELIABLE_ALPHA

    model = cli.PoissonGLM(root=root)
    metadata = json.loads(model.design.with_suffix(".json").read_text())
    column = metadata["column_names"].index(f"{dmat.SIDE}_basis_01")
    scale = metadata["training_scale"][column]
    final = {
        int(item["unit_id"]): 2 * item["coefficients"][column] / scale
        for item in load_unit_records(
            model.fit_dir(args.units) / "checkpoints", "unit_*_final.json"
        )
    }
    weight = np.asarray([final[unit] for unit in unit_ids])
    reference = np.median([scores[0][unit] for unit in unit_ids])
    stats = {
        "units": len(unit_ids),
        "validation_d2_gain": {
            "median": float(np.median(gain)),
            "median_relative_to_in_use": float(np.median(gain) / reference),
            "units_better": int(np.sum(gain > 0)),
            "wilcoxon_p": float(wilcoxon(gain).pvalue),
        },
        "cross_validated_d2_gain": {
            "units_with_the_same_folds": len(paired),
            "median": float(np.median(fold_gain)),
            "wilcoxon_p": float(wilcoxon(fold_gain).pvalue),
            "fraction_units_better_with_term": float(
                np.mean(reliable & (fold_gain > 0))
            ),
            "fraction_units_worse_with_term": float(
                np.mean(reliable & (fold_gain < 0))
            ),
        },
        "delta_log_rate_right_minus_left": {
            "median": float(np.median(weight)),
            "median_size": float(np.median(np.abs(weight))),
            "units_above_0": int(np.sum(weight > 0)),
            "by_unit": weight.tolist(),
        },
        "unit_ids": unit_ids,
        "validation_d2_gain_by_unit": gain.tolist(),
    }
    output = args.root / "side_term_test.json"
    output.write_text(json.dumps(stats, indent=1))
    print(json.dumps({k: stats[k] for k in list(stats)[:4]}, indent=1)[:1400])
    print(output)


if __name__ == "__main__":
    main()
