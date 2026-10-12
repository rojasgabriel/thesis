"""Test one switch of the V1 GLM design on the same folds, unit by unit.

Question: does a term that is off in the model (a switch of `dmat.py`) predict
the spikes better?

Comparison: the design in use against the same design with the switch on. Both
are fit on the same units with the same trials and the same 10 folds, so the
comparison is paired over units, and each unit is also tested by itself over
its folds. This test does not shuffle the regressor, so it also works for a
term that is constant inside a trial (session drift), which the unique-deviance
step cannot test.

- Validation D² with and without the term (two-sided Wilcoxon signed-rank
  test over units).
- The two tests of `inclusion.py` on the cross-validated D² over the same 10
  folds: is the D² higher with the term (Wilcoxon over units), and do more
  units reliably need it than chance gives (binomial test). The fraction of
  units that are reliably worse with the term is given as the measured chance
  level. The same for the units outside the 20 selection units. The p values
  have no correction here; a set of terms is corrected where they are compared.

Units (`--units`, all by default: the importance of a regressor is tested on
all units), trials, alignment (first flash), baseline, and exclusions are those
of `fit.py`. The sampling unit is the unit. The fit with the term goes to a
directory next to the session directory (suffix `_<switch in lower case>`) and
is not run again when it is there. The script writes `term_test_<switch>.json`
in the session directory.

    uv run python -m thesis.ephys.analyses.glm.term_test --switch SESSION_DRIFT
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from thesis.ephys.analyses.glm import cli, dmat
from thesis.ephys.analyses.glm.fit import load_unit_records
from thesis.ephys.analyses.glm.flash_fixation_test import (
    _fold_scores,
    _validation_scores,
)
from thesis.ephys.analyses.glm.hold_effect import _fit_variant
from thesis.ephys.analyses.glm.inclusion import minimum_units, regressor_tests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--switch", required=True, help="A boolean of dmat.py.")
    parser.add_argument("--root", type=Path, default=cli.PoissonGLM().root)
    parser.add_argument("--units", type=cli._unit_set, default="all")
    args = parser.parse_args()
    if getattr(dmat, args.switch) is not False:
        raise SystemExit(f"dmat.{args.switch} is not a switch that is off.")
    suffix = f"_{args.switch.lower()}"
    root = _fit_variant(args.root, args.units, suffix, {args.switch: True})
    scores = [_validation_scores(r, args.units) for r in (args.root, root)]
    folds = [_fold_scores(r, args.units) for r in (args.root, root)]
    unit_ids = sorted(scores[0])
    if unit_ids != sorted(scores[1]):
        raise ValueError("The two designs were not fit on the same units.")
    gain = np.asarray([scores[1][unit] - scores[0][unit] for unit in unit_ids])
    paired = [u for u in unit_ids if len(folds[0][u]) == len(folds[1][u])]
    fold_gains = [folds[1][u] - folds[0][u] for u in paired]
    selection = {
        int(item["unit_id"])
        for item in load_unit_records(
            cli.PoissonGLM(root=args.root).fit_dir("sample") / "checkpoints",
            "unit_*_final.json",
        )
    }
    held_out = [gain for unit, gain in zip(paired, fold_gains) if unit not in selection]
    in_use = np.median([folds[0][u].mean() for u in paired])
    stats = {
        "switch": args.switch,
        "units": len(unit_ids),
        "validation_d2_gain": {
            "median": float(np.median(gain)),
            "units_better": int(np.sum(gain > 0)),
            "wilcoxon_p": float(wilcoxon(gain).pvalue),
        },
        # The two tests of `inclusion.py`, with no correction: one regressor.
        "inclusion_tests": regressor_tests(fold_gains),
        "inclusion_tests_held_out_units": regressor_tests(held_out),
        "units_above_chance_for_one_and_for_12_regressors": [
            minimum_units(len(paired)),
            minimum_units(len(paired), 12),
        ],
        "median_d2_in_use": float(in_use),
        "unit_ids": paired,
        "cross_validated_d2_gain_by_unit": [float(g.mean()) for g in fold_gains],
    }
    output = args.root / f"term_test_{args.switch.lower()}.json"
    output.write_text(json.dumps(stats, indent=1))
    print(json.dumps({k: stats[k] for k in list(stats)[:7]}, indent=1))
    print(output)


if __name__ == "__main__":
    main()
