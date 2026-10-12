"""Test candidate regressors of the V1 GLM on all units, together.

Question: which of the terms that are not in the model does each unit need,
when all of them are in the design at the same time?

Comparison: one design, the design in use plus every term of `CANDIDATES`. It
is fit on all units, and the unique-deviance step of `unique_dev.py` then gives
the unique ΔD² of each regressor: the D² that the model loses when that
regressor is shuffled in time within each trial, on held-out trials. So each
term is tested with all the other terms in the model, and every unit is its
own control.

- Reported for each regressor: the median unique ΔD² over units and the
  fraction of units with a unique ΔD² above zero (one-sided t-test over the 10
  folds, p < `RELIABLE_ALPHA`; a term with no effect gives about 5%), for all
  units and for the units outside the 20 selection units.
- The shuffle is within a trial, so it cannot test a term that is constant in
  a trial (session drift is not a candidate for this reason).

Units (`--units`, all by default), trials, alignment (first flash), baseline,
and exclusions are those of `fit.py`. The sampling unit is the unit. The fit
goes to a directory next to the session directory (suffix `ROOT_SUFFIX`); a
stage that is done is not run again. The script writes `candidate_terms.json`
in the session directory.

    uv run python -m thesis.ephys.analyses.glm.candidate_terms
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import ttest_1samp

from thesis.ephys.analyses.glm import cli, dmat
from thesis.ephys.analyses.glm.fit import load_unit_records
from thesis.ephys.analyses.glm.hold_effect import _fit_variant
from thesis.ephys.analyses.glm.unique_dev import RELIABLE_ALPHA, UNIQUE

ROOT_SUFFIX = "_candidates"
CANDIDATES = {
    "SIDE_COLUMN": dmat.SIDE,
    "FLASH_AFTER_EXIT_TERM": dmat.FLASH_AFTER_EXIT,
    "ENTRY_KERNEL": dmat.ENTRY,
    "FLASH_RECOVERY_TERM": dmat.FLASH_RECOVERY,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=cli.PoissonGLM().root)
    parser.add_argument("--units", type=cli._unit_set, default="all")
    args = parser.parse_args()
    switches = dict.fromkeys(CANDIDATES, True)
    root = _fit_variant(args.root, args.units, ROOT_SUFFIX, switches)
    model = cli.PoissonGLM(root=root)
    unique_dir = model.fit_dir(args.units) / "checkpoints" / "unique"
    final = load_unit_records(
        model.fit_dir(args.units) / "checkpoints", "unit_*_final.json"
    )
    if len(list(unique_dir.glob("unit_*.json"))) < len(final):
        # The unique step reads the bases of the design, so the switches stay on.
        saved = {name: getattr(dmat, name) for name in switches}
        for name in switches:
            setattr(dmat, name, True)
        try:
            model.unique(args.units)
        finally:
            for name, value in saved.items():
                setattr(dmat, name, value)
    records = load_unit_records(unique_dir, "unit_*.json")
    sample_path = cli.PoissonGLM(root=args.root).fit_dir("sample") / "checkpoints"
    selection = {
        int(item["unit_id"])
        for item in load_unit_records(sample_path, "unit_*_final.json")
    }
    held_out = np.asarray([int(item["unit_id"]) not in selection for item in records])
    stats = {"units": len(records), "held_out_units": int(held_out.sum())}
    for name in records[0]["groups"]:
        values = np.asarray([item["groups"][name][UNIQUE] for item in records])
        reliable = np.asarray(
            [
                ttest_1samp(
                    item["groups"][name]["folds"]["unique"], 0, alternative="greater"
                ).pvalue
                < RELIABLE_ALPHA
                for item in records
            ]
        )
        stats[name] = {
            "candidate": name in CANDIDATES.values(),
            "median_unique_d2": float(np.median(values)),
            "mean_unique_d2": float(values.mean()),
            "fraction_units_reliable": float(reliable.mean()),
            "fraction_held_out_units_reliable": float(reliable[held_out].mean()),
        }
    full = np.asarray([item["complete_test_deviance_explained"] for item in records])
    stats["median_d2_of_the_design_with_all_candidates"] = float(np.median(full))
    output = args.root / "candidate_terms.json"
    output.write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))
    print(output)


if __name__ == "__main__":
    main()
