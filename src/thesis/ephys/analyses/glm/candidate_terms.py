"""Test candidate regressors of the V1 GLM on all units, together.

Question: which of the terms that are not in the model does each unit need,
when all of them are in the design at the same time?

Comparison: one design, the design in use plus every term of `CANDIDATES`. It
is fit on all units, and the unique-deviance step of `unique_dev.py` then gives
the unique ΔD² of each regressor: the D² that the model loses when that
regressor is shuffled in time within each trial, on held-out trials. So each
term is tested with all the other terms in the model, and every unit is its
own control.

- For each regressor: the two tests of `inclusion.py` on its unique ΔD² in the
  folds of each unit (is the D² higher with the regressor, and do more units
  reliably need it than chance gives), with a Holm correction over all the
  regressors of the design, and the fraction of units that are reliably worse
  with it. The fraction of units that need it is also given for the units
  outside the 20 selection units.
- The shuffle is within a trial. Session drift and the previous side do not
  change in a trial, so they are shuffled over trials (`unique_dev.py`).

Units (`--units`, all by default), trials, alignment (first flash), baseline,
and exclusions are those of `fit.py`. The sampling unit is the unit. The fit
goes to a directory next to the session directory (suffix `ROOT_SUFFIX`); a
stage that is done is not run again. The script writes
`candidate_terms<ROOT_SUFFIX>.json` in the session directory.

    uv run python -m thesis.ephys.analyses.glm.candidate_terms
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from thesis.ephys.analyses.glm import cli, dmat
from thesis.ephys.analyses.glm.fit import analysis_unit_indices, load_unit_records
from thesis.ephys.analyses.glm.flash_kernels import SESSIONS
from thesis.ephys.analyses.glm.hold_effect import _fit_variant
from thesis.ephys.analyses.glm.inclusion import decide, minimum_units, regressor_tests
from thesis.ephys.units import fetch_unit_table

# Earlier runs (other suffixes, other candidates) are in docs/glm_decisions.md.
# This run (2026-10-21): a kernel for the first flash after the exit, the only
# response the model missed (`first_flash_after_exit.py`).
ROOT_SUFFIX = "_first_flash_after_exit"
CANDIDATES = {"FIRST_FLASH_AFTER_EXIT_TERM": dmat.FIRST_FLASH_AFTER_EXIT}
# The units of the design choices: the sets of 20 units of the grids.
SELECTION_SETS = 4


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
        with dmat.switched(switches):
            model.unique(args.units)
    records = load_unit_records(unique_dir, "unit_*.json")
    metadata = dmat._load_windows(cli.PoissonGLM(root=args.root).windows)["metadata"]
    table = fetch_unit_table(
        metadata["subject_name"],
        metadata["session_name"],
        unit_criteria_id=1,
        stability_param_id=0,
        include_metrics=False,
    )
    # The design choices used units of the first session only; in another
    # session no unit was used.
    selection = (
        {
            int(table.iloc[row]["unit_id"])
            for sample in range(SELECTION_SETS)
            for row in analysis_unit_indices(
                table, "sample" + (str(sample) if sample else "")
            )
        }
        if args.root.name == SESSIONS[0]
        else set()
    )
    held_out = np.asarray([int(item["unit_id"]) not in selection for item in records])
    stats = {"units": len(records), "held_out_units": int(held_out.sum())}
    # Every regressor of the design (the broad group "task" is their sum).
    names = [name for name in records[0]["groups"] if name != "task"]
    fold_gains = {
        name: [np.asarray(item["groups"][name]["folds"]["unique"]) for item in records]
        for name in names
    }
    tests = decide({name: regressor_tests(fold_gains[name]) for name in names})
    for name in names:
        outside = [g for g, keep in zip(fold_gains[name], held_out) if keep]
        stats[name] = {
            "candidate": name in CANDIDATES.values(),
            **tests[name],
            "fraction_held_out_units_that_need_it": regressor_tests(outside)[
                "fraction_units_that_need_it"
            ],
        }
    stats["units_above_chance_for_this_number_of_regressors"] = minimum_units(
        len(records), len(names)
    )
    full = np.asarray([item["complete_test_deviance_explained"] for item in records])
    stats["median_d2_of_the_design_with_all_candidates"] = float(np.median(full))
    output = args.root / f"candidate_terms{ROOT_SUFFIX}.json"
    output.write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))
    print(output)


if __name__ == "__main__":
    main()
