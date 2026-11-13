"""Choose the penalty of each band of the V1 GLM (banded ridge) by prediction.

Question: the model has one L2 penalty for all terms. With it, the fixation
level is too small, and a 100 times weaker penalty on the fixation kernel
changed the flash × after exit result of GRB006 (a sensitivity fit, removed
2026-10-29).
That factor was chosen by hand. Does a penalty for each band of regressors,
chosen by a fixed rule, predict better?

Comparison: the design in use with the factor of one band changed
(`dmat.BAND_SCALES`: flash, task, nuisance; spike history is the reference
band at 1). A factor f makes the penalty of that band f² times weaker; the
overall penalty is still chosen for each unit. Step 1 (`DESIGNS`): one band at
a time, at each of `FACTORS`. Validation D² of each unit on the fixed sets of
20 units of the grids (`--samples 4` gives 80 units of GRB006; GRB064 is held
out of this choice). The comparison with `IN_USE` is paired over units: the
median change of D² and a two-sided Wilcoxon signed-rank test, with a Holm
correction over the designs of the step.

Rule, fixed before the run: a band keeps factor 1 unless a factor predicts
reliably better (Holm p < 0.05 and median change above 0); if several do, the
best one. Step 2 tests the chosen factors together against `IN_USE` in the
same way, and only then are both mice fit again with them.

    THESIS_CACHE_DIR=figures/cache nice -n 10 uv run python -m \\
        thesis.ephys.analyses.glm.band_grid --samples 4

A design with saved scores in `band_grid/band_grid.json` is not fit again.
The comparison goes to `band_grid/comparison.json`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from thesis.ephys.analyses.glm import cli, dmat, grid
from thesis.ephys.analyses.glm.inclusion import holm

FACTORS = (1 / 3, 3.0, 10.0)
IN_USE = "all bands 1"
DESIGNS = {
    IN_USE: {},
    **{
        f"{band} ×{factor:.3g}": {"BAND_SCALES": {**dmat.BAND_SCALES, band: factor}}
        for band in dmat.BAND_SCALES
        for factor in FACTORS
    },
}
NAME = "band_grid"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=cli.PoissonGLM().root)
    parser.add_argument("--samples", type=int, default=1)
    args = parser.parse_args()
    output = args.root / NAME
    scores = grid.score_designs(
        args.root, output, f"{NAME}.json", DESIGNS, args.samples
    )
    units = sorted(set.intersection(*(set(s) for s in scores.values())))
    reference = np.asarray([scores[IN_USE][unit] for unit in units])
    others = [name for name in DESIGNS if name != IN_USE]
    changes = {
        name: np.asarray([scores[name][unit] for unit in units]) - reference
        for name in others
    }
    p_values = {
        name: float(wilcoxon(change).pvalue) for name, change in changes.items()
    }
    adjusted = holm(p_values)
    comparison = {"units": len(units), "in_use": IN_USE}
    for name, change in changes.items():
        comparison[name] = {
            "median_change": float(np.median(change)),
            "units_better": int(np.sum(change > 0)),
            "wilcoxon_p": p_values[name],
            "wilcoxon_p_holm": adjusted[name],
        }
    (output / "comparison.json").write_text(json.dumps(comparison, indent=1))
    print(json.dumps(comparison, indent=1))


if __name__ == "__main__":
    main()
