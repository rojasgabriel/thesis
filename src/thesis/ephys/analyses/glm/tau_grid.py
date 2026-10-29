"""Choose the time constant of the flash-recovery term by prediction.

Question: the recovery term scales the flash kernel by
r = 1 − exp(−gap / τ) (`dmat.FLASH_RECOVERY`). τ = 120 ms came from the fit of
`flash_recovery.py`, but that fit had its c at a bound; with c free it gives
about 54 ms, and the model uses 50 ms since 2026-10-24. In the model, c goes
into each unit's weight and only τ sets the shape of the term. Which τ
predicts best?

Comparison: the design in use with `dmat.FLASH_RECOVERY_TAU_S` at each of
`TAUS_S`. Validation D² of each unit on the fixed sets of 20 units of the
grids (`--samples 4` gives 80 units of GRB006; GRB064 is held out of this
choice). Run it with the unit table of before the gain fix
(`results/glm/_before_gain_fix_2026_10_24/cache`, with a copy of the GRB006
`build_trial_table` files of `figures/cache` in it) to use the same 80 units as
the other grids. The comparison with `IN_USE` is paired
over units: the median change of D², and a two-sided Wilcoxon signed-rank test
with a Holm correction over the other values.

Rule, fixed before the run: τ stays at `IN_USE` unless another τ predicts
reliably better (Holm p < 0.05 and median change above 0); if several do, the
best one.

    THESIS_CACHE_DIR=<cache> nice -n 10 uv run python -m \\
        thesis.ephys.analyses.glm.tau_grid --samples 4

A design with saved scores of the same design signature in
`tau_grid/tau_grid.json` is not fit again (`grid.design_signature`). The
comparison goes to `tau_grid/comparison.json`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from thesis.ephys.analyses.glm import cli, dmat, grid
from thesis.ephys.analyses.glm.inclusion import holm

# The reference is the τ of the model, which is always in the grid.
TAUS_S = tuple(sorted({0.04, 0.06, 0.08, 0.12, 0.16, dmat.FLASH_RECOVERY_TAU_S}))
IN_USE = f"τ {1000 * dmat.FLASH_RECOVERY_TAU_S:.0f} ms"
DESIGNS = {f"τ {1000 * tau:.0f} ms": {"FLASH_RECOVERY_TAU_S": tau} for tau in TAUS_S}
NAME = "tau_grid"


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
    changes = {
        name: np.asarray([scores[name][unit] for unit in units]) - reference
        for name in DESIGNS
        if name != IN_USE
    }
    p_values = {name: float(wilcoxon(c).pvalue) for name, c in changes.items()}
    adjusted = holm(p_values)
    comparison = {
        "units": len(units),
        "in_use": IN_USE,
        "in_use_median_d2": float(np.median(reference)),
        **{
            name: {
                "median_change": float(np.median(change)),
                "units_better": int(np.sum(change > 0)),
                "wilcoxon_p": p_values[name],
                "wilcoxon_p_holm": adjusted[name],
            }
            for name, change in changes.items()
        },
    }
    (output / "comparison.json").write_text(json.dumps(comparison, indent=1))
    print(json.dumps(comparison, indent=1))


if __name__ == "__main__":
    main()
