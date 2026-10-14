"""Test the withdrawal kernel of the V1 GLM.

Question: does the model need the withdrawal kernel, and how wide must it be?

Comparison: for each design in `DESIGNS`, build the full design and run the
model-selection stage of `glm fit`: fit on the training trials and score D² on
the validation trials. Nothing else in the model changes; every design has the
fixation kernel. The comparison is paired: each unit is scored with
each design.

- "none": no withdrawal kernel.
- "0–100": the design in `dmat.py`, a causal kernel of 100 ms after the center
  exit.
- "±100", "±300": a kernel of that range in ms around the center exit.

Units, alignment, baseline, and exclusions are those of `fit.py`. Units are the
sets of 20 units of `grid.py` (`--samples`). The sampling unit is the unit. `design_optimization.py`
draws the scores; this module has no figure.

    uv run python -m thesis.ephys.analyses.glm.withdrawal_grid

A design that has saved scores in `withdrawal_grid/withdrawal_grid.json` is not
fit again.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from thesis.ephys.analyses.glm import cli
from thesis.ephys.analyses.glm.grid import score_designs

WIDE = {"WITHDRAWAL_PRE_S": 0.3, "WITHDRAWAL_WINDOW_S": 0.3, "WITHDRAWAL_BASES": 9}
DESIGNS = {
    "none": {"WITHDRAWAL_KERNEL": False},
    "0–100": {},
    "±100": {"WITHDRAWAL_PRE_S": 0.1},
    "±300": WIDE,
}
IN_USE = "0–100"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=cli.PoissonGLM().root)
    parser.add_argument(
        "--samples",
        type=int,
        default=1,
        help="Number of sets of 20 units (no unit is in two sets).",
    )
    args = parser.parse_args()
    output = args.root / "withdrawal_grid"
    scores = score_designs(
        args.root, output, "withdrawal_grid.json", DESIGNS, args.samples
    )
    print(f"{len(scores)} designs in {output}")


if __name__ == "__main__":
    main()
