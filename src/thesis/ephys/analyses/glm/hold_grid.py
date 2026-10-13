"""Test the form of the stimulus-during-hold term of the V1 GLM.

Question: units fire less from about 80 ms after the first flash until the
animal leaves the center port. Which form of the term for that time predicts
held-out spikes best?

Comparison: for each design in `DESIGNS`, build the full design and run the
model-selection stage of `glm fit`: fit on the training trials and score D² on
the validation trials. Nothing else in the model changes. The comparison is
paired: each unit is scored with each design.

- "none": no term for the hold.
- "step,\nentry": one column that is 1 from center entry to center exit.
- "step,\nflash": one column that is 1 from the first flash to center exit.
- "step +\nonset": that column, and a kernel of 150 ms on the first flash
  (the bases of the flash kernel). The term is flat after 150 ms.
- "kernel": the design in `dmat.py`. One kernel from the first flash to the
  center exit, with bases at even spacing in log(t + 25 ms), and no step.
- "linear\nkernel": the same kernel with bases at even spacing in time.
- "coarse\nkernel": the same kernel with 10 bases and a stretch of 50 ms, which
  has bases about 40 ms apart near the first-flash response.
- "no\nfixation": the design in `dmat.py` without the column for the fixation
  before the stimulus (center entry to first flash).

Units, alignment, baseline, and exclusions are those of `fit.py`. Units are the
fixed 20-unit sample. The sampling unit is the unit. `design_optimization.py`
draws the scores; this module has no figure.

    uv run python -m thesis.ephys.analyses.glm.hold_grid

A design that has saved scores in `hold_grid/hold_grid.json` is not fit again.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from thesis.ephys.analyses.glm import cli
from thesis.ephys.analyses.glm.adaptation_grid import score_designs

STEP = {"HOLD_KERNEL": False, "IN_PORT_COLUMN": True}
DESIGNS = {
    "none": {"HOLD_KERNEL": False},
    "step,\nentry": STEP,
    "step,\nflash": {**STEP, "STATE_FROM_FIRST_FLASH": True},
    "step +\nonset": {**STEP, "STATE_FROM_FIRST_FLASH": True, "FIRST_FLASH_TERM": True},
    "kernel": {},
    "linear\nkernel": {"HOLD_LOG_STRETCH_S": 0.0},
    "coarse\nkernel": {"HOLD_BASES": 10, "HOLD_LOG_STRETCH_S": 0.05},
    "no\nfixation": {"FIXATION_COLUMN": False},
}
IN_USE = "kernel"


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
    scores = score_designs(
        args.root, args.root / "hold_grid", "hold_grid.json", DESIGNS, args.samples
    )
    print(f"{len(scores)} designs in {args.root / 'hold_grid'}")


if __name__ == "__main__":
    main()
