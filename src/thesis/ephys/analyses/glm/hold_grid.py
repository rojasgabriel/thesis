"""Test the parameterization of the stimulus-during-hold term of the V1 GLM.

Question: units fire less from about 80 ms after the first flash until the
animal leaves the center port. Which parameterization of the term for that time predicts
held-out spikes best?

Comparison: for each design in `DESIGNS`, build the full design and run the
model-selection stage of `glm fit`: fit on the training trials and score D² on
the validation trials. Nothing else in the model changes. The comparison is
paired: each unit is scored with each design.

- "none": no term for the hold.
- "step,\nentry": one column that is 1 from center entry to center exit.
- "step,\nflash": one column that is 1 from the first flash to center exit.
- "kernel": the design in `dmat.py`. One kernel from the first flash to the
  center exit, 10 bases at even spacing in log(t + 50 ms), and no step.
- "linear\nkernel": the same kernel with bases at even spacing in time.
- "fine\nkernel": the kernel of 2026-10-15, 12 bases and a stretch of 25 ms,
  about 20 ms apart near the first-flash response. (Until 2026-10-16 that was
  the kernel in use, and this grid tested 10 bases with 50 ms as the "coarse
  kernel", which predicted better and replaced it.)
- "no\nfixation": the design in `dmat.py` without the column for the fixation
  before the stimulus (center entry to first flash).

Units, alignment, baseline, and exclusions are those of `fit.py`. Units are the
fixed 20-unit sample. The sampling unit is the unit. `design_optimization.py`
draws the scores; this module has no figure.

    uv run python -m thesis.ephys.analyses.glm.hold_grid

A design that has saved scores in `hold_grid/hold_grid.json` is not fit again.
"""

from __future__ import annotations

from thesis.ephys.analyses.glm import grid

STEP = {"HOLD_KERNEL": False, "IN_PORT_COLUMN": True}
DESIGNS = {
    "none": {"HOLD_KERNEL": False},
    "step,\nentry": STEP,
    "step,\nflash": {**STEP, "STATE_FROM_FIRST_FLASH": True},
    "kernel": {},
    "linear\nkernel": {"HOLD_LOG_STRETCH_S": 0.0},
    "fine\nkernel": {"HOLD_BASES": 12, "HOLD_LOG_STRETCH_S": 0.025},
    "no\nfixation": {"FIXATION_COLUMN": False},
    "penalty\n÷ 100": {"HOLD_PENALTY_SCALE": 10.0},
}
IN_USE = "kernel"


if __name__ == "__main__":
    grid.main(DESIGNS, "hold_grid", __doc__.splitlines()[0])
