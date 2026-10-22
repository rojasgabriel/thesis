"""Test the end of the fixation kernel of the V1 GLM.

Question: the fixation kernel runs from the first flash to `HOLD_WINDOW_S`
(0.7 s, the 95th percentile of the fixation length in both mice) and keeps its
last value to the center exit. With the end at 0.95 s (the design of
2026-10-13) the kernel after 700 ms was not stable in split halves. Does
another end predict held-out spikes better?

Comparison: for each design in `DESIGNS`, build the full design in use with
the fixation kernel to that end (10 cosines, the same log spacing) and run the
model-selection stage of `glm fit`: fit on the training trials and score D² on
the validation trials. The comparison is paired: each unit is scored with each
design.

Units, alignment, baseline, and exclusions are those of `fit.py`. Units are the
sets of 20 units of `grid.py` (`--samples`). The sampling unit is the unit.
`design_optimization.py` draws the scores; this module has no figure.

    uv run python -m thesis.ephys.analyses.glm.hold_window_grid --samples 4

A design that has saved scores in `hold_window_grid/hold_window_grid.json` is
not fit again.
"""

from __future__ import annotations

from thesis.ephys.analyses.glm import grid

DESIGNS = {
    "0.5 s": {"HOLD_WINDOW_S": 0.5},
    "0.7 s": {},
    "0.95 s": {"HOLD_WINDOW_S": 0.95},
}
IN_USE = "0.7 s"


if __name__ == "__main__":
    grid.main(DESIGNS, "hold_window_grid", __doc__.splitlines()[0])
