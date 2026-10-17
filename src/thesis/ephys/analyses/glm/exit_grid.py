"""Test the basis of the response period and turn direction of the V1 GLM.

Question: the two kernels after the center exit share one basis (`dmat.exit_basis`):
8 raised cosines with log spacing from the exit to `TURN_DIRECTION_S` (1.3 s,
the 95th percentile of the time from the exit to the response entry in both
mice), and the last one held to the response entry. Does another end or
number of cosines predict held-out spikes better?

Comparison: for each design in `DESIGNS`, build the full design of 2026-10-15
with one change and run the model-selection stage of `glm fit`: fit on the
training trials and score D² on the validation trials. The comparison is
paired: each unit is scored with each design.

- "1.0 s", "1.3 s" (in use), "1.6 s": the end of the basis, 8 cosines.
- "6", "10": the number of cosines, to 1.3 s.
- "withdrawal\n0–100": the term at the exit of the design of 2026-10-13 (4
  cosines from 0 to 100 ms after the exit) in place of the response period;
  turn direction stays.

Units, alignment, baseline, and exclusions are those of `fit.py`. Units are the
sets of 20 units of `grid.py` (`--samples`). The sampling unit is the unit.
`design_optimization.py` draws the scores; this module has no figure.

    uv run python -m thesis.ephys.analyses.glm.exit_grid --samples 4

A design that has saved scores in `exit_grid/exit_grid.json` is not fit again.
"""

from __future__ import annotations

from thesis.ephys.analyses.glm import grid

DESIGNS = {
    "withdrawal\n0–100": {"RESPONSE_PERIOD_TERM": False, "WITHDRAWAL_KERNEL": True},
    "1.0 s": {"TURN_DIRECTION_S": 1.0},
    "1.3 s": {},
    "1.6 s": {"TURN_DIRECTION_S": 1.6},
    "6\ncosines": {"TURN_DIRECTION_BASES": 6},
    "10\ncosines": {"TURN_DIRECTION_BASES": 10},
}
IN_USE = "1.3 s"


if __name__ == "__main__":
    grid.main(DESIGNS, "exit_grid", __doc__.splitlines()[0])
