"""Test the parameterization of the session-drift term of the V1 GLM.

Question: the model can have a term for a slow change of the rate over the
session (`dmat.SESSION_DRIFT`). How many raised cosines does it need, and does
a cubic spline do better?

Comparison: the final design with one thing changed, on the fixed 20-unit
sample, with the validation D² of each unit. The designs are in `DESIGNS`:

- "none": no drift term.
- "4", "8", "12": that number of raised cosines over the session (one fewer
  column, because the last one is left out).
- "spline": a cubic spline with 6 knots (7 columns), the first parameterization that was
  tested.

The comparison is paired over units. Where two options do not differ reliably,
the model uses the simpler one. Units, trials, alignment, baseline, and
exclusions are those of `fit.py`.

    uv run python -m thesis.ephys.analyses.glm.drift_grid

A design that has saved scores in `drift_grid/drift_grid.json` is not fit
again. Each design takes about two minutes.
"""

from __future__ import annotations

from thesis.ephys.analyses.glm import grid

DESIGNS = {
    "none": {"SESSION_DRIFT": False},
    "4": {"SESSION_DRIFT": True, "DRIFT_BASES": 4},  # the parameterization in `dmat.py`
    "8": {"SESSION_DRIFT": True, "DRIFT_BASES": 8},
    "12": {"SESSION_DRIFT": True, "DRIFT_BASES": 12},
    "spline": {"SESSION_DRIFT": True, "DRIFT_SPLINE": True},
}


if __name__ == "__main__":
    grid.main(DESIGNS, "drift_grid", __doc__.splitlines()[0])
