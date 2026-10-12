"""Test the form of the session-drift term of the V1 GLM.

Question: the model can have a term for a slow change of the rate over the
session (`dmat.SESSION_DRIFT`). How many raised cosines does it need, and does
a cubic spline do better?

Comparison: the final design with one thing changed, on the fixed 20-unit
sample, with the validation D² of each unit. The designs are in `DESIGNS`:

- "none": no drift term.
- "4", "8", "12": that number of raised cosines over the session (one fewer
  column, because the last one is left out).
- "spline": a cubic spline with 6 knots (7 columns), the first form that was
  tested.

The comparison is paired over units. Where two options do not differ reliably,
the model uses the simpler one. Units, trials, alignment, baseline, and
exclusions are those of `fit.py`.

    uv run python -m thesis.ephys.analyses.glm.drift_grid

A design that has saved scores in `drift_grid/drift_grid.json` is not fit
again. Each design takes about two minutes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from thesis.ephys.analyses.glm import cli
from thesis.ephys.analyses.glm.adaptation_grid import score_design

DESIGNS = {
    "none": {},
    "4": {"SESSION_DRIFT": True, "DRIFT_BASES": 4},
    "8": {"SESSION_DRIFT": True, "DRIFT_BASES": 8},
    "12": {"SESSION_DRIFT": True, "DRIFT_BASES": 12},
    "spline": {"SESSION_DRIFT": True, "DRIFT_SPLINE": True},
}


def run_grid(source: Path, output: Path) -> dict[str, dict]:
    """Score every design that has no saved scores; return all designs."""
    path = output / "drift_grid.json"
    scores = json.loads(path.read_text()) if path.exists() else {}
    for index, (name, settings) in enumerate(DESIGNS.items()):
        if name in scores:
            continue
        scores[name] = score_design(
            source, output / "work", f"design_{index}", settings
        )
        output.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(scores, indent=1))
        print(f"Scored {name!r}.", flush=True)
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=cli.PoissonGLM().root)
    args = parser.parse_args()
    scores = run_grid(args.root, args.root / "drift_grid")
    print(f"{len(scores)} designs in {args.root / 'drift_grid'}")


if __name__ == "__main__":
    main()
