"""Test the end of the fixation kernel of the V1 GLM.

Question: the fixation kernel runs from the first flash to `HOLD_WINDOW_S`
(0.7 s, the 95th percentile of the fixation length in both mice) and keeps its
last value to the center exit. With the end at 0.95 s (the design of
2026-10-13) the kernel after 700 ms was not stable in split halves. Does
another end predict held-out spikes better?

Comparison: for each design in `DESIGNS`, build the full design of 2026-10-15
with the fixation kernel to that end (10 cosines, the same log spacing) and
run the model-selection stage of `glm fit`: fit on the training trials and
score D² on the validation trials. The comparison is paired: each unit is
scored with each design. The design in use is the "1.3 s" design of
`exit_grid.py`; its scores are taken from there when they exist, so it is not
fit two times.

Units, alignment, baseline, and exclusions are those of `fit.py`. Units are the
sets of 20 units of `grid.py` (`--samples`). The sampling unit is the unit.
`design_optimization.py` draws the scores; this module has no figure.

    uv run python -m thesis.ephys.analyses.glm.hold_window_grid --samples 4

A design that has saved scores in `hold_window_grid/hold_window_grid.json` is
not fit again.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from thesis.ephys.analyses.glm import cli, exit_grid
from thesis.ephys.analyses.glm.grid import score_designs

DESIGNS = {
    "0.5 s": {"HOLD_WINDOW_S": 0.5},
    "0.7 s": {},
    "0.95 s": {"HOLD_WINDOW_S": 0.95},
}
IN_USE = "0.7 s"


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
    output = args.root / "hold_window_grid"
    path = output / "hold_window_grid.json"
    shared = args.root / "exit_grid" / "exit_grid.json"
    if shared.exists():
        scores = json.loads(path.read_text()) if path.exists() else {}
        in_use = json.loads(shared.read_text()).get(exit_grid.IN_USE, {})
        if len(in_use) > len(scores.get(IN_USE, {})):
            scores[IN_USE] = in_use
            output.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(scores, indent=1))
    scores = score_designs(args.root, output, path.name, DESIGNS, args.samples)
    print(f"{len(scores)} designs in {output}")


if __name__ == "__main__":
    main()
