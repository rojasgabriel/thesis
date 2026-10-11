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
  center exit, with log-spaced bases, and no step.
- "linear\nkernel": the same kernel with linear spacing of the bases.

Units, alignment, baseline, and exclusions are those of `fit.py`. Units are the
fixed 20-unit sample. The sampling unit is the unit. `design_optimization.py`
draws the scores; this module has no figure.

    uv run python -m thesis.ephys.analyses.glm.hold_grid

A design that has saved scores in `hold_grid/hold_grid.json` is not fit again.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from thesis.ephys.analyses.glm import cli
from thesis.ephys.analyses.glm.adaptation_grid import score_design

STEP = {"HOLD_KERNEL": False, "IN_PORT_COLUMN": True}
DESIGNS = {
    "none": {"HOLD_KERNEL": False},
    "step,\nentry": STEP,
    "step,\nflash": {**STEP, "STATE_FROM_FIRST_FLASH": True},
    "step +\nonset": {**STEP, "STATE_FROM_FIRST_FLASH": True, "FIRST_FLASH_TERM": True},
    "kernel": {},
    # A stretch much longer than the window gives even spacing in time.
    "linear\nkernel": {"HOLD_STRETCH_S": 1000.0},
}
IN_USE = "kernel"


def run_grid(source: Path, output: Path) -> dict[str, dict]:
    """Score every design that has no saved scores; return all designs."""
    path = output / "hold_grid.json"
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
    scores = run_grid(args.root, args.root / "hold_grid")
    print(f"{len(scores)} designs in {args.root / 'hold_grid'}")


if __name__ == "__main__":
    main()
