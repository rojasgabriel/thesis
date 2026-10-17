"""Test how the V1 GLM codes the gap from the flash before.

Question: the response to a flash depends on the gap from the flash before.
Is one flash kernel for each gap class as good as the earlier terms, and how
many classes does the model need?

Comparison: the final design with one thing changed, on the fixed 20-unit
sample, with the validation D² of each unit. The designs are in `DESIGNS`:

- "1 kernel": one flash kernel for all flashes, with no term for the gap.
- "1 kernel +\nflash before": the design in use (`IN_USE`, from 2026-10-17):
  one common kernel on every flash and a second kernel on the flashes with a
  flash up to 50 ms before them (`dmat.FLASH_JUST_BEFORE_TERM`).
- "40 | 80+": two gap classes, with a limit at 50 ms (the design to
  2026-10-16). It spans the same columns as the design in use; only the L2
  penalty differs. (An earlier form also had a flash-recovery term on the log
  of the gap; the two terms were correlated, -0.86, and it is no longer in the
  code.)
- "40 | 80-120 | 160+": three classes.
- "40 | 80 | 120+": three classes with the second limit at 90 ms.
- "40 | 80 | 120 | 160+": four classes.
- "centered\nexit term": the flash × move values centered over the flashes
  (the coding to 2026-10-16), against 0 in the port and 1 after the exit.

In a design with classes, each class has its own flash kernel and there is no
common kernel; the first flash of a trial is in the last class. The flash ×
fixation term is in every design. The comparison is paired over units. Where
two options do not differ reliably, the model uses the simpler one. Units,
trials, alignment, baseline, and exclusions are those of `fit.py`.

    uv run python -m thesis.ephys.analyses.glm.gap_class_grid

A design that has saved scores in `gap_class_grid/gap_class_grid.json` is not
fit again. `--samples N` scores each design on N sets of 20 units
(no unit is in two sets); the comparison is then paired over all those units. Each design takes about two minutes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from thesis.ephys.analyses.glm import cli
from thesis.ephys.analyses.glm.grid import score_designs

# With no gap classes there is no flash × move × gap term: it is the product of
# the period and the class of the shortest gaps.
NO_CLASSES = {
    "FLASH_GAP_EDGES_S": (),
    "FLASH_MOVE_GAP_TERM": False,
    "FLASH_JUST_BEFORE_TERM": False,
}
# The class designs have no flash-just-before kernel: it would code the gap twice.
CLASSES = {"FLASH_JUST_BEFORE_TERM": False}
DESIGNS = {
    "1 kernel": NO_CLASSES,
    "1 kernel +\nflash before": {**NO_CLASSES, "FLASH_JUST_BEFORE_TERM": True},
    "40 | 80+": {**CLASSES, "FLASH_GAP_EDGES_S": (0.05,)},
    "40 | 80-120 | 160+": {**CLASSES, "FLASH_GAP_EDGES_S": (0.05, 0.13)},
    "40 | 80 | 120+": {**CLASSES, "FLASH_GAP_EDGES_S": (0.05, 0.09)},
    "40 | 80 | 120 | 160+": {**CLASSES, "FLASH_GAP_EDGES_S": (0.05, 0.09, 0.13)},
    # The coding of flash × move of the design to 2026-10-16 (centered).
    "centered\nexit term": {"FLASH_AFTER_EXIT_CENTERED": True},
}
IN_USE = "1 kernel +\nflash before"


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
        args.root,
        args.root / "gap_class_grid",
        "gap_class_grid.json",
        DESIGNS,
        args.samples,
    )
    print(f"{len(scores)} designs in {args.root / 'gap_class_grid'}")


if __name__ == "__main__":
    main()
