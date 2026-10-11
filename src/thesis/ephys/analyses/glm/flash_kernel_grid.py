"""Score the window and the number of bases of the V1 GLM flash kernel.

Question: how long must the flash kernel be, how many raised cosines does it
need, and does the model need the second flash term (the flash kernel scaled by
a per-flash value) when the kernel is long enough to hold the suppression that
follows a flash?

Comparison: for each design on the grid (`BASES` x `WINDOWS_S`, with and
without the second flash term), build the full design and run the
model-selection stage of `glm fit`: fit on the training trials and score D² on
the validation trials. Nothing else in the model changes. The comparison is
paired: each unit is scored with each design.

Units, alignment, baseline, and exclusions are those of `fit.py`. Units are the
fixed 20-unit sample. The sampling unit is the unit.

`design_optimization.py` draws the scores; this module has no figure.

More bases can only fit the training trials better, so the rule is the smallest
kernel after which the validation D² is flat, not the maximum.

The script also scores the design in `dmat.py` with one more term, a separate
kernel for the first flash of each trial (`first_flash.json`).

    uv run python -m thesis.ephys.analyses.glm.flash_kernel_grid

A design that has saved scores in `flash_kernel_grid/flash_kernel_grid.json` is
not fit again.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from thesis.ephys.analyses.glm import cli
from thesis.ephys.analyses.glm.adaptation_grid import score_design

BASES = (6, 8, 10, 12)
WINDOWS_S = (0.1, 0.15, 0.2, 0.25, 0.35)


def run_grid(source: Path, output: Path) -> list[dict]:
    """Score every design that has no saved scores; return all designs."""
    path = output / "flash_kernel_grid.json"
    rows = json.loads(path.read_text()) if path.exists() else []
    done = {(row["bases"], row["window"], row["second"]) for row in rows}
    for second in (True, False):
        for window in WINDOWS_S:
            for bases in BASES:
                if (bases, window, second) in done:
                    continue
                settings = {
                    "FLASH_BASES": bases,
                    "FLASH_WINDOW_S": window,
                    "SECOND_FLASH_TERM": second,
                }
                rows.append(
                    {
                        "bases": bases,
                        "window": window,
                        "second": second,
                        "units": score_design(
                            source,
                            output / "work",
                            f"{bases}_{window}_{second}",
                            settings,
                        ),
                    }
                )
                output.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(rows, indent=1))
                print(f"Scored {settings}.", flush=True)
    return rows


def score_first_flash(source: Path, output: Path) -> dict[str, float]:
    """Score the design in `dmat.py` with a separate kernel for the first flash."""
    path = output / "first_flash.json"
    if not path.exists():
        scores = score_design(
            source, output / "work", "first_flash", {"FIRST_FLASH_TERM": True}
        )
        path.write_text(json.dumps(scores, indent=1))
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=cli.PoissonGLM().root)
    args = parser.parse_args()
    output = args.root / "flash_kernel_grid"
    rows = run_grid(args.root, output)
    score_first_flash(args.root, output)
    print(f"{len(rows)} designs and the first-flash design in {output}")


if __name__ == "__main__":
    main()
