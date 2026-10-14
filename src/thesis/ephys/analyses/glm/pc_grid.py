"""Test the number of motion-energy PCs in the final design of the V1 GLM.

Question: how many motion-energy PCs does the model need, with every other
regressor as in `dmat.py`?

Comparison: build the design with `COUNTS` PCs and run the model-selection
stage of `glm fit` once. It fits the model without motion energy and with the
first 10, 25, 50, 100, and 200 PCs on the training trials and scores D² on the
validation trials. The comparison is paired: each unit is scored with each
count.

Units, alignment, baseline, and exclusions are those of `fit.py`. Units are the
sets of 20 units of `grid.py` (`--samples`). The sampling unit is the unit. The script writes the
scores to `pc_grid/pc_grid.json`; `design_optimization.py` draws them.

    uv run python -m thesis.ephys.analyses.glm.pc_grid

The units are fit one after the other in this process (about 15 minutes for
each set of units),
because the PC counts are module constants that worker processes do not see.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from thesis.ephys.analyses.glm import cli, dmat, figures, fit
from thesis.ephys.analyses.glm.grid import INPUTS

COUNTS = (10, 25, 50, 100, 200)
# The names of the scores, for `design_optimization.py`: "0" is no motion energy.
DESIGNS = tuple(str(count) for count in (0, *COUNTS))
IN_USE = "10"


def run_grid(source: Path, output: Path, samples: int) -> dict[str, dict]:
    """Return {PC count: {unit: validation D²}}; count "0" is no motion energy."""
    path = output / "pc_grid.json"
    scores = json.loads(path.read_text()) if path.exists() else {}
    scores = {str(count): scores.get(str(count), {}) for count in (0, *COUNTS)}
    modules = (dmat, fit, figures)
    saved = dmat.VIDEO_COMPONENT_COUNTS
    for sample in range(samples):
        if len(scores["0"]) >= fit.SAMPLE_UNITS * (sample + 1):
            continue
        unit_set = "sample" + (str(sample) if sample else "")
        model = cli.PoissonGLM(root=output / "work")
        model.root.mkdir(parents=True, exist_ok=True)
        for name in INPUTS:
            shutil.copy(source / name, model.root / name)
        for module in modules:  # this run only; restored below
            module.VIDEO_COMPONENT_COUNTS = COUNTS
        try:
            dmat.prepare_common_design(model.windows, model.video, model.design)
            fit.fit_models(
                model.windows,
                model.design,
                unit_set,
                model.fit_dir(unit_set),
                workers=1,
            )
        finally:
            for module in modules:
                module.VIDEO_COMPONENT_COUNTS = saved
        for record_path in (model.fit_dir(unit_set) / "checkpoints").glob(
            "unit_*_validation.json"
        ):
            record = json.loads(record_path.read_text())
            unit = str(record["unit_id"])
            scores["0"][unit] = record["baseline"]["validation"]["deviance_explained"]
            for count in COUNTS:
                scores[str(count)][unit] = record["plus_video"][str(count)][
                    "validation"
                ]["deviance_explained"]
        path.write_text(json.dumps(scores, indent=1))
        shutil.rmtree(model.root)  # the design with 200 PCs is about 700 MB
        print(f"Scored {unit_set}.", flush=True)
    return scores


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
    scores = run_grid(args.root, args.root / "pc_grid", args.samples)
    print(json.dumps({count: len(units) for count, units in scores.items()}))


if __name__ == "__main__":
    main()
