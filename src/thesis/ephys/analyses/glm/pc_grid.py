"""Test the number of motion-energy PCs in the final design of the V1 GLM.

Question: how many motion-energy PCs does the model need, with every other
regressor as in `dmat.py`?

Comparison: build the design with `COUNTS` PCs and run the model-selection
stage of `glm fit` once. It fits the model without motion energy and with the
first 10, 25, 50, 100, and 200 PCs on the training trials and scores D² on the
validation trials. The comparison is paired: each unit is scored with each
count.

Units, alignment, baseline, and exclusions are those of `fit.py`. Units are the
fixed 20-unit sample. The sampling unit is the unit. The script writes the
scores to `pc_grid/pc_grid.json`; `design_optimization.py` draws them.

    uv run python -m thesis.ephys.analyses.glm.pc_grid

The units are fit one after the other in this process (about 15 minutes),
because the PC counts are module constants that worker processes do not see.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from thesis.ephys.analyses.glm import cli, dmat, figures, fit
from thesis.ephys.analyses.glm.adaptation_grid import INPUTS, UNIT_SET

COUNTS = (10, 25, 50, 100, 200)


def run_grid(source: Path, output: Path) -> dict[str, dict]:
    """Return {PC count: {unit: validation D²}}; count "0" is no motion energy."""
    path = output / "pc_grid.json"
    if path.exists():
        return json.loads(path.read_text())
    model = cli.PoissonGLM(root=output / "work")
    model.root.mkdir(parents=True, exist_ok=True)
    for name in INPUTS:
        shutil.copy(source / name, model.root / name)
    modules = (dmat, fit, figures)
    saved = dmat.VIDEO_COMPONENT_COUNTS
    for module in modules:  # this run only; restored below
        module.VIDEO_COMPONENT_COUNTS = COUNTS
    try:
        dmat.prepare_common_design(model.windows, model.video, model.design)
        fit.fit_models(
            model.windows, model.design, UNIT_SET, model.fit_dir(UNIT_SET), workers=1
        )
    finally:
        for module in modules:
            module.VIDEO_COMPONENT_COUNTS = saved
    scores = {str(count): {} for count in (0, *COUNTS)}
    for record_path in (model.fit_dir(UNIT_SET) / "checkpoints").glob(
        "unit_*_validation.json"
    ):
        record = json.loads(record_path.read_text())
        unit = str(record["unit_id"])
        scores["0"][unit] = record["baseline"]["validation"]["deviance_explained"]
        for count in COUNTS:
            scores[str(count)][unit] = record["plus_video"][str(count)]["validation"][
                "deviance_explained"
            ]
    path.write_text(json.dumps(scores, indent=1))
    shutil.rmtree(model.root)  # the design with 200 PCs is about 700 MB
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=cli.PoissonGLM().root)
    args = parser.parse_args()
    scores = run_grid(args.root, args.root / "pc_grid")
    print(json.dumps({count: len(units) for count, units in scores.items()}))


if __name__ == "__main__":
    main()
