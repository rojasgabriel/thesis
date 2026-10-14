"""Score variants of the design of the V1 GLM on fixed sets of units.

The form of a term is chosen on sets of 20 units (`fit.analysis_unit_indices`:
`sample`, `sample1`, ...; no unit is in two sets). `score_design` builds the
design with some constants of `dmat` replaced, runs the model-selection stage
of `glm fit` (fit on the training trials, D² on the validation trials), and
returns the validation D² of each unit. `score_designs` does that for several
designs and several sets and keeps the scores in one JSON file, so the
comparison of two designs is paired over all those units. The grid modules
(`gap_class_grid.py` and the others) give the designs; this module has no
command.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from thesis.ephys.analyses.glm import cli, dmat, fit

UNIT_SET = "sample"
INPUTS = ("stimulus_windows_me.npz", "video_me_features.npz", "frame_times.npy")


def score_design(
    source: Path, work: Path, name: str, settings: dict, unit_set: str = UNIT_SET
) -> dict[str, float]:
    """Return each unit's validation D² with some `dmat` attributes replaced."""
    model = cli.PoissonGLM(root=work / name)
    model.root.mkdir(parents=True, exist_ok=True)
    for name in INPUTS:
        shutil.copy(source / name, model.root / name)
    # ponytail: the design reads module constants; set them for one design and
    # restore them. Make them arguments if the design gets more callers.
    saved = {name: getattr(dmat, name) for name in settings}
    for name, value in settings.items():
        setattr(dmat, name, value)
    try:
        dmat.prepare_common_design(model.windows, model.video, model.design)
        fit.fit_models(model.windows, model.design, unit_set, model.fit_dir(unit_set))
    finally:
        for name, value in saved.items():
            setattr(dmat, name, value)
    count = str(max(dmat.VIDEO_COMPONENT_COUNTS))
    scores = {}
    for path in (model.fit_dir(unit_set) / "checkpoints").glob(
        "unit_*_validation.json"
    ):
        record = json.loads(path.read_text())
        scores[str(record["unit_id"])] = record["plus_video"][count]["validation"][
            "deviance_explained"
        ]
    shutil.rmtree(model.root)  # one design is about 80 MB
    return scores


def score_designs(
    source: Path,
    output: Path,
    filename: str,
    designs: dict[str, dict],
    samples: int = 1,
) -> dict[str, dict]:
    """Score each design on `samples` sets of 20 units; return design -> unit -> D².

    The sets are `sample`, `sample1`, ... of `fit.analysis_unit_indices`: fixed
    sets of 20 units with no unit in two sets, so their scores go in one table
    and the comparison of two designs is paired over all of those units. A
    design with saved scores for a set is not fit again for it.
    """
    path = output / filename
    scores = json.loads(path.read_text()) if path.exists() else {}
    for index, (name, settings) in enumerate(designs.items()):
        for sample in range(samples):
            if len(scores.get(name, {})) >= fit.SAMPLE_UNITS * (sample + 1):
                continue
            unit_set = "sample" + (str(sample) if sample else "")
            scores.setdefault(name, {}).update(
                score_design(
                    source, output / "work", f"design_{index}", settings, unit_set
                )
            )
            output.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(scores, indent=1))
            print(f"Scored {name!r} on {unit_set}.", flush=True)
    return scores
