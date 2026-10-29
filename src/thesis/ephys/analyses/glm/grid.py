"""Score variants of the design of the V1 GLM on fixed sets of units.

The parameterization of a term is chosen on sets of 20 units (`fit.analysis_unit_indices`:
`sample`, `sample1`, ...; no unit is in two sets). `score_design` builds the
design with some constants of `dmat` replaced, runs the model-selection stage
of `glm fit` (fit on the training trials, D² on the validation trials), and
returns the validation D² of each unit. `score_designs` does that for several
designs and several sets and keeps the scores in one JSON file, so the
comparison of two designs is paired over all those units. Saved scores are
reused only when the design signature (`design_signature`) is the same, so a
change of the GLM code, the inputs or the units refits them. The grid modules
(`gap_class_grid.py` and the others) give the designs; this module has no
command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from thesis.ephys.analyses.glm import cli, dmat, fit
from thesis.ephys.units import fetch_unit_table

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
    # ponytail: the design reads module constants; set them for one design.
    # Make them arguments if the design gets more callers.
    with dmat.switched(settings):
        dmat.prepare_common_design(model.windows, model.video, model.design)
        fit.fit_models(model.windows, model.design, unit_set, model.fit_dir(unit_set))
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


def design_signature(source: Path) -> str:
    """Return a hash of the GLM code, the inputs of `source` and its eligible units."""
    # ponytail: whole files, so a comment change in dmat.py or fit.py also
    # refits; hash only the design constants if that gets expensive.
    digest = hashlib.sha256()
    for path in (
        Path(dmat.__file__),
        Path(fit.__file__),
        *(source / n for n in INPUTS),
    ):
        digest.update(path.read_bytes())
    metadata = dmat._load_windows(source / INPUTS[0])["metadata"]
    units = fetch_unit_table(
        metadata["subject_name"],
        metadata["session_name"],
        unit_criteria_id=1,
        stability_param_id=0,
        include_metrics=False,
    )
    ids = units["unit_id"].to_numpy(dtype=int)[fit.analysis_unit_indices(units, "all")]
    digest.update(np.sort(ids).tobytes())
    return digest.hexdigest()


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
    design with saved scores for a set and the same design signature is not fit
    again for it.
    """
    path = output / filename
    scores = json.loads(path.read_text()) if path.exists() else {}
    # The signature of the design of each saved score; another one is refit.
    signature = design_signature(source)
    signature_path = output / f"{Path(filename).stem}_signature.json"
    signatures = (
        json.loads(signature_path.read_text()) if signature_path.exists() else {}
    )
    for index, (name, settings) in enumerate(designs.items()):
        if signatures.get(name) != signature and name in scores:
            print(f"Saved scores of {name!r} are of another design; fit again.")
            del scores[name]
        for sample in range(samples):
            if len(scores.get(name, {})) >= fit.SAMPLE_UNITS * (sample + 1):
                continue
            unit_set = "sample" + (str(sample) if sample else "")
            scores.setdefault(name, {}).update(
                score_design(
                    source, output / "work", f"design_{index}", settings, unit_set
                )
            )
            signatures[name] = signature
            output.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(scores, indent=1))
            signature_path.write_text(json.dumps(signatures, indent=1))
            print(f"Scored {name!r} on {unit_set}.", flush=True)
    return scores


def main(designs: dict[str, dict], name: str, description: str) -> None:
    """Score the designs of one grid module from the command line."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--root", type=Path, default=cli.PoissonGLM().root)
    parser.add_argument(
        "--samples",
        type=int,
        default=1,
        help="Number of sets of 20 units (no unit is in two sets).",
    )
    args = parser.parse_args()
    output = args.root / name
    scores = score_designs(args.root, output, f"{name}.json", designs, args.samples)
    print(f"{len(scores)} designs in {output}")
