"""Test 200 motion-energy PCs with a stronger penalty on the video columns.

Question: in `pc_grid.py`, 200 PCs predict worse than 10. Papers use about 200
video PCs, but with a penalty set for the video. Our model has one L2 penalty
for each unit, shared by all columns. Do 200 PCs predict better than 10 when
the video columns get a stronger penalty than the other terms?

Comparison: the design in use (`dmat.py`) with 0, 10, or 200 PCs. The 200-PC
design is also fit with the video columns multiplied by `SCALES` after they are
z-scored: a factor f makes the video penalty 1/f² times stronger (×100, ×1000).
The overall penalty is still chosen for each unit and each design on the
validation trials. Scores are validation D² from the model-selection stage of
`glm fit`, for each unit. The comparison with `IN_USE` (10 PCs) is paired over
units: median change of D², units better, two-sided Wilcoxon signed-rank test,
Holm correction over the other designs.

Units: the four fixed sets of 20 GRB006 units of the grids (`--sample 0..3`),
from the unit table in `THESIS_CACHE_DIR`. Run it with the unit table of
before the gain fix (`results/glm/_before_gain_fix_2026_10_24/cache`, with the
trial table) to use the same 80 units as `pc_grid.py`. Alignment, baseline,
and exclusions are those of `fit.py`. The sampling unit is the unit.

Each set runs in one process (the PC count is a module constant that worker
processes do not see), so run the four sets in parallel, then compare:

    nice -n 10 uv run python -m thesis.ephys.analyses.glm.pc_penalty_grid --sample 0
    uv run python -m thesis.ephys.analyses.glm.pc_penalty_grid --compare

Scores go to `pc_penalty_grid/sample_<n>.json`; the comparison to
`pc_penalty_grid/comparison.json`.
"""

from __future__ import annotations

import argparse
import json
import shutil
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.stats import wilcoxon

from thesis.ephys.analyses.glm import cli, dmat, fit
from thesis.ephys.analyses.glm.grid import INPUTS
from thesis.ephys.analyses.glm.inclusion import holm

SCALES = {"×100": 0.1, "×1000": 1000**-0.5}
IN_USE = "10 PCs"
NAME = "pc_penalty_grid"


def _score(source: Path, work: Path, unit_set: str, counts: tuple, scale: float):
    """Return {count: {unit: validation D²}}, count 0 included, for one design."""
    model = cli.PoissonGLM(root=work)
    model.root.mkdir(parents=True, exist_ok=True)
    for name in INPUTS:
        shutil.copy(source / name, model.root / name)
    with ExitStack() as stack:
        for module in (dmat, fit):
            stack.enter_context(patch.object(module, "VIDEO_COMPONENT_COUNTS", counts))
        dmat.prepare_common_design(model.windows, model.video, model.design)
        metadata = json.loads(model.design.with_suffix(".json").read_text())
        video = [
            i
            for i, n in enumerate(metadata["column_names"])
            if n.startswith("video_pc_")
        ]
        matrix = np.load(model.design, mmap_mode="r+")
        matrix[:, video[0] : video[-1] + 1] *= np.float32(scale)
        matrix.flush()
        del matrix
        fit.fit_models(
            model.windows, model.design, unit_set, model.fit_dir(unit_set), workers=1
        )
    scores = {0: {}, **{count: {} for count in counts}}
    for path in (model.fit_dir(unit_set) / "checkpoints").glob(
        "unit_*_validation.json"
    ):
        record = json.loads(path.read_text())
        unit = str(record["unit_id"])
        scores[0][unit] = record["baseline"]["validation"]["deviance_explained"]
        for count in counts:
            scores[count][unit] = record["plus_video"][str(count)]["validation"][
                "deviance_explained"
            ]
    shutil.rmtree(model.root)  # a design with 200 PCs is about 700 MB
    return scores


def run_sample(source: Path, output: Path, sample: int) -> None:
    unit_set = "sample" + (str(sample) if sample else "")
    path = output / f"sample_{sample}.json"
    scores = json.loads(path.read_text()) if path.exists() else {}
    work = output / f"work_{sample}"
    if IN_USE not in scores:
        plain = _score(source, work, unit_set, (10, 200), 1.0)
        scores.update({"0 PCs": plain[0], IN_USE: plain[10], "200 PCs": plain[200]})
        path.write_text(json.dumps(scores, indent=1))
    for label, scale in SCALES.items():
        name = f"200 PCs, video penalty {label}"
        if name not in scores:
            scores[name] = _score(source, work, unit_set, (200,), scale)[200]
            path.write_text(json.dumps(scores, indent=1))
        print(f"Scored {name} on {unit_set}.", flush=True)


def compare(output: Path) -> dict:
    scores: dict[str, dict] = {}
    for path in sorted(output.glob("sample_*.json")):
        for name, units in json.loads(path.read_text()).items():
            scores.setdefault(name, {}).update(units)
    units = sorted(set.intersection(*(set(s) for s in scores.values())))
    reference = np.asarray([scores[IN_USE][u] for u in units])
    changes = {
        name: np.asarray([scores[name][u] for u in units]) - reference
        for name in scores
        if name != IN_USE
    }
    p_values = {name: float(wilcoxon(c).pvalue) for name, c in changes.items()}
    adjusted = holm(p_values)
    comparison = {
        "units": len(units),
        "in_use": IN_USE,
        "in_use_median_d2": float(np.median(reference)),
        **{
            name: {
                "median_change": float(np.median(change)),
                "units_better": int(np.sum(change > 0)),
                "wilcoxon_p": p_values[name],
                "wilcoxon_p_holm": adjusted[name],
            }
            for name, change in changes.items()
        },
    }
    (output / "comparison.json").write_text(json.dumps(comparison, indent=1))
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=cli.PoissonGLM().root)
    parser.add_argument("--sample", type=int, help="Set of 20 units, 0 to 3.")
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()
    output = args.root / NAME
    output.mkdir(parents=True, exist_ok=True)
    if args.sample is not None:
        run_sample(args.root, output, args.sample)
    if args.compare:
        print(json.dumps(compare(output), indent=1))


if __name__ == "__main__":
    main()
