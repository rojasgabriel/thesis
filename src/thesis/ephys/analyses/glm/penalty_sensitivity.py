"""Test whether the GLM results depend on the penalty of the fixation kernel.

Question: the model has one L2 penalty for all terms. With it, the fit makes
the fixation kernel too small (its level is recovered at 0.75 / 0.84 of its
size) and can give a part of the rate step at the exit to the flash × after
exit kernel. Do the results change when the penalty on the fixation kernel
alone is 100 times weaker?

Comparison: the design in use, fit again on all units of both mice with
`dmat.HOLD_PENALTY_SCALE` = `SCALE` (the fixation columns are `SCALE` times
larger than a z-score, so the penalty on their weights is `SCALE`² times
weaker; the saved column scale has the factor, so the kernels are read in the
same way). Every other setting, unit, trial, alignment and exclusion is that
of the model in use. Then, on this fit, the measures of the model in use with
their own modules:

- `flash_movement.py`: result 3 of `docs/glm.md` (observed, held-out full
  model and without flash × after exit; flash-excited units, normalized peaks).
- `leak_test.py`: the leak of flash × after exit in the measure of result 3,
  and the fixation level that a refit keeps of a simulated level.

The outputs are in `OUT` (one directory per session and the JSON files of
those modules), and `penalty_sensitivity.json` there compares the two fits:
median held-out D², the fixation level of the kernel, its recovery, and the
result-3 numbers. Units are the sampling unit; values are given per mouse.

    THESIS_CACHE_DIR=figures/cache nice -n 10 uv run python -m \\
        thesis.ephys.analyses.glm.penalty_sensitivity

A stage that is done is not run again (the fit keeps its checkpoints).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from thesis.ephys.analyses.glm import cli, dmat, flash_movement, grid, leak_test
from thesis.ephys.analyses.glm.fit import load_unit_records
from thesis.ephys.analyses.glm.flash_kernels import SESSIONS

MAIN = Path("results/glm")
OUT = MAIN / "weaker_fixation_penalty"
SCALE = 10.0
CLASSES = "raw_video_trust_classification.json"


def fit_session(session: str) -> None:
    """Fit all units of one session with the weaker penalty, in `OUT`."""
    model = cli.PoissonGLM(root=OUT / session)
    model.root.mkdir(parents=True, exist_ok=True)
    for name in grid.INPUTS:
        if not (model.root / name).exists():
            shutil.copy(MAIN / session / name, model.root / name)
    with dmat.switched({"HOLD_PENALTY_SCALE": SCALE}):
        if not model.design.exists():
            dmat.prepare_common_design(model.windows, model.video, model.design)
        checkpoints = model.fit_dir("all") / "checkpoints"
        if (
            len(list(checkpoints.glob("unit_*_final.json")))
            < len(list(checkpoints.glob("unit_*_validation.json")))
            or not checkpoints.exists()
        ):
            model.fit("all")


def summary(out: Path) -> dict:
    """Return the numbers of one fit (the model in use or the weaker penalty)."""
    result = {}
    movement = json.loads((out / "flash_movement.json").read_text())["sessions"]
    leaks = json.loads((out / "leak_test.json").read_text())
    comparison = leaks["main comparison (normalized), real minus leak"]
    for session in SESSIONS:
        subject = session.split("_")[0]
        model = cli.PoissonGLM(root=out / session)
        final = load_unit_records(
            model.fit_dir("all") / "checkpoints", "unit_*_final.json"
        )
        folds = load_unit_records(
            model.fit_dir("all") / "checkpoints", "unit_*_folds.json"
        )
        with np.load(out / session / "leak_test.npz") as saved:
            real_level = saved["real"][:, 2]
        long_gap = movement[subject]["normalized_rate_comparison"][
            flash_movement.GAP_LABELS[0]
        ]
        corrected = comparison[f"excited units, {flash_movement.GAP_LABELS[0]}"][
            "held out"
        ][subject]
        result[subject] = {
            "units": len(final),
            "median_cross_validated_d2": float(
                np.median([item["deviance_explained_mean"] for item in folds])
            ),
            "median_fixation_level_of_kernel": float(np.median(real_level)),
            "fixation_level_kept_by_a_refit": leaks[session]["fixation level"][
                "median_refit_over_simulated"
            ],
            "result_3_long_gap_mean_enhancement": {
                source: long_gap[source]["mean_difference"]
                for source in flash_movement.RATE_SOURCES
            },
            "flash_x_exit_contribution_minus_leak": {
                key: corrected[key]
                for key in (
                    "mean_real",
                    "mean_leak",
                    "mean_real_minus_leak",
                    "wilcoxon_p",
                )
            },
        }
    return result


def main() -> None:
    for session in SESSIONS:
        fit_session(session)
    if not (OUT / CLASSES).exists():
        shutil.copy(MAIN / CLASSES, OUT / CLASSES)
    if not (OUT / "flash_movement.json").exists():
        flash_movement.OUTPUT = OUT  # ponytail: the module reads its output root
        flash_movement.main()
    if not (OUT / "leak_test.json").exists():
        stats = {
            session: leak_test.simulate_session(OUT, session, "all", 6)
            for session in SESSIONS
        }
        stats["main comparison (normalized), real minus leak"] = (
            leak_test.comparison_statistics(OUT)
        )
        (OUT / "leak_test.json").write_text(json.dumps(stats, indent=1))
    report = {
        "hold_penalty_scale": SCALE,
        "model in use": summary(MAIN),
        "weaker fixation penalty": summary(OUT),
    }
    (OUT / "penalty_sensitivity.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
