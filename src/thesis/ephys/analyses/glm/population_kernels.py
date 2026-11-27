"""What kernels does the GLM learn for the flash, the fixation and the exit?

Question: before the model is used for tests, what does it learn for the
terms that the talk is about?

Comparison: none; a description of the fitted model. No new fit: the kernels
are from the final refit of each unit (`fit.py`, unit set `all`) of each
session of `flash_kernels.SESSIONS`. Units of all mice are pooled (Gabriel,
2026-11-27); the curves are the mean ± SEM over units, in Δ log rate.

- Flash: the flash kernel in the port after a gap of 80 ms or more
  (`results.flash_kernel_at`, gap 100 ms, not after the exit), in the
  flash-excited units (`peaks.load_excited_units`).
- Fixation: the hold kernel from the first flash (`hold_effect.kernel_values`),
  all units.
- After the exit: the response-period kernel from the center exit
  (`hold_effect.kernel_values`), all units.
- Motion energy enters as video principal components, with no single time
  kernel, so it is not drawn here.

Writes `figures/glm/population_kernels.svg` and
`results/glm/population_kernels.json`.

    THESIS_CACHE_DIR=figures/cache uv run python -m \\
        thesis.ephys.analyses.glm.population_kernels
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thesis.ephys.analyses.glm import cli, dmat
from thesis.ephys.analyses.glm.fit import load_unit_records
from thesis.ephys.analyses.glm.flash_kernels import SESSIONS
from thesis.ephys.analyses.glm.hold_effect import AFTER, kernel_values
from thesis.ephys.analyses.glm.peaks import load_excited_units
from thesis.ephys.analyses.glm.results import flash_kernel_at
from thesis.plotting import (
    CONCEPT_COLORS,
    FIGURE_STYLE,
    REFERENCE_COLOR,
    label_panels,
    save_figure,
    unit_curves,
)

OUT = Path("results/glm")
FLASH_GAP_MS = 100  # a gap in the 80+ ms class


def flash_kernels(root: Path, excited: set[int]) -> tuple[np.ndarray, np.ndarray]:
    """Return the lags (ms) and the in-port flash kernel of the excited units."""
    model = cli.PoissonGLM(root=root)
    metadata = json.loads(model.design.with_suffix(".json").read_text())
    names = metadata["column_names"]
    scale = np.asarray(metadata["training_scale"])
    basis = dmat.task_temporal_bases()[dmat.FLASH_AFTER_EXIT]
    lags = 1000 * basis.basis_time
    columns = {
        term: [i for i, c in enumerate(names) if c.startswith(f"{term}_basis")]
        for term in (
            dmat.FLASH_COMMON,
            dmat.FLASH_JUST_BEFORE,
            dmat.FLASH_AFTER_EXIT,
            dmat.FLASH_RECOVERY,
        )
    }
    columns = {term: index for term, index in columns.items() if index}
    final = load_unit_records(model.fit_dir("all") / "checkpoints", "unit_*_final.json")
    curves = []
    for item in final:
        if int(item["unit_id"]) not in excited:
            continue
        weights = np.asarray(item["coefficients"], dtype=float)
        terms = {
            term: (lags, (weights[index] / scale[index]) @ basis.basis.T)
            for term, index in columns.items()
        }
        curves.append(
            flash_kernel_at(
                terms, FLASH_GAP_MS, False, metadata.get("flash_recovery_center")
            )
        )
    return lags, np.asarray(curves)


def main() -> None:
    excited = load_excited_units()
    flash, hold, after = [], [], []
    for session in SESSIONS:
        root = OUT / session
        lags, curves = flash_kernels(root, set(excited[session]))
        flash.append(curves)
        values = kernel_values(root, "all")
        hold.append(values[dmat.HOLD]["values"])
        after.append(values[AFTER]["values"])
        hold_t, after_t = values[dmat.HOLD]["times"], values[AFTER]["times"]
    panels = (
        (
            lags,
            flash,
            CONCEPT_COLORS["visual"],
            "flash, in port",
            "time from flash (ms)",
        ),
        (
            hold_t,
            hold,
            CONCEPT_COLORS["task"],
            "fixation",
            "time from the first flash (ms)",
        ),
        (
            after_t,
            after,
            CONCEPT_COLORS["task"],
            "after the exit",
            "time from the center exit (ms)",
        ),
    )
    report = {}
    with plt.rc_context(FIGURE_STYLE):
        figure, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
        for axis, (x, groups, color, name, xlabel) in zip(axes, panels, strict=True):
            mean = unit_curves(axis, x, groups, color)
            n = sum(len(g) for g in groups)
            axis.axhline(0, color=REFERENCE_COLOR, ls="--", lw=0.8)
            axis.set_xlabel(xlabel)
            axis.text(
                0.98,
                0.97,
                f"{name}\n{n} units, {len(groups)} mice",
                color=color,
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=8,
            )
            report[name] = {
                "units": n,
                "units_of_each_mouse": {
                    s.split("_")[0]: len(g)
                    for s, g in zip(SESSIONS, groups, strict=True)
                },
                "extreme_mean": float(mean[np.argmax(np.abs(mean))]),
            }
        axes[0].set_ylabel("kernel (Δ log rate)")
        label_panels(axes, x=-0.18, y=1.04)
        figure.tight_layout()
        print(save_figure(figure, OUT / "population_kernels"))
    (OUT / "population_kernels.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
