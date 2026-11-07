"""Do the movement-related changes of the GLM depend on recording depth?

Question: the GLM finds a fixation drop, a rate change after the exit, a
flash response that changes after the exit, and one that differs with the
side of the turn. Do these change with the depth of the unit on the probe?
Depth stands in for the cortical layer (there is no histology of layers).

Comparison: for every GLM unit, its kernel values from the final refit,
against its depth (`depth` of the unique-deviance records, µm along the
probe as stored by the spike sorting). No new fit.

- Measures (Δ log rate): the fixation level (fixation kernel, 150-500 ms
  from the first flash) and the after-exit level (response-period kernel,
  150-500 ms from the exit), both from `hold_effect.kernel_values`; the flash
  kernel, the flash × after exit kernel and the flash × side kernel (+ = a
  larger flash response for a right turn), each the mean from `RESPONSE_MS`.
- Units: all GLM units with a final fit (172 / 79), each mouse alone: the
  two probes are not at the same place, so depth is not comparable between
  them.
- Statistics: per mouse, the Spearman correlation of each measure with
  depth, Holm correction over the measures of that mouse. The sampling unit
  is the unit.
- Display: the axis of each measure is the same in both mice; points outside
  it are drawn at its edge (open markers), stay in the statistics, and their
  number is in the figure.

The script writes `figures/glm/depth_effects.svg` and
`results/glm/depth_effects.json`.

    uv run python -m thesis.ephys.analyses.glm.depth_effects
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thesis.ephys.analyses.glm import cli, dmat, hold_effect
from thesis.ephys.analyses.glm.fit import load_unit_records
from thesis.ephys.analyses.glm.flash_kernels import SESSIONS
from thesis.ephys.analyses.glm.inclusion import holm
from thesis.plotting import (
    CONCEPT_COLORS,
    FIGURE_STYLE,
    MOUSE_MARKERS,
    label_panels,
    save_figure,
)

OUT = Path("results/glm")
RESPONSE_MS = (40, 100)
MEASURES = {
    "fixation level": CONCEPT_COLORS["task"],
    "after-exit level": CONCEPT_COLORS["task"],
    "flash": CONCEPT_COLORS["visual"],
    "flash × after exit": CONCEPT_COLORS["visual"],
    "flash × side": CONCEPT_COLORS["visual"],
}
TERMS = {
    "flash": dmat.FLASH_COMMON,
    "flash × after exit": dmat.FLASH_AFTER_EXIT,
    "flash × side": dmat.FLASH_SIDE,
}
LIMIT_PERCENTILES = (1, 99)


def session_measures(session: str) -> dict:
    """Return each unit's depth and kernel measures of one session."""
    root = OUT / session
    model = cli.PoissonGLM(root=root)
    metadata = json.loads(model.design.with_suffix(".json").read_text())
    names, scale = metadata["column_names"], np.asarray(metadata["training_scale"])
    checkpoints = model.fit_dir("all") / "checkpoints"
    final = load_unit_records(checkpoints, "unit_*_final.json")
    depth = {
        r["unit_id"]: r["depth"]
        for r in load_unit_records(checkpoints / "unique", "unit_*.json")
    }
    ids = [r["unit_id"] for r in final]
    kernels = hold_effect.kernel_values(root, "all")
    if list(kernels["unit_ids"]) != ids:
        raise ValueError(f"The unit order of the fits differs in {session}.")
    out = {
        "unit_ids": ids,
        "depth": np.asarray([depth[i] for i in ids], dtype=float),
        "fixation level": kernels[dmat.HOLD]["level"],
        "after-exit level": kernels[hold_effect.AFTER]["level"],
    }
    basis = dmat.task_temporal_bases()[dmat.FLASH_COMMON]
    lags = 1000 * basis.basis_time
    window = (lags >= RESPONSE_MS[0]) & (lags < RESPONSE_MS[1])
    for name, term in TERMS.items():
        columns = [i for i, n in enumerate(names) if n.startswith(f"{term}_basis")]
        out[name] = np.asarray(
            [
                (
                    (np.asarray(r["coefficients"])[columns] / scale[columns])
                    @ basis.basis.T
                )[window].mean()
                for r in final
            ]
        )
    return out


def main() -> None:
    sessions = {s.split("_")[0]: session_measures(s) for s in SESSIONS}
    report = {}
    for subject, data in sessions.items():
        tests = {m: spearmanr(data["depth"], data[m]) for m in MEASURES}
        adjusted = holm({m: float(t.pvalue) for m, t in tests.items()})
        report[subject] = {
            "units": len(data["unit_ids"]),
            "depth_range_um": [float(data["depth"].min()), float(data["depth"].max())],
            **{
                m: {
                    "spearman_rho": float(t.statistic),
                    "p": float(t.pvalue),
                    "p_holm": float(p),
                }
                for (m, t), p in ((item, adjusted[item[0]]) for item in tests.items())
            },
        }
    (OUT / "depth_effects.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    with plt.rc_context(FIGURE_STYLE):
        figure, axes = plt.subplots(
            len(sessions),
            len(MEASURES),
            figsize=(2.3 * len(MEASURES), 2.6 * len(sessions)),
            squeeze=False,
        )
        for col, (measure, color) in enumerate(MEASURES.items()):
            every = np.concatenate([d[measure] for d in sessions.values()])
            limits = np.percentile(every, LIMIT_PERCENTILES)
            pad = 0.05 * (limits[1] - limits[0])
            limits = (limits[0] - pad, limits[1] + pad)
            for row, (subject, data) in enumerate(sessions.items()):
                axis = axes[row, col]
                values = data[measure]
                inside = (values >= limits[0]) & (values <= limits[1])
                marker = MOUSE_MARKERS.get(subject, "o")
                axis.scatter(
                    values[inside],
                    data["depth"][inside],
                    s=7,
                    color=color,
                    alpha=0.6,
                    lw=0,
                    marker=marker,
                )
                axis.scatter(
                    np.clip(values[~inside], *limits),
                    data["depth"][~inside],
                    s=10,
                    facecolor="none",
                    edgecolor=color,
                    lw=0.6,
                    marker=marker,
                )
                axis.axvline(0, color="0.7", lw=0.6)
                axis.set_xlim(limits)
                stats = report[subject][measure]
                axis.set_title(
                    f"ρ = {stats['spearman_rho']:+.2f}, Holm p = {stats['p_holm']:.2g}"
                    + (
                        f"\n{int((~inside).sum())} at the axis edge"
                        if (~inside).any()
                        else ""
                    ),
                    fontsize=6,
                )
                if col == 0:
                    axis.set_ylabel(f"{subject}: depth (µm)")
                if row == len(sessions) - 1:
                    axis.set_xlabel(f"{measure}\n(Δ log rate)")
        figure.tight_layout()
        label_panels(axes.ravel())
        print(save_figure(figure, OUT / "depth_effects"))


if __name__ == "__main__":
    main()
