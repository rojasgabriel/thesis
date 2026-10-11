"""Test the withdrawal kernel and the in-port column of the V1 GLM.

Question: how wide must the withdrawal kernel be, and does the model need the
state column for the time in the center port?

Comparison: for each design in `DESIGNS`, build the full design and run the
model-selection stage of `glm fit`: fit on the training trials and score D² on
the validation trials. Nothing else in the model changes. The comparison is
paired: each unit is scored with each design.

- "±300", "±100", "0–100": a withdrawal kernel of that range in ms around the
  center exit, and no in-port column.
- "0–100\n+ in port": the design in `dmat.py`.
- "in port\nonly": the in-port column and no withdrawal kernel.

The script also fits the design in `dmat.py` without the in-port column on the
same units (`glm fit`, with cross-validation and the final refit), in the
directory with the suffix `NO_PORT_SUFFIX` next to the session directory, and
runs `residual_checks` on it. `design_optimization.py` uses its predicted
rates.

Units, alignment, baseline, and exclusions are those of `fit.py`. Units are the
fixed 20-unit sample. The sampling unit is the unit. The figure shows the mean
validation D² over units for each design; the dotted circle is the design in
`dmat.py`.

    uv run python -m thesis.ephys.analyses.glm.withdrawal_grid

A design that has saved scores in `withdrawal_grid/withdrawal_grid.json` is not
fit again.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from thesis.ephys.analyses.glm import cli, dmat, residual_checks
from thesis.ephys.analyses.glm.adaptation_grid import INPUTS, UNIT_SET, score_design
from thesis.plotting import (
    CONCEPT_COLORS,
    FIGURE_STYLE,
    label_panels,
    mark_selected,
    save_figure,
)

WIDE = {"WITHDRAWAL_PRE_S": 0.3, "WITHDRAWAL_WINDOW_S": 0.3, "WITHDRAWAL_BASES": 9}
DESIGNS = {
    "±300": {**WIDE, "IN_PORT_COLUMN": False},
    "±100": {"WITHDRAWAL_PRE_S": 0.1, "IN_PORT_COLUMN": False},
    "0–100": {"IN_PORT_COLUMN": False},
    "0–100\n+ in port": {},
    "in port\nonly": {"WITHDRAWAL_KERNEL": False},
}
IN_USE = "0–100\n+ in port"
NO_PORT_SUFFIX = "_no_in_port"


def run_grid(source: Path, output: Path) -> dict[str, dict]:
    """Score every design that has no saved scores; return all designs."""
    path = output / "withdrawal_grid.json"
    scores = json.loads(path.read_text()) if path.exists() else {}
    for index, (name, settings) in enumerate(DESIGNS.items()):
        if name in scores:
            continue
        scores[name] = score_design(
            source, output / "work", f"design_{index}", settings
        )
        output.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(scores, indent=1))
        print(f"Scored {name!r}.", flush=True)
    return scores


def fit_without_in_port(source: Path) -> Path:
    """Fit the design in use without the in-port column; run its residual checks."""
    root = source.with_name(source.name + NO_PORT_SUFFIX)
    model = cli.PoissonGLM(root=root)
    if (model.fit_dir(UNIT_SET) / "residual_checks.json").exists():
        return root
    root.mkdir(exist_ok=True)
    for name in INPUTS:
        shutil.copy(source / name, root / name)
    dmat.IN_PORT_COLUMN = False  # this fit only; restored below
    try:
        if not model.design.exists():
            dmat.prepare_common_design(model.windows, model.video, model.design)
        model.fit(UNIT_SET)
        data = residual_checks.run_checks(root, UNIT_SET)
    finally:
        dmat.IN_PORT_COLUMN = True
    residual_checks.save(data, model.fit_dir(UNIT_SET))
    return root


def plot_grid(scores: dict[str, dict], output: Path | None, axes=None) -> Path | None:
    """Plot mean validation D² for the kernel widths and for the in-port column.

    With `axes`, draw into them and do not label or save (for a combined figure).
    """
    mean = {
        name: float(np.mean(list(units.values()))) for name, units in scores.items()
    }
    color = CONCEPT_COLORS["task"]
    widths = ("±300", "±100", "0–100")
    port = ("0–100", "in port\nonly", IN_USE)
    with plt.rc_context(FIGURE_STYLE):
        figure = None
        if axes is None:
            figure, axes = plt.subplots(1, 2, figsize=(5.2, 2.6), layout="constrained")
        axes[0].plot(
            range(3), [mean[name] for name in widths], color=color, marker="o", mfc="w"
        )
        axes[0].set_xticks(range(3), widths)
        axes[0].set_xlabel("withdrawal kernel (ms), no in-port column")
        axes[1].plot(
            range(3),
            [mean[name] for name in port],
            color=color,
            marker="o",
            mfc="w",
            linestyle="none",
        )
        axes[1].axhline(mean["±300"], color=color, linestyle="--", linewidth=0.8)
        axes[1].text(
            -0.4, mean["±300"], "±300 ms kernel", color=color, va="bottom", fontsize=7
        )
        mark_selected(axes[1], port.index(IN_USE), mean[IN_USE])
        axes[1].set_xticks(range(3), port, fontsize=7)
        axes[1].set_xlabel("withdrawal terms")
        for axis in axes:
            axis.set_xlim(-0.5, 2.5)
            axis.margins(y=0.18)
            axis.set_ylabel(r"mean validation $D^2$")
        if figure is None:
            return None
        label_panels(axes, -0.3)
        return save_figure(figure, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=cli.PoissonGLM().root)
    args = parser.parse_args()
    output = args.root / "withdrawal_grid"
    scores = run_grid(args.root, output)
    print(plot_grid(scores, output / "withdrawal_grid"))
    print(fit_without_in_port(args.root))


if __name__ == "__main__":
    main()
