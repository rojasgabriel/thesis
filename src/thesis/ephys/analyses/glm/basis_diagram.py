"""Draw the basis functions of each kernel of the GLM (supplemental diagram).

Question: what shapes can each kernel of the model have? A kernel is a weighted
sum of basis functions, and the fit gives one weight for each function. The
diagram shows the functions that `dmat.py` makes, with no data and no fit, so
there is no comparison, baseline, exclusion, or sampling unit.

- a, flash kernels (each gap class and flash × move): time from the flash.
- b, fixation: time from the first flash. After the last
  function the kernel keeps its last value until the center exit.
- c, response period and turn direction (the same functions): time from the
  center exit. After the last function the kernel keeps its last value until
  the response entry.
- d, spike history: time before the bin, from 1 ms.
- e, session drift: time in the session, as a fraction. The last function is
  not a column, because the model has an intercept.

The pre-stimulus column, the previous side, and each motion-energy PC are one
column each, with no basis.

    uv run python -m thesis.ephys.analyses.glm.basis_diagram
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from thesis.ephys.analyses.glm import dmat
from thesis.ephys.analyses.glm.main_figure import DRIFT_COLOR, HISTORY, TASK, VISUAL
from thesis.plotting import FIGURE_STYLE, label_panels, save_figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("figures/glm"))
    args = parser.parse_args()
    task = dmat.task_temporal_bases()
    flash, hold = task[dmat.flash_gap_names()[0]], task["hold"]
    exit, history = dmat.exit_basis(), dmat.spike_history_basis()
    session = np.linspace(0, 1, 500)
    # The function that is left out is 1 minus the sum of the columns.
    drift = dmat.session_drift_columns(session)
    drift = np.column_stack([drift, 1 - drift.sum(1)])
    panels = [
        ("flash kernels", 1000 * flash.basis_time, flash.basis, VISUAL, "time from flash (ms)"),
        ("fixation", 1000 * hold.basis_time, hold.basis, TASK, "time from first flash (ms)"),
        ("response period, turn direction", 1000 * exit.basis_time, exit.basis, TASK, "time from center exit (ms)"),
        ("spike history", 1000 * history.basis_time + 1, history.basis, HISTORY, "time before the bin (ms)"),
        ("session drift", session, drift, DRIFT_COLOR, "time in session (fraction)"),
    ]  # fmt: skip
    with plt.rc_context(FIGURE_STYLE):
        figure, axes = plt.subplots(1, 5, figsize=(13, 2.3), layout="constrained")
        for ax, (name, time, basis, color, xlabel) in zip(axes, panels):
            basis = basis / basis.max()
            columns = basis.shape[1] - (name == "session drift")
            ax.plot(time, basis[:, :columns], color=color, lw=1)
            ax.plot(time, basis[:, columns:], color=color, lw=1, ls=":")
            ax.set_title(f"{name}\n{columns} functions", color=color, fontsize=9)
            ax.set_xlabel(xlabel)
            ax.set_yticks([0, 1])
        axes[0].set_ylabel("basis function\n(peak = 1)")
        label_panels(axes, y=1.25)
        print(save_figure(figure, args.out / "basis_functions"))


if __name__ == "__main__":
    main()
