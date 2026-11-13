"""Draw how one term of the GLM is built from basis functions (teaching schematic).

Question: how does a weighted sum of basis functions become a term of the
model? An illustration with made-up weights, so there is no data, no fit,
no comparison, baseline, exclusion, or sampling unit. The real basis
functions of each kernel are in `basis_diagram.py`.

- a: a kernel = Σ wᵢ · basisᵢ, eight raised cosines over 0 to 150 ms.
- b: the design matrix: each basis function placed at every flash of an
  example train, one column per function.
- c: the term = columns × weights, the kernel at every flash, summed.

Writes `figures/glm/basis_schematic.svg`.

    uv run python -m thesis.ephys.analyses.glm.basis_schematic
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thesis.plotting import CONCEPT_COLORS, FIGURE_STYLE, label_panels, save_figure

OUT = Path("results/glm")
KERNEL_MS = 150
COUNT = 8
WEIGHTS = np.array([0.0, 0.15, 0.85, 1.0, 0.5, 0.05, -0.15, 0.0])  # made up
TRIAL_MS = 600
FLASHES_MS = (0, 80, 120, 280, 440)


def bump(x, center, width):
    return (
        0.5 * (1 + np.cos(np.pi * (x - center) / width)) * (np.abs(x - center) < width)
    )


def main() -> None:
    color = CONCEPT_COLORS["visual"]
    t = np.arange(KERNEL_MS + 1)
    centers = np.linspace(0, KERNEL_MS, COUNT)
    basis = np.array([bump(t, c, centers[1] - centers[0]) for c in centers])
    kernel = WEIGHTS @ basis
    time = np.arange(TRIAL_MS)
    train = np.zeros(TRIAL_MS)
    train[list(FLASHES_MS)] = 1
    columns = np.array([np.convolve(train, b)[:TRIAL_MS] for b in basis])
    with plt.rc_context(FIGURE_STYLE):
        fig = plt.figure(figsize=(12, 3.6))
        gs = fig.add_gridspec(2, 3, width_ratios=(1, 1.6, 1.6), hspace=0.5, wspace=0.3)
        a = fig.add_subplot(gs[:, 0])
        for w, b in zip(WEIGHTS, basis, strict=True):
            a.plot(t, w * b, color=color, alpha=0.45, lw=1)
        a.plot(t, kernel, color="black", lw=2)
        a.set_xlabel("time from flash (ms)")
        a.set_ylabel("Δ log rate")
        a.text(
            0,
            1.03,
            f"kernel = Σ wᵢ · basisᵢ ({COUNT} functions)",
            transform=a.transAxes,
            fontsize=8,
        )
        b = fig.add_subplot(gs[0, 1:])
        for i, column in enumerate(columns):
            b.plot(time, column * 0.8 + i, color=color, lw=0.9)
        for f in FLASHES_MS:
            b.axvline(f, color="0.5", lw=0.6, ls=":")
        b.set_yticks([0, COUNT - 1], ["column 1", f"column {COUNT}"])
        b.tick_params(labelbottom=False)
        b.text(
            0,
            1.04,
            "design matrix: each basis function placed at every flash "
            "(one column per function)",
            transform=b.transAxes,
            fontsize=8,
        )
        c = fig.add_subplot(gs[1, 1:], sharex=b)
        c.plot(time, WEIGHTS @ columns, color="black", lw=1.6)
        for f in FLASHES_MS:
            c.axvline(f, color="0.5", lw=0.6, ls=":")
        c.set_xlabel("time in trial (ms); dotted lines: flashes")
        c.set_ylabel("Δ log rate")
        c.text(
            0,
            1.04,
            "term = columns × weights: the kernel at every flash, summed; "
            "other terms add the same way",
            transform=c.transAxes,
            fontsize=8,
        )
        label_panels([a, b, c], x=-0.08, y=1.12)
        print(save_figure(fig, OUT / "basis_schematic"))


if __name__ == "__main__":
    main()
