"""Draw the collinearity of the GLM terms.

Question: are the model terms so correlated that their weights cannot be
read? This script makes no new computation. It draws the statistics of
`collinearity.py` (see there for the definitions), from the design matrix of
each GLM session (`flash_kernels.SESSIONS`), on the valid bins of all trials.

- Panel a: the generalized variance inflation factor of each term,
  GVIF^(1/p) for a term of p columns (on the scale of a VIF): the factor by
  which the correlation with all the other terms multiplies the variance of
  its weights. One point per unit (the spike-history columns differ between
  units), mouse = marker. Dashed lines: the usual limits 5 and 10.
- Panels b, c: for each pair of terms, the first canonical correlation, the
  largest correlation between a weighted sum of the columns of one term and a
  weighted sum of the columns of the other (the worst case for that pair).
  Pairs with spike history: the median over units.
- The largest condition index of each unit is in the text of panel a
  (strong dependency above 30).

No unit is excluded beyond those of `fit.py`. There is no baseline or event
alignment; the sampling unit is the unit.

Writes `figures/glm/collinearity.svg`.

    uv run python -m thesis.ephys.analyses.glm.collinearity_figure
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thesis.ephys.analyses.glm.collinearity import VIF_LIMITS
from thesis.ephys.analyses.glm.flash_kernels import SESSIONS
from thesis.plotting import (
    CONCEPT_COLORS,
    FIGURE_STYLE,
    REFERENCE_COLOR,
    label_panels,
    mouse_marker,
    save_figure,
)

OUT = Path("results/glm")
VISUAL, TASK = CONCEPT_COLORS["visual"], CONCEPT_COLORS["task"]
# Key in the saved fits: (label, color), in the order of the deck.
TERMS = {
    "visual_flash": ("flash", VISUAL),
    "flash_just_before": ("short gap", VISUAL),
    "flash_after_exit": ("flash × after exit", VISUAL),
    "flash_recovery": ("recovery", VISUAL),
    "flash_side": ("flash × side", VISUAL),
    "fixation": ("pre-stimulus", TASK),
    "hold": ("fixation", TASK),
    "response_period": ("response period", TASK),
    "turn_direction": ("turn", TASK),
    "previous_side": ("previous side", TASK),
    "session_drift": ("drift", REFERENCE_COLOR),
    "video": ("video (10 PCs)", CONCEPT_COLORS["motion energy"]),
    "history": ("spike history", CONCEPT_COLORS["history"]),
}


def main() -> None:
    stats = json.loads((OUT / "collinearity.json").read_text())
    data = []
    for sess in SESSIONS:
        z = np.load(OUT / sess / "collinearity.npz", allow_pickle=False)
        order = [list(z["regressors"]).index(k) for k in TERMS]
        data.append(
            dict(
                subject=sess.split("_")[0],
                gvif=z["gvif"][:, order],
                shared=z["shared_directions"][np.ix_(order, order)],
                index=stats[sess]["largest_condition_index"],
            )
        )
    labels = [v[0] for v in TERMS.values()]
    colors = [v[1] for v in TERMS.values()]
    y = np.arange(len(TERMS))
    with plt.rc_context(FIGURE_STYLE):
        fig = plt.figure(figsize=(13, 4.6), layout="constrained")
        grid = fig.add_gridspec(1, 3, width_ratios=(0.8, 1, 1))
        a = fig.add_subplot(grid[0])
        rng = np.random.default_rng(0)
        for m, d in enumerate(data):
            for j in range(len(TERMS)):
                v = d["gvif"][:, j]
                off = (m - 0.5) * 0.36 + rng.uniform(-0.08, 0.08, len(v))
                a.scatter(
                    v,
                    j + off,
                    s=6,
                    marker=mouse_marker(d["subject"], m),
                    color=colors[j],
                    alpha=0.35,
                    lw=0,
                )
        for lim in VIF_LIMITS:
            a.axvline(lim, color="0.5", ls="--", lw=0.8)
            a.text(lim, -0.9, str(lim), ha="center", fontsize=7, color="0.4")
        a.set_yticks(y, labels)
        for t, c in zip(a.get_yticklabels(), colors, strict=True):
            t.set_color(c)
        a.invert_yaxis()
        a.set_xlim(0.8, 11)
        a.set_xscale("log")
        a.set_xticks([1, 2, 5, 10], ["1", "2", "5", "10"])
        a.set_xlabel("variance inflation, GVIF^(1/p)")
        a.text(
            1.0,
            -0.13,
            "\n".join(
                f"{'●' if m == 0 else '▲'} {d['subject']}: condition index ≤ "
                f"{d['index']['max']:.0f} (30 = strong)"
                for m, d in enumerate(data)
            ),
            transform=a.transAxes,
            ha="right",
            va="top",
            fontsize=7,
        )
        panels = [a]
        for m, d in enumerate(data):
            h = fig.add_subplot(grid[m + 1])
            # Lower triangle without the empty first row and last column.
            s = d["shared"].copy()
            s[np.triu_indices_from(s)] = np.nan
            s = s[1:, :-1]
            im = h.imshow(s, vmin=0, vmax=1, cmap="Greys")
            for i, j in zip(*np.tril_indices_from(s), strict=True):
                h.text(
                    j,
                    i,
                    f"{s[i, j]:.1f}".lstrip("0"),
                    ha="center",
                    va="center",
                    fontsize=5.5,
                    color="white" if s[i, j] > 0.6 else "black",
                )
            h.set_xticks(y[:-1], labels[:-1], rotation=60, ha="right", fontsize=7)
            h.set_yticks(y[:-1], labels[1:] if m == 0 else [], fontsize=7)
            for t, c in zip(h.get_xticklabels(), colors[:-1], strict=True):
                t.set_color(c)
            for t, c in zip(h.get_yticklabels(), colors[1:], strict=False):
                t.set_color(c)
            h.text(
                0.98,
                0.98,
                d["subject"],
                transform=h.transAxes,
                ha="right",
                va="top",
                fontsize=9,
            )
            for side in ("top", "right"):
                h.spines[side].set_visible(False)
            panels.append(h)
        fig.colorbar(
            im,
            ax=panels[1:],
            shrink=0.7,
            label="largest correlation between the two terms",
        )
        label_panels(panels, x=-0.05, y=1.04)
        print(save_figure(fig, OUT / "collinearity"))


if __name__ == "__main__":
    main()
