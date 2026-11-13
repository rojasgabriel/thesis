"""Show how much the flash × after exit term adds to each flash-excited unit.

Question: the flash peak is larger after the center exit. Does the model need
a flash-specific change after the exit (the flash × after exit term), once
its bias with no such effect (the leak) is taken out?

Comparison: per unit, the held-out flash × after exit part minus its held-out
leak, as a log ratio: log(after exit ÷ in port) of the model peak, with the
term minus without it (`leak_test.comparison_values`, "log real (held out)"
minus "log leak (held out)"). 0 (×1) = the term adds nothing. Paired: each
unit has its real part and its leak.

- Units: the flash-excited units of `flash_excited_units.json`. A unit with a
  model peak ≤ 0 has no log ratio and is left out; the figure gives the
  number of units kept.
- Flashes: the paired last in-port and first after-exit flashes of the main
  comparison (`flash_movement.py`), long gaps (80 ms or more, gap index 0).
- Statistics: Wilcoxon signed-rank against 0 within each mouse; both mice:
  the hierarchical bootstrap of `leak_test.hierarchical_interval` (mice, then
  units). The sampling unit is the unit.
- Display limit: ×1/4.5 to ×4.5; units past it stay in the statistics and the
  figure gives their number.

No new fit: the values come from the saved `leak_test.npz`,
`flash_movement.json` and the held-out simulations of `power_test.py`. The
full test with the final-refit leak is in `model_trust.py` (panels g to i).

Writes `figures/glm/flash_exit_effect.svg` and `results/glm/flash_exit_effect.json`.

    THESIS_CACHE_DIR=figures/cache uv run python -m \\
        thesis.ephys.analyses.glm.flash_exit_effect
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy.stats import wilcoxon

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thesis.ephys.analyses.glm import leak_test
from thesis.plotting import FIGURE_STYLE, OBSERVED_COLOR, mouse_marker, save_figure

OUT = Path("results/glm")
GAP = 0  # long gaps
LIMIT = np.log(4.5)
SEED = 0


def main() -> None:
    values = leak_test.comparison_values(OUT)
    rng = np.random.default_rng(SEED)
    effects, report = {}, {}
    with plt.rc_context(FIGURE_STYLE):
        fig, ax = plt.subplots(figsize=(6.2, 3.6))
        for j, (subject, d) in enumerate(values.items()):
            keep = d["excited"]
            r = (
                d["log real (held out)"][keep, GAP]
                - d["log leak (held out)"][keep, GAP]
            )
            left_out = int(np.sum(~np.isfinite(r)))
            r = r[np.isfinite(r)]
            effects[subject] = r
            ax.scatter(
                j + rng.uniform(-0.12, 0.12, len(r)),
                np.clip(r, -LIMIT, LIMIT),
                s=16,
                color=OBSERVED_COLOR,
                alpha=0.5,
                lw=0,
            )
            med, p = np.median(r), wilcoxon(r).pvalue
            ax.plot(
                j,
                med,
                mouse_marker(subject, j),
                color="white",
                mec="black",
                ms=11,
                mew=2,
                zorder=5,
            )
            ax.text(
                j + 0.22,
                med,
                f"median ×{np.exp(med):.2f}\np = {p:.2g}, {len(r)} units",
                va="center",
                fontsize=8,
            )
            report[subject] = dict(
                units=len(r),
                left_out=left_out,
                median_ratio=float(np.exp(med)),
                p=float(p),
            )
        pooled = leak_test.hierarchical_interval(list(effects.values()))
        m, (lo, hi) = pooled["median"], pooled["interval_95"]
        ax.errorbar(
            2, m, yerr=[[m - lo], [hi - m]], fmt="o", color="black", ms=8, capsize=4
        )
        ax.text(
            2.18,
            m,
            f"pooled ×{np.exp(m):.2f}\n[×{np.exp(lo):.2f}, ×{np.exp(hi):.2f}]"
            f"\np = {pooled['p_bootstrap']:.2g}",
            va="center",
            fontsize=8,
        )
        report["both mice"] = dict(
            median_ratio=float(np.exp(m)),
            interval_95=[float(np.exp(lo)), float(np.exp(hi))],
            p_bootstrap=float(pooled["p_bootstrap"]),
        )
        ax.axhline(0, color="0.5", ls="--", lw=0.9)
        ticks = np.log([0.25, 0.5, 1, 2, 4])
        ax.set_yticks(ticks, ["×0.25", "×0.5", "×1", "×2", "×4"])
        ax.set_xticks([0, 1, 2], [*values, "both mice"])
        ax.set_xlim(-0.5, 2.9)
        ax.set_ylim(-LIMIT * 1.03, LIMIT * 1.03)
        outside = sum(int(np.sum(np.abs(r) > LIMIT)) for r in effects.values())
        report["outside_display_limit"] = outside
        if outside:
            ax.text(
                0.99,
                0.01,
                f"{outside} units past the axis, drawn at the edge",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=7,
            )
        ax.set_ylabel(
            "flash × exit part, minus leak\n(after-exit peak ÷ prediction without it)"
        )
        fig.tight_layout()
        print(save_figure(fig, OUT / "flash_exit_effect"))
    (OUT / "flash_exit_effect.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
