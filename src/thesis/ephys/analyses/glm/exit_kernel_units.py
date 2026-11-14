"""Compare each unit's model flash response in the port and after the center exit.

Question: the population flash kernel is larger and earlier after the exit
(`main_figure.py`, the kernel panel). Do single units follow this, or do a few
units make the mean?

Comparison, per unit (paired): the flash kernel of the final refit at a gap of
80 ms or more, in the port against after the exit (`flash_kernels.py`: the
flash kernel plus the after-exit modifier). Measure: the window peak of the
kernel (`flash_kernels.py` `peak`: the mean within `peaks.PEAK_HALF_MS` of its
largest value from 30 to 120 ms, Δ log rate), the peak measure of the other
figures (until 2026-11-13: the mean kernel from 40 to 100 ms). Uncertainty: the SD
of the same measure over the cross-validation fold fits, on both axes.

- Sessions: `flash_kernels.SESSIONS`. Units: all GLM units with a final fit;
  the flash-excited units (`peaks.load_excited_units`) are black, the other
  units gray. Statistics: Wilcoxon signed-rank (after exit against in port),
  each mouse alone, all units and flash-excited units. The sampling unit is
  the unit. Big markers: the median of each mouse.
- Display limit (`LIMIT`): units past it are drawn at the edge as open
  triangles and stay in the statistics; the report lists them.

Needs `flash_kernels.npz` with `response_sd` (run `flash_kernels` first).
Writes `figures/glm/exit_kernel_units.svg` and
`results/glm/exit_kernel_units.json`.

    THESIS_CACHE_DIR=figures/cache uv run python -m \\
        thesis.ephys.analyses.glm.exit_kernel_units
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy.stats import wilcoxon

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thesis.ephys.analyses.glm.flash_kernels import SESSIONS
from thesis.ephys.analyses.glm.peaks import PEAK_HALF_MS, load_excited_units
from thesis.plotting import FIGURE_STYLE, OBSERVED_COLOR, mouse_marker, save_figure

OUT = Path("results/glm")
PORT, EXIT = "in port, 80+ ms gap", "after exit, 80+ ms gap"
LIMIT = (-1.5, 2.5)
OTHER = "0.65"


def main() -> None:
    excited = load_excited_units()
    report = {"limit": list(LIMIT), "peak_half_ms": PEAK_HALF_MS}
    with plt.rc_context(FIGURE_STYLE):
        fig, ax = plt.subplots(figsize=(4.6, 4.6))
        lo, hi = LIMIT
        for m, session in enumerate(SESSIONS):
            subject = session.split("_")[0]
            d = np.load(OUT / session / "flash_kernels.npz")
            cond = list(d["conditions"])
            i, j = cond.index(PORT), cond.index(EXIT)
            x, y = d["peak"][:, i], d["peak"][:, j]
            sx, sy = d["peak_sd"][:, i], d["peak_sd"][:, j]
            exc = np.isin(d["unit_ids"], excited[session])
            out = (x < lo) | (x > hi) | (y < lo) | (y > hi)
            marker = mouse_marker(subject, m)
            for keep, color in ((~exc, OTHER), (exc, OBSERVED_COLOR)):
                inside = keep & ~out
                ax.errorbar(
                    x[inside],
                    y[inside],
                    xerr=sx[inside],
                    yerr=sy[inside],
                    fmt=marker,
                    ms=3.5,
                    color=color,
                    ecolor=color,
                    elinewidth=0.4,
                    alpha=0.7,
                    lw=0,
                )
                ax.scatter(
                    np.clip(x[keep & out], lo, hi),
                    np.clip(y[keep & out], lo, hi),
                    marker="^",
                    s=22,
                    facecolor="white",
                    edgecolor=color,
                    lw=0.8,
                )
            ax.plot(
                np.median(x[exc]),
                np.median(y[exc]),
                marker,
                ms=11,
                color="white",
                mec=OBSERVED_COLOR,
                mew=1.8,
                zorder=5,
            )
            report[subject] = {
                group: dict(
                    units=int(keep.sum()),
                    higher_after_exit=int(np.sum(y[keep] > x[keep])),
                    median_in_port=float(np.median(x[keep])),
                    median_after_exit=float(np.median(y[keep])),
                    p=float(wilcoxon(y[keep], x[keep]).pvalue),
                )
                for group, keep in (("all", np.ones_like(exc)), ("excited", exc))
            }
            report[subject]["past_limit"] = [
                [int(u), float(a), float(b)]
                for u, a, b in zip(d["unit_ids"][out], x[out], y[out], strict=True)
            ]
        ax.plot(LIMIT, LIMIT, color="0.6", ls="--", lw=0.8)
        ax.axhline(0, color="0.85", lw=0.6, zorder=0)
        ax.axvline(0, color="0.85", lw=0.6, zorder=0)
        ax.set_xlim(lo - 0.1, hi + 0.1)
        ax.set_ylim(lo - 0.1, hi + 0.1)
        ax.set_aspect("equal")
        ax.set_xlabel("kernel peak in port (Δ log rate)")
        ax.set_ylabel("kernel peak after exit (Δ log rate)")
        ax.text(
            0.03,
            0.97,
            "flash-excited units",
            color=OBSERVED_COLOR,
            transform=ax.transAxes,
            va="top",
            fontsize=7,
        )
        ax.text(
            0.03,
            0.91,
            "other units",
            color=OTHER,
            transform=ax.transAxes,
            va="top",
            fontsize=7,
        )
        print(save_figure(fig, OUT / "exit_kernel_units"))
    (OUT / "exit_kernel_units.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
