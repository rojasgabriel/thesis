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

- One panel, one point per unit, for the units with a flash classification:
  flash-excited units (`peaks.load_excited_units`) with the peak, and
  flash-suppressed units (`peaks.load_suppressed_units`) with the dip of the
  kernel (`flash_kernels.py` `dip`: the same window around its most negative
  value), as a peak has no meaning for them. The sign of the measure tells
  the two apart. Units with no flash classification are not shown.
- Color: the direction of the fixation modulation
  (`hold_effect.fixation_groups`, `plotting.FIXATION_COLORS`).
- Sessions: `flash_kernels.SESSIONS`. Statistics: Wilcoxon signed-rank (after
  exit against in port), each mouse alone, for flash-excited and
  flash-suppressed units, also in each fixation group. The sampling unit is
  the unit. Big open markers: the median of each mouse for flash-excited and
  for flash-suppressed units.
- Display limit (`LIMIT`): units past it are drawn at the edge as open
  triangles and stay in the statistics; the report lists them.

Needs `flash_kernels.npz` with `dip` (run `flash_kernels` first) and the saved
unit files of `peaks.py`.
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
from thesis.ephys.analyses.glm.hold_effect import FIXATION_GROUPS, fixation_groups
from thesis.ephys.analyses.glm.peaks import (
    PEAK_HALF_MS,
    load_excited_units,
    load_suppressed_units,
)
from thesis.plotting import (
    FIGURE_STYLE,
    FIXATION_COLORS,
    OBSERVED_COLOR,
    mouse_marker,
    save_figure,
)

OUT = Path("results/glm")
PORT, EXIT = "in port, 80+ ms gap", "after exit, 80+ ms gap"
LIMIT = (-2.25, 2.5)


def summary(x: np.ndarray, y: np.ndarray) -> dict:
    return dict(
        units=int(x.size),
        higher_after_exit=int(np.sum(y > x)),
        median_in_port=float(np.median(x)) if x.size else float("nan"),
        median_after_exit=float(np.median(y)) if x.size else float("nan"),
        p=float(wilcoxon(y, x).pvalue) if x.size > 1 else float("nan"),
    )


def main() -> None:
    flash = {"excited": load_excited_units(), "suppressed": load_suppressed_units()}
    report = {"limit": list(LIMIT), "peak_half_ms": PEAK_HALF_MS}
    lo, hi = LIMIT
    with plt.rc_context(FIGURE_STYLE):
        fig, ax = plt.subplots(figsize=(4.8, 4.8), layout="constrained")
        for m, session in enumerate(SESSIONS):
            subject = session.split("_")[0]
            d = np.load(OUT / session / "flash_kernels.npz")
            cond = list(d["conditions"])
            i, j = cond.index(PORT), cond.index(EXIT)
            groups = fixation_groups(OUT / session)
            group = np.asarray([groups.get(int(u), "") for u in d["unit_ids"]])
            marker = mouse_marker(subject, m)
            report[subject] = {}
            for kind, measure in (("excited", "peak"), ("suppressed", "dip")):
                keep = np.isin(d["unit_ids"], flash[kind][session])
                x, y = d[measure][:, i], d[measure][:, j]
                sx, sy = d[f"{measure}_sd"][:, i], d[f"{measure}_sd"][:, j]
                out = (x < lo) | (x > hi) | (y < lo) | (y > hi)
                for name in FIXATION_GROUPS:
                    color = FIXATION_COLORS[name]
                    inside = keep & (group == name) & ~out
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
                        alpha=0.75,
                        lw=0,
                    )
                    edge = keep & (group == name) & out
                    ax.scatter(
                        np.clip(x[edge], lo, hi),
                        np.clip(y[edge], lo, hi),
                        marker="^",
                        s=22,
                        facecolor="white",
                        edgecolor=color,
                        lw=0.8,
                    )
                ax.plot(
                    np.median(x[keep]),
                    np.median(y[keep]),
                    marker,
                    ms=11,
                    color="white",
                    mec=OBSERVED_COLOR,
                    mew=1.8,
                    zorder=5,
                )
                report[subject][kind] = {
                    "measure": measure,
                    **summary(x[keep], y[keep]),
                    "by_fixation_group": {
                        name: summary(
                            x[keep & (group == name)], y[keep & (group == name)]
                        )
                        for name in FIXATION_GROUPS
                    },
                    "past_limit": [
                        [int(u), float(a), float(b)]
                        for u, a, b in zip(
                            d["unit_ids"][keep & out],
                            x[keep & out],
                            y[keep & out],
                            strict=True,
                        )
                    ],
                }
        ax.plot(LIMIT, LIMIT, color="0.6", ls="--", lw=0.8)
        ax.axhline(0, color="0.85", lw=0.6, zorder=0)
        ax.axvline(0, color="0.85", lw=0.6, zorder=0)
        ax.set_xlim(lo - 0.1, hi + 0.1)
        ax.set_ylim(lo - 0.1, hi + 0.1)
        ax.set_aspect("equal")
        ax.set_xlabel("kernel response in port (Δ log rate)")
        ax.set_ylabel("kernel response after exit (Δ log rate)")
        for k, name in enumerate(FIXATION_GROUPS):
            ax.text(
                0.03,
                0.97 - 0.055 * k,
                name,
                color=FIXATION_COLORS[name],
                transform=ax.transAxes,
                va="top",
                fontsize=7,
            )
        ax.text(
            0.97,
            0.03,
            "flash-excited: peak\nflash-suppressed: dip",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7,
        )
        print(save_figure(fig, OUT / "exit_kernel_units"))
    (OUT / "exit_kernel_units.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
