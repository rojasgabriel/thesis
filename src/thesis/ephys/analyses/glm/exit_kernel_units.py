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

- Panel a: the flash-excited units (`peaks.load_excited_units`), with the
  kernel peak. Panel b: the flash-suppressed units
  (`peaks.load_suppressed_units`), with the kernel dip (`flash_kernels.py`
  `dip`: the same window around its most negative value), as a peak has no
  meaning for them. One point per unit. Units with no flash classification
  are not shown.
- Color: the direction of the fixation modulation
  (`hold_effect.fixation_groups`, `plotting.FIXATION_COLORS`).
- Sessions: `flash_kernels.SESSIONS`. Statistics: Wilcoxon signed-rank (after
  exit against in port), each mouse alone, for flash-excited and
  flash-suppressed units, also in each fixation group. The sampling unit is
  the unit. Open markers: the median of each mouse and fixation group.
- Display limits (`PANELS`): units past it are drawn at the edge as open
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
    POINT_ALPHA,
    label_panels,
    median_marker,
    mouse_marker,
    save_figure,
)

OUT = Path("results/glm")
PORT, EXIT = "in port, 80+ ms gap", "after exit, 80+ ms gap"
# One panel per flash response: (units, measure, display limit).
PANELS = {
    "excited": ("peak", (-0.75, 2.5)),
    "suppressed": ("dip", (-2.25, 0.5)),
}


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
    report = {
        "limits": {kind: list(limit) for kind, (_, limit) in PANELS.items()},
        "peak_half_ms": PEAK_HALF_MS,
    }
    sessions = []
    for session in SESSIONS:
        d = np.load(OUT / session / "flash_kernels.npz")
        groups = fixation_groups(OUT / session)
        sessions.append(
            (
                session.split("_")[0],
                d,
                np.asarray([groups.get(int(u), "") for u in d["unit_ids"]]),
            )
        )
    with plt.rc_context(FIGURE_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.6), layout="constrained")
        for ax, (kind, (measure, limit)) in zip(axes, PANELS.items(), strict=True):
            lo, hi = limit
            for m, (subject, d, group) in enumerate(sessions):
                cond = list(d["conditions"])
                i, j = cond.index(PORT), cond.index(EXIT)
                marker = mouse_marker(subject, m)
                keep = np.isin(d["unit_ids"], flash[kind][SESSIONS[m]])
                x, y = d[measure][:, i], d[measure][:, j]
                sx, sy = d[f"{measure}_sd"][:, i], d[f"{measure}_sd"][:, j]
                out = (x < lo) | (x > hi) | (y < lo) | (y > hi)
                for name in FIXATION_GROUPS:
                    color = FIXATION_COLORS[name]
                    k = keep & (group == name)
                    ax.errorbar(
                        x[k & ~out],
                        y[k & ~out],
                        xerr=sx[k & ~out],
                        yerr=sy[k & ~out],
                        fmt=marker,
                        ms=3.5,
                        color=color,
                        ecolor=color,
                        elinewidth=0.4,
                        alpha=POINT_ALPHA,
                        lw=0,
                    )
                    ax.scatter(
                        np.clip(x[k & out], lo, hi),
                        np.clip(y[k & out], lo, hi),
                        marker="^",
                        s=22,
                        facecolor="white",
                        edgecolor=color,
                        lw=0.8,
                    )
                    if k.any():
                        median_marker(
                            ax, np.median(x[k]), np.median(y[k]), marker, color
                        )
                report.setdefault(subject, {})[kind] = {
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
            ax.plot(limit, limit, color="0.6", ls="--", lw=0.8)
            ax.axhline(0, color="0.85", lw=0.6, zorder=0)
            ax.axvline(0, color="0.85", lw=0.6, zorder=0)
            ax.set_xlim(lo - 0.1, hi + 0.1)
            ax.set_ylim(lo - 0.1, hi + 0.1)
            ax.set_aspect("equal")
            ax.set_xlabel(f"kernel {measure} in port (Δ log rate)")
            ax.set_ylabel(f"kernel {measure} after exit (Δ log rate)")
            ax.text(
                0,
                1.02,
                f"flash-{kind} units",
                transform=ax.transAxes,
                va="bottom",
            )
            if kind != "excited":
                continue
            # One color legend per figure, in panel a, next to the points below
            # the diagonal.
            for k, name in enumerate(FIXATION_GROUPS):
                ax.text(
                    0.68,
                    0.33 - 0.055 * k,
                    name,
                    color=FIXATION_COLORS[name],
                    transform=ax.transAxes,
                    va="top",
                    fontsize=7,
                )
        label_panels(axes, x=-0.2, y=1.02)
        print(save_figure(fig, OUT / "exit_kernel_units"))
    (OUT / "exit_kernel_units.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
