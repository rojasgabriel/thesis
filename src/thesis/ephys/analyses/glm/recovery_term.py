"""Show where the recovery term of the GLM comes from.

Question: the GLM scales the flash kernel by how far the unit has recovered
from the flash before. What is that recovery value, and how was its time
constant set?

Comparison: none new; a description of two existing results.

- a: the in-port flash peak against the gap from the flash before, mean ±
  95% interval over the flash-excited units, each mouse, from
  `flash_recovery.json` (`flash_recovery.py`: baseline -40 to 0 ms, peak 30
  to 120 ms, ÷ the unit's mean rate; the sampling unit is the unit). The
  curve is the GRB006 fit peak = A(1 − c e^(−gap/τ)), the recovery of a
  depressed resource (Tsodyks and Markram 1997, Eq. 2; Varela et al. 1997,
  Eq. 8). The first flash of a trial (no flash before) is drawn at the right
  end.
- b: the recovery value of the model, r = 1 − e^(−gap/τ) with
  τ = `dmat.FLASH_RECOVERY_TAU_S`, for gaps above the short gap; shorter gaps
  have the short-gap term instead.

Writes `figures/glm/recovery_term.svg`.

    uv run python -m thesis.ephys.analyses.glm.recovery_term
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thesis.ephys.analyses.glm import dmat
from thesis.plotting import (
    CONCEPT_COLORS,
    FIGURE_STYLE,
    MODEL_COLOR,
    OBSERVED_COLOR,
    label_panels,
    mouse_marker,
    save_figure,
)

OUT = Path("results/glm")
FIRST_FLASH_X_S = 0.36  # where the first flash of a trial is drawn
SHOWN_GAPS_S = (0.08, 0.12, 0.16, 0.2)


def main() -> None:
    fit = json.loads((OUT / "flash_recovery.json").read_text())
    tau_model = dmat.FLASH_RECOVERY_TAU_S
    short = dmat.short_gap_s()
    gaps = np.linspace(0, 0.4, 400)
    visual = CONCEPT_COLORS["visual"]
    with plt.rc_context(FIGURE_STYLE):
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
        a = ax[0]
        for j, subject in enumerate(("GRB006", "GRB064")):
            x = np.array(fit[subject]["median_gap_s"])
            y = np.array(fit[subject]["mean_peak"])
            e = 1.96 * np.array(fit[subject]["sem_peak"])
            x = np.where(x > 1, FIRST_FLASH_X_S, x)
            a.errorbar(
                1000 * x,
                y,
                yerr=e,
                fmt=mouse_marker(subject, j),
                color=OBSERVED_COLOR,
                mfc="white" if j else OBSERVED_COLOR,
                ms=6,
                capsize=2,
                ls="none",
                label=subject,
            )
        A, c, tau = fit["GRB006"]["A"], fit["GRB006"]["c"], fit["GRB006"]["tau_s"]
        a.plot(
            1000 * gaps, A * (1 - c * np.exp(-gaps / tau)), color=MODEL_COLOR, lw=1.8
        )
        low, high = fit["GRB006"]["tau_interval_95_s"]
        interval = f"\n(95% interval of τ: {1000 * low:.0f}–{1000 * high:.0f} ms)"
        a.text(
            0.97,
            0.06,
            f"peak = A·(1 − c·e^(−gap/τ))\nGRB006 fit: A = {A:.2f}, c = {c:.2f}, "
            f"τ = {1000 * tau:.0f} ms{interval}",
            transform=a.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color=MODEL_COLOR,
        )
        a.axhline(0, color="0.6", ls=":", lw=0.8)
        a.set_xticks(
            [40, 80, 120, 160, 200, 240, 320, 1000 * FIRST_FLASH_X_S],
            ["40", "80", "120", "160", "200", "240", "320", "first"],
        )
        a.set_xlabel("gap from the flash before (ms)")
        a.set_ylabel("in-port flash peak ÷ mean rate\n(mean ± 95% interval over units)")
        a.legend(frameon=False, fontsize=8, loc="upper left")
        a.set_ylim(-0.6, 5.5)
        b = ax[1]
        r = 1 - np.exp(-gaps / tau_model)
        b.axvspan(0, 1000 * short, color="0.9", lw=0)
        b.text(
            500 * short,
            0.5,
            f"≤ {1000 * short:.0f} ms:\nshort-gap\nterm instead",
            ha="center",
            va="center",
            fontsize=7,
            color="0.35",
        )
        b.plot(1000 * gaps[gaps > short], r[gaps > short], color=visual, lw=2)
        for g in SHOWN_GAPS_S:
            value = 1 - np.exp(-g / tau_model)
            b.plot(1000 * g, value, "o", color=visual, ms=5)
            b.annotate(
                f"{value:.2f}",
                (1000 * g, value),
                xytext=(0, -14),
                textcoords="offset points",
                ha="center",
                fontsize=7,
            )
        b.text(
            0.97,
            0.06,
            f"r = 1 − e^(−gap/{1000 * tau_model:.0f} ms)\nflash term = kernel + "
            "(r − mean r)·recovery kernel\n(in log rate, so it scales the response)",
            transform=b.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color=visual,
        )
        b.set_ylim(-0.05, 1.08)
        b.set_xlim(0, 400)
        b.set_xlabel("gap from the flash before (ms)")
        b.set_ylabel("recovery value r in the GLM")
        label_panels(ax, x=-0.16, y=1.04)
        fig.tight_layout()
        print(save_figure(fig, OUT / "recovery_term"))


if __name__ == "__main__":
    main()
