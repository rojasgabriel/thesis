"""Supplementary figure: every V1 GLM regressor and how it is constructed.

Draws each regressor from the real basis definitions in glm/dmat.py (no fit needed):
event regressors are raised-cosine bases placed at their event, video PCs enter
contemporaneously with no basis, and spike history uses log-spaced causal bases.

Run from the repo root:
    uv run python figures/supp/regressor_overview.py
"""

from pathlib import Path
from string import ascii_lowercase

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thesis.ephys.analyses.glm.dmat import spike_history_basis, task_temporal_bases

FIGURE_STYLE = {
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 8,
}
GROUP_COLORS = {"sensory": "C0", "task": "C1", "video": "C2", "history": "C3"}


def draw_bases(ax, basis, color, offset_ms=0.0):
    times_ms = basis.basis_time * 1000 + offset_ms
    for index in range(basis.basis.shape[1]):
        ax.plot(times_ms, basis.basis[:, index], color=color, linewidth=1.0, alpha=0.9)
    ax.set_xlim(times_ms[0], times_ms[-1])


def main() -> Path:
    bases = task_temporal_bases()
    hist = spike_history_basis()
    panels = [
        (
            "visual_flash",
            "Visual flash",
            "sensory",
            "6 raised-cosine bases, weight +1 at every flash",
            "time from flash (ms)",
        ),
        (
            "flash_pre_post_withdrawal",
            "Flash: pre/post withdrawal",
            "sensory",
            "same bases, weight -0.5 pre / +0.5 post withdrawal",
            "time from flash (ms)",
        ),
        (
            "flash_adaptation",
            "Flash: adaptation",
            "sensory",
            "same bases, centered synaptic resource at each flash",
            "time from flash (ms)",
        ),
        (
            "initiation",
            "Initiation (center poke)",
            "task",
            "4 bases, 0-90 ms, truncated at that trial's first flash",
            "time from center poke (ms)",
        ),
        (
            "withdrawal",
            "Withdrawal (center exit)",
            "task",
            "9 bases spanning +/-300 ms (peri-event)",
            "time from center exit (ms)",
        ),
        (
            "response_entry",
            "Response entry",
            "task",
            "6 bases before the event, -300 to -1 ms",
            "time from response port entry (ms)",
        ),
        (
            "video",
            "Motion-energy PC",
            "video",
            "1 column per PC, contemporaneous, no temporal basis",
            "time from current bin (ms)",
        ),
        (
            "history",
            "Spike history",
            "history",
            "10 log-spaced causal bases, 1-100 ms strictly past",
            "time since spike (ms)",
        ),
    ]

    with plt.rc_context(FIGURE_STYLE):
        fig, axes = plt.subplots(4, 2, figsize=(9.0, 9.6))
        axes = axes.ravel()
        for index, (ax, (key, title, group, coding, xlabel)) in enumerate(
            zip(axes, panels, strict=True)
        ):
            color = GROUP_COLORS[group]
            if key == "video":
                times_ms = np.linspace(-300, 300, 601)
                score = (np.sin(times_ms / 70) + 0.4 * np.sin(times_ms / 23 + 1)) * 0.6
                ax.plot(times_ms, score, color=color, linewidth=1.2)
                ax.axvline(0, color="0.6", linewidth=0.8, linestyle="--")
                ax.set_xlim(-300, 300)
                ax.set_ylabel("example PC score")
            elif key == "history":
                # History columns use spike times shifted by one bin: lags 1-100 ms.
                draw_bases(ax, hist, color, offset_ms=1000 * hist.binwidth_s)
                ax.set_ylabel("basis weight")
            else:
                draw_bases(ax, bases[key], color)
                ax.set_ylabel("basis weight")
                ax.axhline(0, color="0.85", linewidth=0.6, zorder=0)
                if bases[key].basis_time[0] <= 0 <= bases[key].basis_time[-1]:
                    ax.axvline(0, color="0.5", linewidth=0.8)
            ax.set_title(title, fontsize=9, color=color, pad=12)
            ax.text(
                0.5,
                1.005,
                coding,
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=7,
                color="0.35",
            )
            ax.text(
                -0.14,
                1.12,
                ascii_lowercase[index],
                transform=ax.transAxes,
                fontsize=10,
                fontweight="bold",
                va="bottom",
            )
            ax.set_xlabel(xlabel)

        fig.tight_layout(h_pad=2.4, w_pad=2.0)
        output = Path(__file__).resolve().parent / "regressor_overview.svg"
        fig.savefig(output, bbox_inches="tight")
        plt.close(fig)
    return output


if __name__ == "__main__":
    print("wrote", main())
