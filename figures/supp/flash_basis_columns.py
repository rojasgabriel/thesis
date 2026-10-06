"""Supplementary figure: one flash expands into 6 basis columns.

Clarifies that the diagonal "gradient" in the flash design block is the basis
tiling time, not model weight: each of the 6 columns is one raised-cosine bump
peaking at a later lag. Design values are fixed inputs; the model's temporal
weighting appears only after fitting, in the reconstructed kernel.

Run from the repo root:
    uv run python figures/supp/flash_basis_columns.py
"""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thesis.ephys.analyses.glm.dmat import task_temporal_bases

FIGURE_STYLE = {
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def main() -> Path:
    flash = task_temporal_bases()["visual_flash"]  # (151, 6)
    times_ms = flash.basis_time * 1000
    basis = flash.basis

    with plt.rc_context(FIGURE_STYLE):
        fig, (curves, block) = plt.subplots(
            1,
            2,
            figsize=(8.6, 5.2),
            sharey=True,
            gridspec_kw={"width_ratios": (1.25, 1.0)},
        )
        colors = plt.cm.viridis(np.linspace(0, 0.9, basis.shape[1]))
        for index in range(basis.shape[1]):
            curves.plot(basis[:, index], times_ms, color=colors[index], linewidth=1.6)
            # Keep edge labels off the axes.
            peak = np.clip(times_ms[np.argmax(basis[:, index])], 4, 146)
            curves.text(
                1.03, peak, str(index + 1), color=colors[index], fontsize=8, va="center"
            )
        curves.set_ylim(150, 0)
        curves.set_xlabel("design value (basis height)")
        curves.set_ylabel("time from flash (ms)")
        curves.set_title(
            "the 6 basis functions\n(each peaks at a different lag)", fontsize=9
        )
        curves.set_xlim(0, 1.12)

        image = block.imshow(
            basis,
            aspect="auto",
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
            extent=(0.5, 6.5, 150, 0),
            interpolation="nearest",
        )
        block.set_xticks(range(1, 7))
        block.set_xlabel("column (basis index)")
        block.set_title("the same thing as a\n6-column design block", fontsize=9)
        bar = fig.colorbar(image, ax=block, fraction=0.06, pad=0.04)
        bar.set_label("design value (an input, not a fitted weight)", fontsize=8)
        for ax, letter in ((curves, "a"), (block, "b")):
            ax.text(
                -0.12,
                1.1,
                letter,
                transform=ax.transAxes,
                fontsize=10,
                fontweight="bold",
                va="bottom",
            )

        fig.tight_layout()
        output = Path(__file__).resolve().parent / "flash_basis_columns.svg"
        fig.savefig(output, bbox_inches="tight")
        plt.close(fig)
    return output


if __name__ == "__main__":
    print("wrote", main())
