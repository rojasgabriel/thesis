"""Supplementary figure: the flash interaction design blocks for one trial.

Shows that the three flash regressors are the SAME 6-column raised-cosine basis
placed at the SAME flash times, each block scaled per flash by its contrast code
(pre/post withdrawal, first/later). Rows are 1 ms bins; 18 columns = 3 blocks x
6 basis functions. Red is positive, blue is negative.

Run from the repo root:
    uv run python figures/supp/flash_interaction_design.py
"""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from damn.design_matrix import generate_aligned_bases

from thesis.ephys.analyses.glm import task_temporal_bases
from thesis.ephys.preprocessing.prepare_glm import BINWIDTH_S

FIGURE_STYLE = {"font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42}


def main() -> Path:
    flash = task_temporal_bases()["visual_flash"]  # 6 bases, 0-150 ms

    # One example trial. t=0 is the trial reference (the first flash here).
    align = np.array([0.30])
    pre, post = 0.05, 0.30
    flashes = np.array([0.30, 0.34, 0.38, 0.44])
    center_exit = 0.40  # withdrawal time
    state = np.where(flashes < center_exit, -0.5, 0.5)  # pre -0.5 / post +0.5
    serial = np.full(len(flashes), -0.5)
    serial[0] = 0.5  # first +0.5 / later -0.5

    kw = dict(
        master_alignment_times=align,
        master_pre_s=pre,
        master_post_s=post,
        event_times=flashes,
        binwidth_s=BINWIDTH_S,
        basis=flash.basis,
        basis_time=flash.basis_time,
    )
    common, _, _ = generate_aligned_bases(
        basis_scaling_vals=np.ones(len(flashes)), **kw
    )
    state_block, _, _ = generate_aligned_bases(basis_scaling_vals=state, **kw)
    serial_block, _, _ = generate_aligned_bases(basis_scaling_vals=serial, **kw)
    design = np.hstack([common, state_block, serial_block])
    nbin = design.shape[0]

    with plt.rc_context(FIGURE_STYLE):
        fig = plt.figure(figsize=(9.0, 6.2))
        grid = fig.add_gridspec(1, 2, width_ratios=(1.0, 3.4), wspace=0.05)

        events = fig.add_subplot(grid[0])
        for index, flash_time in enumerate(flashes):
            row = (flash_time - align[0] + pre) / BINWIDTH_S
            events.axhline(row, color="0.3", linewidth=1.1)
            tag = ("first, " if serial[index] > 0 else "") + (
                "pre" if state[index] < 0 else "post"
            )
            events.text(
                0.55,
                row,
                f"flash {index + 1}\n({tag})",
                va="center",
                ha="left",
                fontsize=7,
            )
        withdrawal_row = (center_exit - align[0] + pre) / BINWIDTH_S
        events.axhline(withdrawal_row, color="C1", linewidth=1.3, linestyle="--")
        events.text(0.05, withdrawal_row - 4, "withdrawal", color="C1", fontsize=7)
        events.set_ylim(nbin, 0)
        events.set_xlim(0, 1)
        events.axis("off")
        events.set_title("events", fontsize=9)

        matrix = fig.add_subplot(grid[1])
        image = matrix.imshow(
            design,
            aspect="auto",
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
            extent=(0, 18, nbin, 0),
            interpolation="nearest",
        )
        for boundary in (6, 12):
            matrix.axvline(boundary, color="black", linewidth=1.2)
        matrix.set_xticks([3, 9, 15])
        matrix.set_xticklabels(
            [
                "visual_flash\n(common, x1)",
                "flash_pre_post\n(x -0.5 / +0.5)",
                "flash_first_later\n(x +0.5 / -0.5)",
            ],
            fontsize=8,
        )
        matrix.set_yticks([])
        matrix.set_xlabel("18 design columns  =  3 blocks x 6 basis functions")
        matrix.set_title("design matrix for this trial (rows = 1 ms bins)", fontsize=9)
        bar = fig.colorbar(image, ax=matrix, fraction=0.03, pad=0.02)
        bar.set_label("design value", fontsize=8)

        fig.suptitle(
            "Same flash bumps, scaled per flash by its contrast code",
            y=0.98,
            fontsize=11,
        )
        fig.subplots_adjust(left=0.02, right=0.92, top=0.9, bottom=0.1)
        output = Path(__file__).resolve().parent / "flash_interaction_design.pdf"
        fig.savefig(output, bbox_inches="tight")
        plt.close(fig)
    return output


if __name__ == "__main__":
    print("wrote", main())
