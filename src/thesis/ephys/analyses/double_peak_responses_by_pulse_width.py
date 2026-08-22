"""Secondary pulse-width control for the double-peak analysis surface.

This script is kept because it isolates the 15 ms vs 30 ms pulse-width
comparison as its own analysis figure. Use
`double_peak_responses_across_sessions.py` for the collaborator-facing summary
figure.

Sessions in current scope: GRB058 only (longstim sessions for 15 vs 30 ms).
GRB059 and GRB060 are not in the current analysis scope.

Layout: two rows.
  • Top row — single-peak reference example from GRB058.
  • Bottom row — double-peak units (GRB058, 15 ms + 30 ms overlaid).

Classification uses the canonical parameters in peak_classification.py
(FDR selectivity + 5 sp/s height floor on both peaks).
"""

import os
from pathlib import Path
from typing import TypedDict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from spks.event_aligned import population_peth

matplotlib.use("Agg")

from thesis.ephys.io_digital_events import fetch_session_events
from thesis.ephys.io_session_units import fetch_good_units
from thesis.ephys.peak_classification import (
    BASELINE_WINDOW,
    PETH_BINWIDTH_MS,
    PETH_POST_SECONDS,
    PETH_PRE_SECONDS,
    classify_double_peak_units,
    classify_peak_count,
    mark_peaks,
)
from thesis.ephys.peak_classification import (
    plot_mean_sem_trace as plot_trace,
)

GRB058_SESSIONS = ["20260312_134952", "20260319_131303"]
REFERENCE_SESSION = "20260312_134952"
FIGURE_ROOT = Path(os.environ.get("THESIS_FIGURE_ROOT", "figures"))
OUT_PATH = FIGURE_ROOT / "double_peak" / "pulse_split.pdf"


class DoublePeakRow(TypedDict):
    session: str
    uid: int
    peth_15: np.ndarray
    peth_30: np.ndarray
    n_tr_15: int
    n_tr_30: int
    peaks_df_row: pd.Series
    bin_centers: np.ndarray


class SinglePeakRow(TypedDict):
    subject: str
    session: str
    uid: int
    peth_15: np.ndarray
    n_tr_15: int
    peaks_df_row: pd.Series
    bin_centers: np.ndarray


# ---------------------------------------------------------------------------
# Collect double-peak units (GRB058, both sessions)
# ---------------------------------------------------------------------------
def main() -> None:
    dp_rows: list[DoublePeakRow] = []
    reference_data = None

    for session in GRB058_SESSIONS:
        spike_times_by_unit = fetch_good_units("GRB058", session)
        align_ev = fetch_session_events("GRB058", session)
        unit_ids = list(spike_times_by_unit)
        spike_times = list(spike_times_by_unit.values())
        n_tr_15 = len(align_ev["first_stim_ev_15ms"])
        n_tr_30 = len(align_ev["first_stim_ev_30ms"])

        double_peak_rows, peth_15, bin_centers, excited_ids = (
            classify_double_peak_units(
                spike_times, align_ev["first_stim_ev_15ms"], unit_ids
            )
        )
        if session == REFERENCE_SESSION:
            reference_data = (
                unit_ids,
                peth_15,
                bin_centers,
                excited_ids,
                n_tr_15,
            )
        double_ids = double_peak_rows["unit"].astype(int).tolist()

        print(
            f"\nGRB058/{session[:8]}  15ms_trials={n_tr_15}  30ms_trials={n_tr_30}"
            f"  double-peak={double_ids}"
        )

        if not double_ids:
            continue

        dp_idx = [unit_ids.index(uid) for uid in double_ids]
        dp_peth_15 = peth_15[dp_idx]
        dp_spike_times = [spike_times[i] for i in dp_idx]

        peth_30_all, _, _ = population_peth(
            all_spike_times=dp_spike_times,
            alignment_times=align_ev["first_stim_ev_30ms"],
            pre_seconds=PETH_PRE_SECONDS,
            post_seconds=PETH_POST_SECONDS,
            binwidth_ms=PETH_BINWIDTH_MS,
        )
        peth_30_all = peth_30_all / (PETH_BINWIDTH_MS / 1000)

        for j, uid in enumerate(double_ids):
            dp_rows.append(
                dict(
                    session=session,
                    uid=uid,
                    peth_15=dp_peth_15[j],
                    peth_30=peth_30_all[j],
                    n_tr_15=n_tr_15,
                    n_tr_30=n_tr_30,
                    peaks_df_row=double_peak_rows[double_peak_rows["unit"] == uid].iloc[
                        0
                    ],
                    bin_centers=bin_centers,
                )
            )

    if reference_data is None:
        raise RuntimeError(f"Reference session {REFERENCE_SESSION} was not loaded.")
    unit_ids, peth_15, bin_centers, excited_ids, n_tr_15 = reference_data
    excited_peth = peth_15[[unit_ids.index(unit_id) for unit_id in excited_ids]]
    peaks_df = classify_peak_count(excited_peth, bin_centers, excited_ids)
    single_ids = peaks_df.loc[peaks_df["n_peaks"] == 1, "unit"].tolist()

    sensitive_peaks = classify_peak_count(
        excited_peth,
        bin_centers,
        excited_ids,
        min_prominence_frac=0.10,
    )
    robust_single_ids = [
        unit_id
        for unit_id in single_ids
        if sensitive_peaks.loc[sensitive_peaks["unit"] == unit_id, "n_peaks"].iloc[0]
        == 1
    ] or single_ids
    best = max(
        robust_single_ids,
        key=lambda unit_id: (
            excited_peth[excited_ids.index(unit_id)].mean(0).max()
            - excited_peth[excited_ids.index(unit_id)]
            .mean(axis=0)[
                (bin_centers >= BASELINE_WINDOW[0]) & (bin_centers < BASELINE_WINDOW[1])
            ]
            .mean()
        ),
    )
    best_index = excited_ids.index(best)
    sp_rows: list[SinglePeakRow] = [
        {
            "subject": "GRB058",
            "session": REFERENCE_SESSION,
            "uid": best,
            "peth_15": excited_peth[best_index],
            "n_tr_15": n_tr_15,
            "peaks_df_row": peaks_df[peaks_df["unit"] == best].iloc[0],
            "bin_centers": bin_centers,
        }
    ]
    print(f"\nSingle-peak reference  GRB058/{REFERENCE_SESSION[:8]}  unit={best}")

    # ---------------------------------------------------------------------------
    # Figure — 2 rows × N columns (N = number of double-peak units found)
    # Top row: GRB058 single-peak reference (one example, repeated empty otherwise)
    # Bottom row: double-peak units, 15 ms vs 30 ms overlaid
    # ---------------------------------------------------------------------------
    ncols = max(len(dp_rows), 1)
    fig, axes = plt.subplots(
        2, ncols, figsize=(3.5 * ncols, 7), sharey=False, squeeze=False
    )

    # ---- Top row: single-peak reference (only first cell used; others hidden) ---
    for col in range(ncols):
        ax = axes[0, col]
        if col < len(sp_rows):
            row = sp_rows[col]
            bc = row["bin_centers"]
            plot_trace(
                ax, bc, row["peth_15"], "tab:gray", f"15 ms (n={row['n_tr_15']})"
            )
            mark_peaks(ax, row["peaks_df_row"], color="dimgray")
            ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
            ax.set_title(
                f"{row['subject']}  unit {row['uid']}\n{row['session'][:8]}  (single peak)",
                fontsize=8,
            )
            ax.set_ylabel("sp/s", fontsize=8)
            ax.tick_params(labelsize=7)
            if col == 0:
                ax.set_xlabel("Time from stim onset (s)", fontsize=8)
        else:
            ax.axis("off")

    # ---- Bottom row: double-peak units -----------------------------------------
    for col, row in enumerate(dp_rows):
        ax = axes[1, col]
        bc = row["bin_centers"]

        plot_trace(ax, bc, row["peth_15"], "tab:blue", f"15 ms (n={row['n_tr_15']})")
        mark_peaks(ax, row["peaks_df_row"], color="tab:blue")

        plot_trace(ax, bc, row["peth_30"], "tab:orange", f"30 ms (n={row['n_tr_30']})")
        peak_df_30 = classify_peak_count(
            row["peth_30"][np.newaxis, :, :],
            bc,
            [row["uid"]],
        )
        if not peak_df_30.empty:
            mark_peaks(ax, peak_df_30.iloc[0], color="tab:orange")

        ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.legend(fontsize=7, frameon=False, loc="upper right")
        ax.set_title(
            f"GRB058  unit {row['uid']}\n{row['session'][:8]}  (double peak)",
            fontsize=8,
        )
        ax.set_ylabel("sp/s", fontsize=8)
        ax.set_xlabel("Time from stim onset (s)", fontsize=8)
        ax.tick_params(labelsize=7)

    # Row labels on the left margin
    axes[0, 0].annotate(
        "Single-peak\nexample",
        xy=(-0.22, 0.5),
        xycoords="axes fraction",
        fontsize=8,
        ha="right",
        va="center",
        rotation=90,
        fontweight="bold",
    )
    axes[1, 0].annotate(
        "Double-peak\nunits",
        xy=(-0.22, 0.5),
        xycoords="axes fraction",
        fontsize=8,
        ha="right",
        va="center",
        rotation=90,
        fontweight="bold",
    )

    fig.suptitle(
        "Double-peaked V1 responses to LED flashes  —  15 ms vs 30 ms pulse width\n"
        "Used as a control against a simple pulse-offset explanation; triangles mark detected peaks",
        fontsize=10,
        y=1.02,
    )
    fig.tight_layout()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUT_PATH) as pdf:
        pdf.savefig(fig, bbox_inches="tight")

    print(f"\nFigure saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
