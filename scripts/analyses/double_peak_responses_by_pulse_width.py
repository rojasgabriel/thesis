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

Classification uses canonical params from src/thesis/ephys/config/double_peak.py
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

matplotlib.use("Agg")

from thesis.ephys.config.double_peak import (
    BASELINE_WINDOW,
    PEAK_KWARGS,
    PETH_KWARGS,
    SELECTIVITY_KWARGS,
)
from thesis.ephys.config.typing_params import PeakCountParams
from thesis.ephys.utils.analysis_peak_counts import classify_peak_count
from thesis.ephys.utils.analysis_peth import compute_population_peth
from thesis.ephys.utils.analysis_selectivity import compute_unit_selectivity
from thesis.ephys.utils.io_digital_events import fetch_session_events
from thesis.ephys.utils.io_session_units import fetch_good_units
from thesis.ephys.utils.peak_classification import (
    baseline_mean,
    classify_double_peak_units,
    mark_peaks,
)
from thesis.ephys.utils.peak_classification import (
    plot_mean_sem_trace as plot_trace,
)

GRB058_SESSIONS = ["20260312_134952", "20260319_131303"]
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


def load_session(subject, session):
    """Return (unit_ids, spike_times, align_ev, peth_15, bin_edges, bin_centers, masks)."""
    st_per_unit = fetch_good_units(subject, session)
    align_ev = fetch_session_events(subject, session)
    unit_ids = list(st_per_unit.keys())
    spike_times = list(st_per_unit.values())

    peth_15, bin_edges, bin_centers = compute_population_peth(
        spike_times_per_unit=spike_times,
        alignment_times=align_ev["first_stim_ev_15ms"],
        **PETH_KWARGS,
    )
    _, masks = compute_unit_selectivity(
        peth_15, bin_edges, unit_ids=unit_ids, **SELECTIVITY_KWARGS
    )
    return unit_ids, spike_times, align_ev, peth_15, bin_edges, bin_centers, masks


# ---------------------------------------------------------------------------
# Collect double-peak units (GRB058, both sessions)
# ---------------------------------------------------------------------------
def main() -> None:
    dp_rows: list[DoublePeakRow] = []

    for session in GRB058_SESSIONS:
        spike_times_by_unit = fetch_good_units("GRB058", session)
        align_ev = fetch_session_events("GRB058", session)
        unit_ids = list(spike_times_by_unit)
        spike_times = list(spike_times_by_unit.values())
        n_tr_15 = len(align_ev["first_stim_ev_15ms"])
        n_tr_30 = len(align_ev["first_stim_ev_30ms"])

        double_peak_rows, peth_15, _, bin_centers, _ = classify_double_peak_units(
            spike_times, align_ev["first_stim_ev_15ms"], unit_ids
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

        peth_30_all, _, _ = compute_population_peth(
            spike_times_per_unit=dp_spike_times,
            alignment_times=align_ev["first_stim_ev_30ms"],
            **PETH_KWARGS,
        )

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

    # ---------------------------------------------------------------------------
    # Collect single-peak reference examples — one per animal (GRB058/059/060),
    # best single-peak excited unit chosen by response amplitude above baseline.
    # Using one per animal ensures all recorded subjects appear in the figure.
    # ---------------------------------------------------------------------------
    SP_ANIMAL_SESSIONS = [
        ("GRB058", "20260312_134952"),
    ]

    sp_rows: list[SinglePeakRow] = []
    for subject, session in SP_ANIMAL_SESSIONS:
        unit_ids, spike_times, align_ev, peth_15, bin_edges, bin_centers, masks = (
            load_session(subject, session)
        )
        n_tr_15 = len(align_ev["first_stim_ev_15ms"])

        exc_idx = np.where(masks["excited"])[0]
        exc_peth = peth_15[exc_idx]
        exc_ids = [unit_ids[i] for i in exc_idx]

        peaks_df = classify_peak_count(
            exc_peth, bin_centers, unit_ids=exc_ids, **PEAK_KWARGS
        )
        single_ids = peaks_df.loc[peaks_df["n_peaks"] == 1, "unit"].tolist()

        # Stability check: exclude units that grow a second peak when the
        # prominence threshold is relaxed to 10 %.  Those units have a
        # borderline secondary bump and would be misleading as "single-peak"
        # reference examples.
        sensitive_peak_kwargs: PeakCountParams = {
            **PEAK_KWARGS,
            "min_prominence_frac": 0.10,
        }
        sensitive_peaks = classify_peak_count(
            exc_peth,
            bin_centers,
            unit_ids=exc_ids,
            **sensitive_peak_kwargs,
        )
        robust_single_ids = [
            uid
            for uid in single_ids
            if sensitive_peaks.loc[sensitive_peaks["unit"] == uid, "n_peaks"].iloc[0]
            == 1
        ]
        if not robust_single_ids:
            robust_single_ids = single_ids  # fallback if all have secondary bumps

        best = max(
            robust_single_ids,
            key=lambda uid: (
                exc_peth[exc_ids.index(uid)].mean(0).max()
                - baseline_mean(
                    exc_peth[exc_ids.index(uid)], bin_centers, BASELINE_WINDOW
                )
            ),
        )
        i = exc_ids.index(best)
        sp_rows.append(
            dict(
                subject=subject,
                session=session,
                uid=best,
                peth_15=exc_peth[i],
                n_tr_15=n_tr_15,
                peaks_df_row=peaks_df[peaks_df["unit"] == best].iloc[0],
                bin_centers=bin_centers,
            )
        )
        print(f"\nSingle-peak reference  {subject}/{session[:8]}  unit={best}")

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
            unit_ids=[row["uid"]],
            **PEAK_KWARGS,
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
