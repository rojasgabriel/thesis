"""Compact scatter grid: waveform profile of double-peak units.

Sessions: GRB006 20240821 + GRB058 20260312.

Layout
------
         GRB006 20240821     |  GRB058 20260312
Row 0:   FR vs spike_dur     |  FR vs spike_dur

All good units shown (not just excited). Double-peak units in orange,
all others in blue. FS/RS boundary line at 0.4 ms (visual reference only).

Excited units come from the stored stimulus-responsiveness results. Peak shape
uses the 15 ms trials in mixed-width sessions and a 5 sp/s height floor on both
peaks.

GRB006 event loading uses its thresholded NIDQ analog-input events.
GRB006 spike times use good-unit rows.
GRB058 uses the same event and spike pipeline.

Output
------
    figures/double_peak/waveform_grid.pdf
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from thesis.ephys.io_digital_events import fetch_session_events
from thesis.ephys.io_session_units import (
    fetch_good_unit_metrics_table,
    fetch_stimulus_excited_unit_ids,
)
from thesis.ephys.peak_classification import (
    PEAK_SEARCH_WINDOW,
    classify_double_peak_units,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SESSIONS = [
    ("GRB006", "20240821_121447"),
    ("GRB058", "20260312_134952"),
]

FIGURE_ROOT = Path(os.environ.get("THESIS_FIGURE_ROOT", "figures"))
OUT_PATH = FIGURE_ROOT / "double_peak" / "waveform_grid.pdf"
OUT_PATH_MONO = FIGURE_ROOT / "double_peak" / "waveform_grid_nocolor.pdf"

UNIT_CRITERIA_ID = 1
STABILITY_PARAM_ID = 0
RESPONSIVENESS_PARAM_ID = 0
NARROW_BROAD_MS = 0.4  # FS/RS boundary, visual reference only

COL_OTHER = "#4C72B0"
COL_DOUBLE = "#DD8452"  # orange

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def fetch_unit_table(subject: str, session: str) -> pd.DataFrame:
    return fetch_good_unit_metrics_table(
        subject, session, UNIT_CRITERIA_ID, STABILITY_PARAM_ID
    ).reset_index(drop=True)


def classify_double_peak(
    df: pd.DataFrame,
    first_stim: np.ndarray,
    excited_unit_ids: set[int],
) -> pd.DataFrame:
    unit_ids = df["unit_id"].tolist()
    spike_times = df["spike_times_s"].tolist()

    if len(first_stim) == 0:
        df["is_double"] = False
        return df

    double_peak_rows, *_ = classify_double_peak_units(
        spike_times, first_stim, unit_ids, excited_unit_ids
    )
    double_ids = set(double_peak_rows["unit"].astype(int))
    df["is_double"] = df["unit_id"].isin(double_ids)

    print(f"    double-peak n={len(double_ids)}")
    if double_ids:
        print(f"      unit_ids: {sorted(double_ids)}")
    return df


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def make_grid(session_data, color_by_double: bool = True):
    """session_data: list of dicts with keys subject, session, df."""
    ncols = len(session_data)
    fig, axes = plt.subplots(
        1,
        ncols,
        figsize=(4.5 * ncols, 4.5),
        constrained_layout=True,
    )
    if ncols == 1:
        axes = [axes]

    for col, sd in enumerate(session_data):
        df = sd["df"]
        subj, sess = sd["subject"], sd["session"]

        n_double = int(df["is_double"].sum())
        col_title = f"{subj}  {sess[:8]}\nn={len(df)}  double-peak={n_double}"

        ax = axes[col]
        if color_by_double:
            other_units = df[~df["is_double"]]
            double_units = df[df["is_double"]]
            ax.scatter(
                other_units["spike_duration_ms"],
                other_units["firing_rate"],
                s=14,
                alpha=0.40,
                color=COL_OTHER,
                rasterized=True,
                label=f"other (n={len(other_units)})",
            )
            ax.scatter(
                double_units["spike_duration_ms"],
                double_units["firing_rate"],
                s=28,
                alpha=0.90,
                color=COL_DOUBLE,
                zorder=3,
                edgecolors="k",
                linewidths=0.3,
                label=f"double-peak (n={len(double_units)})",
            )
            ax.legend(frameon=False, fontsize=8, loc="upper right")
        else:
            ax.scatter(
                df["spike_duration_ms"],
                df["firing_rate"],
                s=16,
                alpha=0.45,
                color="0.25",
                rasterized=True,
            )
        ax.axvline(NARROW_BROAD_MS, color="k", lw=0.7, ls="--", alpha=0.6)
        ax.set_xlabel("Spike duration (ms)", fontsize=9)
        ax.set_ylabel("Firing rate (sp/s)", fontsize=9)
        ax.set_title(col_title, fontsize=9)
        ax.tick_params(labelsize=8)

    if color_by_double:
        fig.suptitle(
            "Double-peak waveform profile  ·  all good units shown  ·  "
            f"peak search={PEAK_SEARCH_WINDOW}  ·  "
            f"FS/RS boundary = {NARROW_BROAD_MS} ms",
            fontsize=10,
        )
    else:
        fig.suptitle(
            "Waveform profile of all good units  ·  no double-peak color split  ·  "
            f"FS/RS boundary = {NARROW_BROAD_MS} ms",
            fontsize=10,
        )
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print(f"Figure root: {FIGURE_ROOT}")
    print(f"Will write: {OUT_PATH}")
    print(f"Will write: {OUT_PATH_MONO}")
    session_data = []

    for subject, session in SESSIONS:
        print(f"\n{subject} / {session}")
        try:
            df = fetch_unit_table(subject, session)
        except Exception as e:
            print(f"  ✗ unit table: {e}")
            continue
        print(f"  units: {len(df)}")

        try:
            _, stimulus_pulses = fetch_session_events(subject, session)
            widths = stimulus_pulses["width_ms"].dropna()
            first_pulses = stimulus_pulses[stimulus_pulses["first_in_train"]]
            if widths.nunique() > 1:
                first_pulses = first_pulses[first_pulses["width_ms"].eq(15)]
            first_stim = first_pulses["timestamp"].to_numpy(dtype=float)
            excited_unit_ids = fetch_stimulus_excited_unit_ids(
                subject, session, UNIT_CRITERIA_ID, RESPONSIVENESS_PARAM_ID
            )
        except Exception as e:
            print(f"  ✗ events: {e}")
            continue
        print(f"  first_stim events: {len(first_stim)}")

        df = classify_double_peak(df, first_stim, excited_unit_ids)
        n_dp = int(df["is_double"].sum())
        print(f"  double-peak: {n_dp}")

        session_data.append(dict(subject=subject, session=session, df=df))

    if not session_data:
        print("\nNo sessions loaded (missing data backend/plugins?). Nothing to plot.")
        return

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig = make_grid(session_data)
    with PdfPages(OUT_PATH) as pdf:
        pdf.savefig(fig, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"\nSaved → {OUT_PATH}")

    fig_mono = make_grid(session_data, color_by_double=False)
    with PdfPages(OUT_PATH_MONO) as pdf:
        pdf.savefig(fig_mono, bbox_inches="tight", dpi=300)
    plt.close(fig_mono)
    print(f"Saved → {OUT_PATH_MONO}")


if __name__ == "__main__":
    main()
