"""Double-peak PSTH figure for GRB006 first-stim responses.

Uses the archived local GRB006 trial timestamps plus the KS4 spike-time export
to identify double-peak units with the same unsmoothed 10 ms-bin settings used
for the double-peak summary analysis.

Output:
    figures/double_peak/grb006_examples.pdf

Figure layout:
    2 rows x 3 columns showing the top-ranked GRB006 double-peak units,
    ranked by the smaller of the two peak heights above baseline.
"""

import os
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thesis.ephys.utils.io_digital_events import fetch_session_events
from thesis.ephys.utils.io_session_units import fetch_good_units
from thesis.ephys.utils.peak_classification import (
    classify_double_peak_units,
    mark_peaks,
)
from thesis.ephys.utils.peak_classification import (
    plot_mean_sem_trace as plot_trace,
)

FIGURE_ROOT = Path(os.environ.get("THESIS_FIGURE_ROOT", "figures"))
OUT_PATH = FIGURE_ROOT / "double_peak" / "grb006_examples.pdf"

SUBJECT = "GRB006"
SESSION = "20240821_121447"
N_PANELS = 6


def collect_double_peak_rows():
    first_stim = fetch_session_events(SUBJECT, SESSION)["first_stim_ev_15ms"]
    spike_times_by_unit = fetch_good_units(SUBJECT, SESSION)
    unit_ids = list(spike_times_by_unit)
    spike_times = list(spike_times_by_unit.values())
    double_peak_rows, peth, bin_centers, excited_ids = classify_double_peak_units(
        spike_times, first_stim, unit_ids
    )

    rows = []
    for _, peak_row in double_peak_rows.iterrows():
        uid = int(peak_row["unit"])
        rows.append(
            dict(
                uid=uid,
                peth=peth[unit_ids.index(uid)],
                n_trials=len(first_stim),
                peaks_df_row=peak_row,
                bin_centers=bin_centers,
                baseline=peak_row["baseline"],
                min_above=peak_row["min_peak_height_above_baseline"],
                max_above=peak_row["max_peak_height_above_baseline"],
                peak_times=peak_row["peak_times"],
            )
        )

    rows.sort(key=lambda row: (row["min_above"], row["max_above"]), reverse=True)
    return rows, len(unit_ids), len(excited_ids), len(first_stim)


def make_figure(rows):
    ncols = 3
    nrows = int(np.ceil(len(rows[:N_PANELS]) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10.5, 7), sharex=True, sharey=False)
    axes = np.atleast_1d(axes).ravel()

    for ax, row in zip(axes, rows[:N_PANELS]):
        bc = row["bin_centers"]
        plot_trace(ax, bc, row["peth"], "tab:blue")
        mark_peaks(ax, row["peaks_df_row"], "tab:blue")
        ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
        peak_ms = ", ".join(f"{int(round(1000 * t))}" for t in row["peak_times"])
        ax.set_title(
            f"GRB006 unit {row['uid']}\npeaks at {peak_ms} ms",
            fontsize=8,
        )
        ax.set_xlabel("Time from first stim onset (s)", fontsize=8)
        ax.set_ylabel("sp/s", fontsize=8)
        ax.tick_params(labelsize=7)

    for ax in axes[len(rows[:N_PANELS]) :]:
        ax.axis("off")

    fig.suptitle(
        "GRB006 double-peak V1 units aligned to first stim onset\n"
        "Unsmoothened 10 ms bins; triangles mark detected peaks",
        fontsize=10,
        y=0.98,
    )
    fig.tight_layout()
    return fig


def main():
    rows, n_units, n_excited, n_trials = collect_double_peak_rows()
    if not rows:
        raise RuntimeError("No GRB006 double-peak units passed the filters.")

    print(f"Session: {SESSION}")
    print("Spike times: good units")
    print(f"Units loaded: {n_units}")
    print(f"First-stim events: {n_trials}")
    print(f"Excited units: {n_excited}")
    print(f"Double-peak units: {len(rows)}")
    print("\nRanked candidates:")
    for rank, row in enumerate(rows, start=1):
        peak_ms = [int(round(1000 * t)) for t in row["peak_times"]]
        print(
            f"  {rank}. unit {row['uid']}  peaks={peak_ms} ms  "
            f"min_above={row['min_above']:.1f} sp/s  "
            f"max_above={row['max_above']:.1f} sp/s"
        )

    fig = make_figure(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUT_PATH) as pdf:
        pdf.savefig(fig, bbox_inches="tight", dpi=300)
    print(f"\nFigure saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
