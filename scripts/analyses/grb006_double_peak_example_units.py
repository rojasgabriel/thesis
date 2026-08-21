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

from thesis.ephys.config.double_peak import (
    BASELINE_WINDOW,
    MIN_PEAK_HEIGHT_ABS,
    PEAK_KWARGS,
    PETH_KWARGS,
    SELECTIVITY_KWARGS,
)
from thesis.ephys.utils.analysis_peak_counts import classify_peak_count
from thesis.ephys.utils.analysis_peth import compute_population_peth
from thesis.ephys.utils.analysis_selectivity import compute_unit_selectivity
from thesis.ephys.utils.grb006_data import (
    GRB006_SESSION as SESSION,
)
from thesis.ephys.utils.grb006_data import (
    fetch_grb006_spike_times,
    load_grb006_first_stim,
)
from thesis.ephys.utils.peak_classification import (
    baseline_mean,
    mark_peaks,
)
from thesis.ephys.utils.peak_classification import (
    plot_mean_sem_trace as plot_trace,
)

FIGURE_ROOT = Path(os.environ.get("THESIS_FIGURE_ROOT", "figures"))
OUT_PATH = FIGURE_ROOT / "double_peak" / "grb006_examples.pdf"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

N_PANELS = 6


def collect_double_peak_rows():
    first_stim = load_grb006_first_stim()
    unit_ids, spike_times = fetch_grb006_spike_times()

    peth, bin_edges, bin_centers = compute_population_peth(
        spike_times_per_unit=spike_times,
        alignment_times=first_stim,
        **PETH_KWARGS,
    )
    _, masks = compute_unit_selectivity(
        peth, bin_edges, unit_ids=unit_ids, **SELECTIVITY_KWARGS
    )

    exc_idx = np.where(masks["excited"])[0]
    exc_ids = [unit_ids[i] for i in exc_idx]
    exc_peth = peth[exc_idx]
    peaks_df = classify_peak_count(
        exc_peth, bin_centers, unit_ids=exc_ids, **PEAK_KWARGS
    )

    rows = []
    for _, peak_row in peaks_df.loc[peaks_df["n_peaks"] == 2].iterrows():
        uid = int(peak_row["unit"])
        i = exc_ids.index(uid)
        base = baseline_mean(exc_peth[i], bin_centers, BASELINE_WINDOW)
        heights_above = [float(h - base) for h in peak_row["peak_heights"]]
        if min(heights_above) < MIN_PEAK_HEIGHT_ABS:
            continue
        rows.append(
            dict(
                uid=uid,
                peth=exc_peth[i],
                n_trials=len(first_stim),
                peaks_df_row=peak_row,
                bin_centers=bin_centers,
                baseline=base,
                min_above=min(heights_above),
                max_above=max(heights_above),
                peak_times=peak_row["peak_times"],
            )
        )

    rows.sort(key=lambda row: (row["min_above"], row["max_above"]), reverse=True)
    return rows, len(unit_ids), len(exc_ids), len(first_stim)


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
    with PdfPages(OUT_PATH) as pdf:
        pdf.savefig(fig, bbox_inches="tight", dpi=300)
    print(f"\nFigure saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
