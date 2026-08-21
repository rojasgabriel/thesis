"""Build the collaborator-facing double-peak summary figure.

Story:
1. Anne and Gabriel first noticed the double-peak response shape in GRB006.
2. The same motif reappeared in GRB058, but in fewer units.
3. A simple onset+offset explanation motivated the GRB058 pulse-width test.
4. March 12 and March 19 are shown separately because their recorded 30 ms
   fractions differed and the outcomes should not be pooled.
"""

import os
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thesis.ephys.config.double_peak import (
    PEAK_KWARGS,
    PETH_KWARGS,
)
from thesis.ephys.utils.analysis_peak_counts import classify_peak_count
from thesis.ephys.utils.analysis_peth import compute_population_peth
from thesis.ephys.utils.grb006_data import (
    GRB006_SESSION,
    fetch_grb006_spike_times,
    load_grb006_first_stim,
)
from thesis.ephys.utils.io_chipmunk_trials import fetch_trial_metadata
from thesis.ephys.utils.io_digital_events import fetch_session_events
from thesis.ephys.utils.io_session_units import fetch_good_units
from thesis.ephys.utils.peak_classification import (
    classify_double_peak_units,
    mark_peaks,
)
from thesis.ephys.utils.peak_classification import (
    plot_mean_sem_trace as plot_trace,
)

GRB006_SHOW_UNITS = [579, 694, 217]

GRB058_SUBJECT = "GRB058"
SESSION_ORDER = ["20260312_134952", "20260319_131303"]
SESSION_LABELS = {
    "20260312_134952": "GRB058  2026-03-12",
    "20260319_131303": "GRB058  2026-03-19",
}
INTENDED_4HZ_30MS_FRAC = {
    "20260312_134952": 0.25,
    "20260319_131303": 0.50,
}
SESSION_SHOW_UNITS = {
    "20260312_134952": [410, 651],
    "20260319_131303": [515],
}

FIGURE_ROOT = Path(os.environ.get("THESIS_FIGURE_ROOT", "figures"))
FIGURE_DIR = FIGURE_ROOT / "double_peak"
OUT_PATH = FIGURE_DIR / "dario_story.pdf"


def collect_grb006():
    first_stim = load_grb006_first_stim()
    unit_ids, spike_times = fetch_grb006_spike_times()
    double_peak_rows, peth, _, bin_centers, excited_ids = classify_double_peak_units(
        spike_times, first_stim, unit_ids
    )

    rows = {}
    for _, peak_row in double_peak_rows.iterrows():
        uid = int(peak_row["unit"])
        rows[uid] = dict(
            uid=uid,
            peth=peth[unit_ids.index(uid)],
            bin_centers=bin_centers,
            peaks_df_row=peak_row,
            peak_times_ms=[int(round(1000 * t)) for t in peak_row["peak_times"]],
        )

    return {
        "session": GRB006_SESSION,
        "n_units": len(unit_ids),
        "n_excited": len(excited_ids),
        "n_double": len(double_peak_rows),
        "rows": rows,
    }


def classify_first_stim_widths_by_trial(align_ev, trial_df):
    first15 = pd.DataFrame(
        {
            "stim_onset": np.asarray(align_ev["first_stim_ev_15ms"], dtype=float),
            "width_ms": 15,
        }
    )
    first30 = pd.DataFrame(
        {
            "stim_onset": np.asarray(align_ev["first_stim_ev_30ms"], dtype=float),
            "width_ms": 30,
        }
    )
    first = (
        pd.concat([first15, first30], ignore_index=True)
        .sort_values("stim_onset")
        .reset_index(drop=True)
    )

    trial_starts = trial_df["trial_start_ts"].to_numpy(dtype=float)
    trial_idx = (
        np.searchsorted(
            trial_starts, first["stim_onset"].to_numpy(dtype=float), side="right"
        )
        - 1
    )
    valid = (trial_idx >= 0) & (trial_idx < len(trial_df))
    first = first.loc[valid].copy()
    first["trial_idx"] = trial_idx[valid]
    first = first.drop_duplicates("trial_idx", keep="first")

    merged = trial_df.reset_index(drop=True).join(
        first.set_index("trial_idx")[["width_ms"]], how="left"
    )
    merged["has_classified_first_stim"] = merged["width_ms"].notna()
    return merged


def collect_grb058_session(session):
    st_per_unit = fetch_good_units(GRB058_SUBJECT, session)
    align_ev = fetch_session_events(GRB058_SUBJECT, session)
    trial_df = fetch_trial_metadata(GRB058_SUBJECT, session, align_ev)
    if trial_df is None:
        raise RuntimeError(
            f"Could not load trial metadata for {GRB058_SUBJECT} {session}"
        )

    unit_ids = list(st_per_unit.keys())
    spike_times = list(st_per_unit.values())
    double_peak_rows, peth_15, _, bin_centers, excited_ids = classify_double_peak_units(
        spike_times, align_ev["first_stim_ev_15ms"], unit_ids
    )
    double_ids = double_peak_rows["unit"].astype(int).tolist()

    rows = {}
    if double_ids:
        peth_30, _, _ = compute_population_peth(
            spike_times_per_unit=[
                spike_times[unit_ids.index(uid)] for uid in double_ids
            ],
            alignment_times=align_ev["first_stim_ev_30ms"],
            **PETH_KWARGS,
        )
        peaks_df_30 = classify_peak_count(
            peth_30, bin_centers, unit_ids=double_ids, **PEAK_KWARGS
        )

        for j, uid in enumerate(double_ids):
            rows[uid] = dict(
                uid=uid,
                peth_15=peth_15[unit_ids.index(uid)],
                peth_30=peth_30[j],
                bin_centers=bin_centers,
                peak_row_15=double_peak_rows[double_peak_rows["unit"] == uid].iloc[0],
                peak_row_30=peaks_df_30[peaks_df_30["unit"] == uid].iloc[0],
            )

    classified_trial_df = classify_first_stim_widths_by_trial(align_ev, trial_df)
    rate4 = classified_trial_df[
        (classified_trial_df["has_classified_first_stim"])
        & (classified_trial_df["stim_rate_vision"] == 4)
    ]
    n_4hz_total = int((trial_df["stim_rate_vision"] == 4).sum())
    n_4hz_classified = len(rate4)
    n_4hz_15 = int((rate4["width_ms"] == 15).sum())
    n_4hz_30 = int((rate4["width_ms"] == 30).sum())
    frac_30 = n_4hz_30 / n_4hz_classified if n_4hz_classified else np.nan

    return {
        "label": SESSION_LABELS[session],
        "session": session,
        "n_units": len(unit_ids),
        "n_excited": len(excited_ids),
        "n_double": len(double_ids),
        "rows": rows,
        "n_tr_15": len(align_ev["first_stim_ev_15ms"]),
        "n_tr_30": len(align_ev["first_stim_ev_30ms"]),
        "n_4hz_total": n_4hz_total,
        "n_4hz_classified": n_4hz_classified,
        "n_4hz_15": n_4hz_15,
        "n_4hz_30": n_4hz_30,
        "frac_30": frac_30,
        "intended_frac_30": INTENDED_4HZ_30MS_FRAC[session],
    }


def plot_session_mix(ax, session_data):
    counts = [session_data["n_4hz_15"], session_data["n_4hz_30"]]
    ax.bar(["15 ms", "30 ms"], counts, color=["tab:blue", "tab:orange"], width=0.6)
    ax.set_title(
        f"{session_data['label']}\n4 Hz first-stim trial mix",
        fontsize=10,
    )
    ax.set_ylabel("Classified trials", fontsize=10)
    ax.tick_params(labelsize=9)


def plot_summary_table(ax, session_0312, session_0319):
    ax.axis("off")
    summary = (
        "GRB058 session summary\n\n"
        f"2026-03-12\n"
        f"  double-peak: {session_0312['n_double']}/{session_0312['n_units']} good units\n"
        f"  4 Hz long pulses: {session_0312['n_4hz_30']}/{session_0312['n_4hz_classified']} "
        f"({100 * session_0312['frac_30']:.1f}%)\n"
        f"  classified first pulses: {session_0312['n_tr_15']} 15 ms, {session_0312['n_tr_30']} 30 ms\n\n"
        f"2026-03-19\n"
        f"  double-peak: {session_0319['n_double']}/{session_0319['n_units']} good units\n"
        f"  4 Hz long pulses: {session_0319['n_4hz_30']}/{session_0319['n_4hz_classified']} "
        f"({100 * session_0319['frac_30']:.1f}%)\n"
        f"  classified first pulses: {session_0319['n_tr_15']} 15 ms, {session_0319['n_tr_30']} 30 ms"
    )
    ax.text(
        0.02,
        0.98,
        summary,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        linespacing=1.5,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#f7f7f7", "alpha": 0.95},
    )


def make_figure(grb006, session_0312, session_0319):
    fig = plt.figure(figsize=(12.5, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.9, wspace=0.45)

    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    for ax, uid in zip(axes, GRB006_SHOW_UNITS):
        row = grb006["rows"][uid]
        plot_trace(ax, row["bin_centers"], row["peth"], "tab:blue", "15 ms")
        mark_peaks(ax, row["peaks_df_row"], "tab:blue")
        ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
        peak_ms = ", ".join(str(x) for x in row["peak_times_ms"])
        ax.set_title(f"GRB006 unit {uid}\npeaks at {peak_ms} ms", fontsize=10)
        ax.set_xlabel("Time from first stim onset (s)", fontsize=10)
        ax.set_ylabel("sp/s", fontsize=10)
        ax.tick_params(labelsize=9)
        ax.set_ylim(bottom=0)

    session_0312_axes = [fig.add_subplot(gs[1, i]) for i in range(3)]
    for ax, uid in zip(session_0312_axes[:2], SESSION_SHOW_UNITS["20260312_134952"]):
        row = session_0312["rows"][uid]
        bc = row["bin_centers"]
        plot_trace(
            ax, bc, row["peth_15"], "tab:blue", f"15 ms (n={session_0312['n_tr_15']})"
        )
        mark_peaks(ax, row["peak_row_15"], "tab:blue")
        plot_trace(
            ax, bc, row["peth_30"], "tab:orange", f"30 ms (n={session_0312['n_tr_30']})"
        )
        mark_peaks(ax, row["peak_row_30"], "tab:orange")
        ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_title(f"{session_0312['label']}\nunit {uid}", fontsize=10)
        ax.set_xlabel("Time from first stim onset (s)", fontsize=10)
        ax.set_ylabel("sp/s", fontsize=10)
        ax.tick_params(labelsize=9)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8, frameon=False, loc="upper left")
    plot_session_mix(session_0312_axes[2], session_0312)

    session_0319_axes = [fig.add_subplot(gs[2, i]) for i in range(3)]
    row = session_0319["rows"][SESSION_SHOW_UNITS["20260319_131303"][0]]
    bc = row["bin_centers"]
    plot_trace(
        session_0319_axes[0],
        bc,
        row["peth_15"],
        "tab:blue",
        f"15 ms (n={session_0319['n_tr_15']})",
    )
    mark_peaks(session_0319_axes[0], row["peak_row_15"], "tab:blue")
    plot_trace(
        session_0319_axes[0],
        bc,
        row["peth_30"],
        "tab:orange",
        f"30 ms (n={session_0319['n_tr_30']})",
    )
    mark_peaks(session_0319_axes[0], row["peak_row_30"], "tab:orange")
    session_0319_axes[0].axvline(0, color="gray", linestyle="--", linewidth=0.8)
    session_0319_axes[0].set_title(
        f"{session_0319['label']}\nunit {SESSION_SHOW_UNITS['20260319_131303'][0]}",
        fontsize=10,
    )
    session_0319_axes[0].set_xlabel("Time from first stim onset (s)", fontsize=10)
    session_0319_axes[0].set_ylabel("sp/s", fontsize=10)
    session_0319_axes[0].tick_params(labelsize=9)
    session_0319_axes[0].set_ylim(bottom=0)
    session_0319_axes[0].legend(fontsize=8, frameon=False, loc="upper left")
    plot_session_mix(session_0319_axes[1], session_0319)
    plot_summary_table(session_0319_axes[2], session_0312, session_0319)

    fig.text(
        0.06,
        0.94,
        (
            "A. Double peaks were first noticed in GRB006 "
            f"({grb006['n_double']}/{grb006['n_units']} good units)"
        ),
        fontsize=10,
        fontweight="bold",
    )
    fig.text(
        0.06,
        0.63,
        (
            "B. GRB058 2026-03-12: long-pulse manipulation session "
            f"({session_0312['n_double']}/{session_0312['n_units']} good units)"
        ),
        fontsize=10,
        fontweight="bold",
    )
    fig.text(
        0.06,
        0.315,
        (
            "C. GRB058 2026-03-19: follow-up session shown separately "
            f"({session_0319['n_double']}/{session_0319['n_units']} good units)"
        ),
        fontsize=10,
        fontweight="bold",
    )

    fig.suptitle(
        "Double-peaked flash responses in mouse VISp\n"
        "Unsmoothed 10 ms bins; triangles mark detected peaks",
        fontsize=12,
        y=0.985,
    )
    fig.subplots_adjust(top=0.88, bottom=0.08, left=0.07, right=0.98)
    return fig


def main():
    grb006 = collect_grb006()
    session_0312 = collect_grb058_session("20260312_134952")
    session_0319 = collect_grb058_session("20260319_131303")
    fig = make_figure(grb006, session_0312, session_0319)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUT_PATH) as pdf:
        pdf.savefig(fig, bbox_inches="tight", dpi=300)

    print(f"Figure saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
