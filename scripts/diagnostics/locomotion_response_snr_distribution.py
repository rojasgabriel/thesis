"""Legacy SNR diagnostic for the removed locomotion gate.

Plots the per-unit SNR (mean response / SEM across trials) in the RESP_WINDOW
for both stat and move conditions, separately for GRB006 and GRB058.
A vertical line marks the historical SNR threshold of 3.0 so we can inspect
why that gate was dropped. This is a reference script, not part of the
main condition-specific peak locomotion analysis.

Output: figures/locomotion/snr_distribution.pdf
"""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from thesis.ephys.config.locomotion import (
    PETH_KWARGS,
    RESP_WINDOW,
)
from thesis.ephys.utils.analysis_conditioned_stim import (
    build_trial_stim_classification,
    extract_conditioned_stim_anchors,
)
from thesis.ephys.utils.analysis_peth import compute_population_peth
from thesis.ephys.utils.grb006_data import (
    fetch_grb006_spike_times,
    load_grb006_aligned_trial_data,
)
from thesis.ephys.utils.io_chipmunk_trials import fetch_trial_metadata
from thesis.ephys.utils.io_digital_events import fetch_session_events
from thesis.ephys.utils.io_session_units import fetch_good_units

OUT_PATH = Path("figures/locomotion/snr_distribution.pdf")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
LEGACY_SNR_THRESHOLD = 3.0


def snr_per_unit(peth, bc, window=RESP_WINDOW):
    mask = (bc >= window[0]) & (bc < window[1])
    resp = peth[:, :, mask].mean(axis=2)  # (n_units, n_trials)
    unit_mean = resp.mean(axis=1)
    unit_sem = resp.std(axis=1) / np.sqrt(resp.shape[1])
    return unit_mean / (unit_sem + 1e-3)


def load_grb006():
    print("Loading GRB006...")
    _, trial_ts = load_grb006_aligned_trial_data()
    first_stim = trial_ts["first_stim_ts"].to_numpy(dtype=float)
    first_stim = first_stim[np.isfinite(first_stim)]
    unit_ids, spike_times = fetch_grb006_spike_times()

    trial_ts["trial_idx"] = np.arange(len(trial_ts))
    anchors = extract_conditioned_stim_anchors(trial_ts)
    paired_stat = anchors["paired_last_stationary"]
    paired_move = anchors["paired_first_movement"]
    print(f"  {len(unit_ids)} units, {len(paired_stat)} paired trials")
    return unit_ids, spike_times, paired_stat, paired_move


def load_grb058():
    print("Loading GRB058...")
    subject, session = "GRB058", "20260312_134952"
    st_per_unit = fetch_good_units(subject, session)
    align_ev = fetch_session_events(subject, session)
    trial_df = fetch_trial_metadata(subject, session, align_ev)
    trial_ts = build_trial_stim_classification(align_ev, trial_df)

    unit_ids = list(st_per_unit.keys())
    spike_times = list(st_per_unit.values())
    anchors = extract_conditioned_stim_anchors(trial_ts)
    paired_stat = anchors["paired_last_stationary"]
    paired_move = anchors["paired_first_movement"]
    print(f"  {len(unit_ids)} units, {len(paired_stat)} paired trials")
    return unit_ids, spike_times, paired_stat, paired_move


def compute_snrs(spike_times, paired_stat, paired_move):
    peth_s, _, bc = compute_population_peth(spike_times, paired_stat, **PETH_KWARGS)
    peth_m, _, _ = compute_population_peth(spike_times, paired_move, **PETH_KWARGS)
    snr_s = snr_per_unit(peth_s, bc)
    snr_m = snr_per_unit(peth_m, bc)
    snr_min = np.minimum(snr_s, snr_m)  # the binding constraint
    return snr_s, snr_m, snr_min


def plot_panel(ax, snr_s, snr_m, snr_min, label):
    bins = np.linspace(0, max(snr_s.max(), snr_m.max(), 20), 50)
    ax.hist(snr_s, bins=bins, alpha=0.45, color="steelblue", label="stat")
    ax.hist(snr_m, bins=bins, alpha=0.45, color="tomato", label="move")

    n_pass = int((snr_min >= LEGACY_SNR_THRESHOLD).sum())
    n_total = len(snr_min)
    n_drop = n_total - n_pass

    ax.axvline(
        LEGACY_SNR_THRESHOLD,
        color="black",
        linewidth=1.5,
        linestyle="--",
        label=f"legacy threshold = {LEGACY_SNR_THRESHOLD}",
    )
    ax.set_title(
        f"{label}\n{n_pass}/{n_total} pass  ({n_drop} dropped)",
        fontsize=10,
    )
    ax.set_xlabel("SNR  (mean / SEM in 30–120 ms window)", fontsize=9)
    ax.set_ylabel("Unit count", fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    ax.tick_params(labelsize=8)

    # Print the SNR values for units near the threshold
    near = snr_min[
        (snr_min >= LEGACY_SNR_THRESHOLD * 0.5) & (snr_min < LEGACY_SNR_THRESHOLD * 1.5)
    ]
    print(f"  {label}: {n_pass}/{n_total} pass SNR≥{LEGACY_SNR_THRESHOLD}")
    print(f"  Units near threshold (0.5–1.5×): n={len(near)}")
    if len(near):
        print(f"  SNR values: {np.sort(near)}")


print("Computing SNRs...")
unit_ids_6, st_6, stat_6, move_6 = load_grb006()
snr_s6, snr_m6, snr_min6 = compute_snrs(st_6, stat_6, move_6)

unit_ids_58, st_58, stat_58, move_58 = load_grb058()
snr_s58, snr_m58, snr_min58 = compute_snrs(st_58, stat_58, move_58)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
plot_panel(axes[0], snr_s6, snr_m6, snr_min6, "GRB006  20240821")
plot_panel(axes[1], snr_s58, snr_m58, snr_min58, "GRB058  20260312")
fig.suptitle(
    "Per-unit SNR distribution — historical locomotion gate diagnostic\n"
    "Dashed line = legacy SNR threshold (3.0); binding constraint = min(stat SNR, move SNR)",
    fontsize=10,
)

with PdfPages(OUT_PATH) as pdf:
    pdf.savefig(fig, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved → {OUT_PATH}")
