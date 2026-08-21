#!/usr/bin/env python
# %% Import libraries
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from thesis.ephys.utils.analysis_peth import compute_population_peth
from thesis.ephys.utils.analysis_selectivity import compute_unit_selectivity
from thesis.ephys.utils.grb006_data import (
    fetch_grb006_spike_times,
    load_grb006_aligned_trial_data,
)

plt.rcParams["text.usetex"] = False
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.sans-serif"] = "Arial"
plt.rcParams["font.size"] = 12
plt.rcParams["figure.dpi"] = 100

# %% Load data
animal = "GRB006"
session = "20240821_121447"

_, trial_ts = load_grb006_aligned_trial_data()
unit_ids, spike_times = fetch_grb006_spike_times()
spike_times_per_unit: list[np.ndarray] = list(spike_times)


# %% Compute PETHs
trial_ts = trial_ts[
    trial_ts["stationary_stims"].apply(lambda x: len(x) > 0)
    & trial_ts["movement_stims"].apply(lambda x: len(x) > 0)
    & trial_ts["center_port_entries"].apply(lambda x: len(x) > 0)
].copy()

first_stim_ts = trial_ts["first_stim_ts"].to_numpy(dtype=float).copy()

first_stim_peth, bin_edges, bin_centers = compute_population_peth(
    spike_times_per_unit=spike_times_per_unit,
    alignment_times=first_stim_ts,
    pre_seconds=0.1,
    post_seconds=0.15,
    binwidth_ms=10,
    t_rise=0.001,
    t_decay=0.025,
)

n_units, n_trials, n_timepoints = first_stim_peth.shape

# %% Selectivity: baseline vs response (simple + robust)
# Assumptions:
# - Baseline: -0.1 to 0.0 s relative to event
# - Response: +0.04 to +0.1 s relative to event
# - Test: paired within-trial baseline vs response per unit

# Compute selectivity with default windows and Wilcoxon test
results_df, masks = compute_unit_selectivity(
    first_stim_peth,
    bin_edges,
    unit_ids=unit_ids,
    base_window=(-0.1, 0.0),
    resp_window=(0.04, 0.10),
    test="wilcoxon",
    alpha=0.05,
)

print(
    f"Units selective for first stimulus: {masks['selective'].sum()} / {n_units} "
    f"({masks['excited'].sum()} excited, {masks['suppressed'].sum()} suppressed)"
)

# %% Optional quick looks
fig, ax = plt.subplots(1, 2, figsize=(8, 3))
sns.histplot(results_df["delta"], bins=31, ax=ax[0], color="steelblue")
ax[0].set(
    title="Mean(resp - base) across units", xlabel="Delta FR (spk/s)", ylabel="Count"
)

sns.scatterplot(
    data=results_df,
    x="cohen_d",
    y=-np.log10(np.clip(results_df["q"], 1e-12, 1.0)),
    hue="selective",
    palette={True: "crimson", False: "gray"},
    ax=ax[1],
    s=20,
)
ax[1].set(title="Volcano (effect vs -log10 q)", xlabel="Cohen's d", ylabel="-log10(q)")
ax[1].legend(title="Selective", loc="best")
plt.tight_layout()


# %% Example raster/PSTH for a top selective unit (if available)
if masks["selective"].any():
    # Choose the most selective by q then |delta|
    top_idx = results_df.sort_values(["q", "delta"], ascending=[True, False]).iloc[0][
        "unit"
    ]

    mean_psth = first_stim_peth[top_idx].mean(axis=0)

    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    ax.plot(bin_centers, mean_psth, color="black")
    ax.axvspan(-0.1, 0.0, color="tab:blue", alpha=0.15, label="Baseline")
    ax.axvspan(0.04, 0.10, color="tab:orange", alpha=0.15, label="Response")
    ax.axvline(0, color="red", linestyle="--", linewidth=1, label="Stimulus")
    ax.set(
        xlabel="Time (s)",
        ylabel="Firing rate (spk/s)",
        title=f"Unit {int(top_idx)}: First-stim PSTH",
    )
    ax.legend(loc="best")
    plt.tight_layout()

# %%
