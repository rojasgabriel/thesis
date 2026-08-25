"""Test whether double-peaked stimulus responses reflect pulse onset and offset.

The supplemental figure shows discovery examples, the predicted change in peak
latency for 15 versus 30 ms pulses, and whether double-peak units cluster by
waveform duration or recording depth. Pulse-duration comparisons exclude trials
with another pulse inside the peak-search window. All spike times are
synchronized to the behavioral-event clock before alignment.
"""

import os
from collections.abc import Collection
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401
from spks.event_aligned import population_peth

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from labdata_plugin.schema import StimulusResponsiveness
from thesis.ephys.events import fetch_session_events
from thesis.ephys.units import fetch_unit_table
from thesis.plotting import separate_axes

DISCOVERY_SESSIONS = [
    ("GRB006", "20240821_121447"),
    ("GRB058", "20260224_152424"),
]
PULSE_WIDTH_SESSION = ("GRB058", "20260312_134952")
UNIT_CRITERIA_ID = 1
STABILITY_PARAM_ID = 0
RESPONSIVENESS_PARAM_ID = 0
PETH_PRE_SECONDS = 0.1
PETH_POST_SECONDS = 0.15
PETH_BINWIDTH_MS = 10
BASELINE_WINDOW = (-0.04, 0.0)
PEAK_SEARCH_WINDOW = (0.0, 0.12)
MIN_PROMINENCE_FRACTION = 0.25
MIN_PROMINENCE_SPIKES_PER_SECOND = 1.0
MIN_PEAK_DISTANCE_MS = 20.0
NARROW_BROAD_MS = 0.4
PULSE_COLORS = {15: "#0072B2", 30: "#D55E00"}
DISCOVERY_COLOR = "0.25"
GROUP_COLORS = ["0.7", "0.15"]
FIGURE_STYLE = {
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.fontsize": 6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

FIGURE_ROOT = Path(os.environ.get("THESIS_FIGURE_ROOT", "figures"))
OUT_PATH = FIGURE_ROOT / "double_peak" / "double_peak_supplement.pdf"


def window_maximum(
    peth: np.ndarray, bin_centers: np.ndarray, center: float
) -> tuple[float, float]:
    """Return the trial-mean maximum within 15 ms of an expected peak."""
    mean_peth = peth.mean(axis=0)
    mask = np.abs(bin_centers - center) <= 0.015
    index = np.flatnonzero(mask)[np.argmax(mean_peth[mask])]
    return float(bin_centers[index]), float(mean_peth[index])


def classify_double_peak_units(
    spike_times: list[np.ndarray],
    alignment_times: np.ndarray,
    unit_ids: list[int],
    excited_unit_ids: Collection[int],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[int]]:
    peth, bin_edges, _ = population_peth(
        all_spike_times=spike_times,
        alignment_times=alignment_times,
        pre_seconds=PETH_PRE_SECONDS,
        post_seconds=PETH_POST_SECONDS,
        binwidth_ms=PETH_BINWIDTH_MS,
    )
    peth = peth / (PETH_BINWIDTH_MS / 1000)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    excited_unit_set = set(excited_unit_ids)
    excited_indices = np.array(
        [
            index
            for index, unit_id in enumerate(unit_ids)
            if unit_id in excited_unit_set
        ],
        dtype=int,
    )
    excited_unit_ids = [unit_ids[index] for index in excited_indices]
    excited_peth = peth[excited_indices]
    if not excited_unit_ids:
        return (
            pd.DataFrame(
                {
                    "unit": [],
                    "n_peaks": [],
                    "peak_times": [],
                    "peak_heights": [],
                    "baseline": [],
                    "min_peak_height_above_baseline": [],
                    "max_peak_height_above_baseline": [],
                }
            ),
            peth,
            bin_centers,
            [],
        )
    peak_rows = StimulusResponsiveness.classify_components(
        excited_peth,
        bin_centers,
        excited_unit_ids,
        search_window=PEAK_SEARCH_WINDOW,
        baseline_window=BASELINE_WINDOW,
        min_prominence_fraction=MIN_PROMINENCE_FRACTION,
        min_prominence_spikes_per_second=MIN_PROMINENCE_SPIKES_PER_SECOND,
        min_peak_distance_ms=MIN_PEAK_DISTANCE_MS,
        binwidth_ms=PETH_BINWIDTH_MS,
    )

    double_peak_rows = []
    for _, peak_row in peak_rows.loc[peak_rows["n_peaks"] == 2].iterrows():
        unit_id = int(peak_row["unit"])
        excited_index = excited_unit_ids.index(unit_id)
        baseline_bins = (bin_centers >= BASELINE_WINDOW[0]) & (
            bin_centers < BASELINE_WINDOW[1]
        )
        baseline = float(excited_peth[excited_index].mean(axis=0)[baseline_bins].mean())
        peak_heights_above_baseline = [
            float(height - baseline) for height in peak_row["peak_heights"]
        ]
        if min(peak_heights_above_baseline) < 5.0:
            continue
        double_peak_row = peak_row.copy()
        double_peak_row["baseline"] = baseline
        double_peak_row["min_peak_height_above_baseline"] = min(
            peak_heights_above_baseline
        )
        double_peak_row["max_peak_height_above_baseline"] = max(
            peak_heights_above_baseline
        )
        double_peak_rows.append(double_peak_row)

    double_peak_rows = pd.DataFrame(double_peak_rows)
    if double_peak_rows.empty:
        double_peak_rows = peak_rows.iloc[0:0].reindex(
            columns=list(peak_rows.columns)
            + [
                "baseline",
                "min_peak_height_above_baseline",
                "max_peak_height_above_baseline",
            ]
        )
    return double_peak_rows, peth, bin_centers, excited_unit_ids


def plot_mean_sem_trace(ax, bin_centers, peth_trials, color) -> None:
    mean = peth_trials.mean(axis=0)
    sem = peth_trials.std(axis=0) / np.sqrt(peth_trials.shape[0])
    ax.plot(bin_centers, mean, color=color, linewidth=1.5)
    ax.fill_between(bin_centers, mean - sem, mean + sem, alpha=0.25, color=color)


def collect_session(subject: str, session: str) -> dict:
    units = fetch_unit_table(
        subject, session, UNIT_CRITERIA_ID, STABILITY_PARAM_ID
    ).reset_index(drop=True)
    excited_ids = StimulusResponsiveness.fetch_excited_unit_ids(
        subject,
        session,
        UNIT_CRITERIA_ID,
        RESPONSIVENESS_PARAM_ID,
        STABILITY_PARAM_ID,
    )
    _, pulses = fetch_session_events(subject, session)
    pulses = pulses.copy()
    pulses["next_pulse_delay_s"] = pulses["timestamp"].shift(-1) - pulses["timestamp"]
    first_pulses = pulses[pulses["first_in_train"]]
    mixed_widths = pulses["width_ms"].dropna().nunique() > 1
    alignment_pulses = first_pulses
    if mixed_widths:
        alignment_pulses = first_pulses[first_pulses["width_ms"].eq(15)]
    first_15ms = alignment_pulses["timestamp"].to_numpy(dtype=float)

    unit_ids = units["unit_id"].astype(int).tolist()
    spike_times = units["spike_times_s"].tolist()
    double_rows, peth_15, bin_centers, _ = classify_double_peak_units(
        spike_times, first_15ms, unit_ids, excited_ids
    )
    double_rows = double_rows.sort_values(
        "min_peak_height_above_baseline", ascending=False
    )
    double_ids = double_rows["unit"].astype(int).tolist()

    excited_units = units[units["unit_id"].isin(excited_ids)].copy()
    excited_units["is_double"] = excited_units["unit_id"].isin(double_ids)
    excited_units["subject"] = subject
    excited_units["session"] = session

    rows: dict[int, dict] = {}
    for uid in double_ids:
        peak_row = double_rows[double_rows["unit"] == uid].iloc[0]
        rows[uid] = {
            "uid": uid,
            "peth_15": peth_15[unit_ids.index(uid)],
            "peak_row_15": peak_row,
            "bin_centers": bin_centers,
            "n_15ms": len(first_15ms),
        }

    if mixed_widths and double_ids:
        isolated = first_pulses[
            first_pulses["next_pulse_delay_s"].isna()
            | first_pulses["next_pulse_delay_s"].gt(PEAK_SEARCH_WINDOW[1])
        ]
        selected_spikes = [spike_times[unit_ids.index(uid)] for uid in double_ids]
        for width_ms in (15, 30):
            alignment_times = isolated.loc[
                isolated["width_ms"].eq(width_ms), "timestamp"
            ].to_numpy(dtype=float)
            if not len(alignment_times):
                raise ValueError(
                    f"No isolated {width_ms} ms pulses for {subject} {session}"
                )
            peth, _, _ = population_peth(
                all_spike_times=selected_spikes,
                alignment_times=alignment_times,
                pre_seconds=PETH_PRE_SECONDS,
                post_seconds=PETH_POST_SECONDS,
                binwidth_ms=PETH_BINWIDTH_MS,
            )
            peth = peth / (PETH_BINWIDTH_MS / 1000)
            for index, uid in enumerate(double_ids):
                discovery_peaks = rows[uid]["peak_row_15"]["peak_times"]
                expected = [
                    discovery_peaks[0],
                    discovery_peaks[1] + (0.015 if width_ms == 30 else 0.0),
                ]
                maxima = [
                    window_maximum(peth[index], bin_centers, center)
                    for center in expected
                ]
                rows[uid].update(
                    {
                        f"peth_control_{width_ms}": peth[index],
                        f"peak_row_control_{width_ms}": {
                            "peak_times": [value[0] for value in maxima],
                            "peak_heights": [value[1] for value in maxima],
                        },
                        f"n_control_{width_ms}": len(alignment_times),
                    }
                )

    return {
        "subject": subject,
        "session": session,
        "n_stable": len(units),
        "units": excited_units,
        "rows": rows,
    }


def plot_response(
    ax, row: dict, show_control: bool = False, show_legend: bool = False
) -> None:
    bins = row["bin_centers"]
    peth_15 = row["peth_control_15"] if show_control else row["peth_15"]
    n_15ms = row["n_control_15"] if show_control else row["n_15ms"]
    color_15 = PULSE_COLORS[15] if show_control else DISCOVERY_COLOR
    plot_mean_sem_trace(ax, bins, peth_15, color_15)
    if show_control:
        plot_mean_sem_trace(
            ax,
            bins,
            row["peth_control_30"],
            PULSE_COLORS[30],
        )
        if show_legend:
            ax.text(
                0.98,
                0.96,
                f"15 ms (n={n_15ms})",
                color=PULSE_COLORS[15],
                ha="right",
                va="top",
                transform=ax.transAxes,
                fontsize=6,
            )
            ax.text(
                0.98,
                0.84,
                f"30 ms (n={row['n_control_30']})",
                color=PULSE_COLORS[30],
                ha="right",
                va="top",
                transform=ax.transAxes,
                fontsize=6,
            )
    ax.axvline(0, color="0.4", linestyle="--", linewidth=0.6)
    ax.set_xlabel("Time from onset (s)")
    ax.set_ylabel("Spikes s$^{-1}$")
    ax.set_ylim(bottom=0)


def plot_second_peak_prediction_error(ax, rows: list[dict]) -> None:
    """Plot observed minus onset-offset-predicted second-peak latency."""
    prediction_errors = []
    for row in rows:
        latency_15 = 1000 * row["peak_row_control_15"]["peak_times"][1]
        latency_30 = 1000 * row["peak_row_control_30"]["peak_times"][1]
        prediction_errors.append(latency_30 - (latency_15 + 15))
    x = np.linspace(-0.06, 0.06, len(prediction_errors))
    ax.scatter(x, prediction_errors, color="0.2", s=16, alpha=0.8, zorder=3)
    ax.axhline(0, color="0.6", linewidth=0.7)
    ax.set_xlim(-0.35, 0.35)
    ax.set_xticks([0], ["30 ms pulse"])
    ax.set_ylabel("Δ second-peak latency (ms)")
    ax.margins(y=0.2)


def plot_unit_property(ax, units: pd.DataFrame, column: str, ylabel: str) -> None:
    groups = [
        units.loc[units["is_double"] == is_double, column].to_numpy(float)
        for is_double in (False, True)
    ]
    boxes = ax.boxplot(
        groups,
        positions=[0, 1],
        widths=0.42,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 0.8},
        whiskerprops={"color": "0.35", "linewidth": 0.6},
        capprops={"color": "0.35", "linewidth": 0.6},
    )
    for patch, color in zip(boxes["boxes"], GROUP_COLORS):
        patch.set(facecolor=color, edgecolor=color, alpha=0.22, linewidth=0.8)
    rng = np.random.default_rng(0)
    for position, (values, color) in enumerate(zip(groups, GROUP_COLORS)):
        x = position + rng.uniform(-0.12, 0.12, len(values))
        ax.scatter(x, values, s=8, alpha=0.8, color=color, rasterized=True)
    ax.set_xticks([0, 1], ["Other\nexcited", "Double\npeak"])
    ax.set_ylabel(ylabel)


def make_figure(discovery_data: list[dict], pulse_width_data: dict):
    discovery_rows = [
        row for data in discovery_data for row in list(data["rows"].values())[:2]
    ]
    pulse_rows = [
        row for row in pulse_width_data["rows"].values() if "peth_control_30" in row
    ]
    all_units = pd.concat([data["units"] for data in discovery_data], ignore_index=True)

    with plt.style.context("nature"), plt.rc_context(FIGURE_STYLE):
        fig = plt.figure(figsize=(7.2, 5.8))
        grid = fig.add_gridspec(3, 1, height_ratios=[1, 0.9, 1.05], hspace=0.7)

        discovery_grid = grid[0].subgridspec(1, 4, wspace=0.55)
        discovery_axes = [
            fig.add_subplot(discovery_grid[0, index]) for index in range(4)
        ]
        for ax, row in zip(discovery_axes, discovery_rows):
            plot_response(ax, row)
        for ax in discovery_axes[len(discovery_rows) :]:
            ax.axis("off")

        properties = grid[1].subgridspec(1, 2, wspace=0.4)
        duration_ax = fig.add_subplot(properties[0, 0])
        plot_unit_property(
            duration_ax, all_units, "spike_duration_ms", "Spike duration (ms)"
        )
        duration_ax.axhline(NARROW_BROAD_MS, color="0.4", linestyle="--", linewidth=0.6)

        depth_ax = fig.add_subplot(properties[0, 1])
        plot_unit_property(depth_ax, all_units, "depth", "Depth from probe tip (µm)")

        pulse_grid = grid[2].subgridspec(1, 3, wspace=0.65)
        pulse_axes = [fig.add_subplot(pulse_grid[0, index]) for index in range(2)]
        for index, (ax, row) in enumerate(zip(pulse_axes, pulse_rows)):
            plot_response(ax, row, show_control=True, show_legend=index == 0)
        for ax in pulse_axes[len(pulse_rows) :]:
            ax.axis("off")

        prediction_error_ax = fig.add_subplot(pulse_grid[0, 2])
        plot_second_peak_prediction_error(prediction_error_ax, pulse_rows)

        for label, ax in zip(
            "abcde",
            [
                discovery_axes[0],
                duration_ax,
                depth_ax,
                pulse_axes[0],
                prediction_error_ax,
            ],
        ):
            ax.text(
                -0.25,
                1.08,
                label,
                transform=ax.transAxes,
                fontweight="bold",
                fontsize=9,
                va="top",
            )
        fig.subplots_adjust(top=0.95, bottom=0.09, left=0.09, right=0.97)
        for ax in fig.axes:
            if ax.axison:
                separate_axes(ax)
    return fig


def main() -> None:
    discovery_data = [
        collect_session(subject, session) for subject, session in DISCOVERY_SESSIONS
    ]
    pulse_width_data = collect_session(*PULSE_WIDTH_SESSION)
    session_data = [*discovery_data, pulse_width_data]
    for data in session_data:
        print(
            f"{data['subject']} {data['session']}: "
            f"{len(data['rows'])}/{len(data['units'])} stable excited units are double peak"
        )
        if data["rows"] and "n_control_30" in next(iter(data["rows"].values())):
            row = next(iter(data["rows"].values()))
            print(
                "Pulse-duration control: "
                f"{row['n_control_15']} isolated 15 ms trials, "
                f"{row['n_control_30']} isolated 30 ms trials"
            )
    fig = make_figure(discovery_data, pulse_width_data)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Figure saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
