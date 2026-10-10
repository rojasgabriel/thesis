"""Test whether double-peaked visual-pulse responses reflect pulse onset and offset.

This supplemental figure shows discovery examples (a), spike duration (b) and
depth (c) of double-peak units against other excited units, 15 and 30 ms pulse
responses (d), and the second-peak latency error for 30 ms pulses (e).

Sessions: discovery uses GRB006 20240821_121447 and GRB058 20260224_152424. The
pulse-width control uses GRB058 20260312_134952. The double-peak test runs
separately in each session.

Unit selection: units pass quality criterion 1, stability parameter 0, and the
stored excited classification (stimulus-response parameter 0). A double-peak
unit has exactly two trial-mean peaks from 0 to 120 ms. The peaks are at least
20 ms apart. Each peak has a prominence of at least 25% of the maximum
baseline-subtracted response, and at least 1 sp/s. Each peak is also at least 5
sp/s above baseline.

Baseline: the trial-mean rate from -40 to 0 ms.

Event alignment: spikes align to the onsets of first-in-train pulses. A
first-in-train pulse is the first pulse after a gap of more than 1 s. The PETH
uses 10 ms bins from -100 to 150 ms. Pulse widths are 15 or 30 ms within ±2 ms.
Other widths are unknown. The double-peak test uses only first pulses with a 15
ms width. If every stored pulse has zero duration (onset-only events, as in
repaired GRB006 data), the code sets all widths to 15 ms. In that case, the
width is not measured. labdata maps spike times to the event clock through
StreamSync. If no StreamSync row exists, labdata uses the sampling rate and
gives a warning.

Pulse-width control: the code uses first-in-train 15 and 30 ms pulses with no
other pulse in the next 120 ms. The control second peak is the trial-mean
maximum within 15 ms of the unit's double-peak-test second-peak time in the same
session. For 30 ms pulses, the window moves 15 ms later. This is the
onset-offset prediction. Panel e shows the 30 ms second-peak latency minus the
predicted latency (the 15 ms second-peak latency plus 15 ms).

Exclusions: first pulses of unknown width (counted and printed per session).
The double-peak test also excludes 30 ms first pulses. The control excludes
pulses with another pulse in the next 120 ms.

Sampling unit: the trial for the mean ± SEM traces in a and d. Panels a and d
show the first two double-peak units per session, ranked by the smaller peak
height above baseline. The unit for b, c, and e. Panels b and c pool the two
discovery sessions.
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
STIM_RESPONSE_PARAM_ID = 0
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


def _window_maximum(peth: np.ndarray, bin_centers: np.ndarray, center: float) -> float:
    """Return the time of the trial-mean maximum within 15 ms of an expected peak."""
    mean_peth = peth.mean(axis=0)
    mask = np.abs(bin_centers - center) <= 0.015
    return float(bin_centers[np.flatnonzero(mask)[np.argmax(mean_peth[mask])]])


def classify_double_peak_units(
    spike_times: list[np.ndarray],
    alignment_times: np.ndarray,
    unit_ids: list[int],
    excited_unit_ids: Collection[int],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    from labdata_plugin.schema import StimulusResponse

    peth, bin_edges, _ = population_peth(
        all_spike_times=spike_times,
        alignment_times=alignment_times,
        pre_seconds=PETH_PRE_SECONDS,
        post_seconds=PETH_POST_SECONDS,
        binwidth_ms=PETH_BINWIDTH_MS,
    )
    peth = peth / (PETH_BINWIDTH_MS / 1000)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    # classify_components works per unit, so classifying every unit and keeping
    # the excited ones gives the same rows as classifying excited units only.
    peak_rows = StimulusResponse.classify_components(
        peth,
        bin_centers,
        unit_ids,
        search_window=PEAK_SEARCH_WINDOW,
        baseline_window=BASELINE_WINDOW,
        min_prominence_fraction=MIN_PROMINENCE_FRACTION,
        min_prominence_spikes_per_second=MIN_PROMINENCE_SPIKES_PER_SECOND,
        min_peak_distance_ms=MIN_PEAK_DISTANCE_MS,
        binwidth_ms=PETH_BINWIDTH_MS,
    ).rename(
        columns={
            "unit_id": "unit",
            "n_components": "n_peaks",
            "component_latencies_s": "peak_times",
            "component_rates": "peak_heights",
        }
    )

    baseline_bins = (bin_centers >= BASELINE_WINDOW[0]) & (
        bin_centers < BASELINE_WINDOW[1]
    )
    candidates = peak_rows["n_peaks"].eq(2) & peak_rows["unit"].isin(excited_unit_ids)
    double_peak_rows = []
    for _, peak_row in peak_rows.loc[candidates].iterrows():
        unit_index = unit_ids.index(int(peak_row["unit"]))
        baseline = float(peth[unit_index].mean(axis=0)[baseline_bins].mean())
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
        double_peak_rows.append(double_peak_row)

    double_peak_rows = pd.DataFrame(double_peak_rows)
    if double_peak_rows.empty:
        double_peak_rows = peak_rows.iloc[0:0].reindex(
            columns=list(peak_rows.columns)
            + ["baseline", "min_peak_height_above_baseline"]
        )
    return double_peak_rows, peth, bin_centers


def _plot_mean_sem_trace(ax, bin_centers, peth_trials, color) -> None:
    mean = peth_trials.mean(axis=0)
    sem = peth_trials.std(axis=0) / np.sqrt(peth_trials.shape[0])
    ax.plot(bin_centers, mean, color=color, linewidth=1.5)
    ax.fill_between(bin_centers, mean - sem, mean + sem, alpha=0.25, color=color)


def _collect_session(subject: str, session: str, *, controls: bool = True) -> dict:
    from labdata_plugin.schema import StimulusResponse

    units = fetch_unit_table(
        subject, session, UNIT_CRITERIA_ID, STABILITY_PARAM_ID
    ).reset_index(drop=True)
    response_query = {
        "subject_name": subject,
        "session_name": session,
        "unit_criteria_id": UNIT_CRITERIA_ID,
        "stim_response_param_id": STIM_RESPONSE_PARAM_ID,
        "response_type": "excited",
    }
    excited_ids = set((StimulusResponse.Unit & response_query).fetch("unit_id"))
    _, pulses = fetch_session_events(subject, session)
    pulses = pulses.copy()
    pulses["next_pulse_delay_s"] = pulses["timestamp"].shift(-1) - pulses["timestamp"]
    first_pulses = pulses[pulses["first_in_train"]]
    mixed_widths = pulses["width_ms"].dropna().nunique() > 1
    first_15ms = first_pulses.loc[
        first_pulses["width_ms"].eq(15), "timestamp"
    ].to_numpy(dtype=float)
    if not first_15ms.size:
        raise ValueError(f"No first 15 ms pulses for {subject} {session}")
    unknown = int(first_pulses["width_ms"].isna().sum())
    if unknown:
        print(f"{subject} {session}: excluded {unknown} first pulses of unknown width")

    unit_ids = units["unit_id"].astype(int).tolist()
    spike_times = units["spike_times_s"].tolist()
    double_peak_table, peth_15, bin_centers = classify_double_peak_units(
        spike_times, first_15ms, unit_ids, excited_ids
    )
    double_peak_table = double_peak_table.sort_values(
        "min_peak_height_above_baseline", ascending=False
    )
    double_peak_unit_ids = double_peak_table["unit"].astype(int).tolist()

    excited_units = units[units["unit_id"].isin(excited_ids)].copy()
    excited_units["is_double"] = excited_units["unit_id"].isin(double_peak_unit_ids)

    unit_responses: dict[int, dict] = {}
    for unit_id in double_peak_unit_ids:
        peak_row = double_peak_table[double_peak_table["unit"] == unit_id].iloc[0]
        unit_responses[unit_id] = {
            "unit_id": unit_id,
            "peth_15": peth_15[unit_ids.index(unit_id)],
            "peak_row_15": peak_row,
            "bin_centers": bin_centers,
            "n_15ms": len(first_15ms),
        }

    if controls and mixed_widths and double_peak_unit_ids:
        isolated = first_pulses[
            first_pulses["next_pulse_delay_s"].isna()
            | first_pulses["next_pulse_delay_s"].gt(PEAK_SEARCH_WINDOW[1])
        ]
        selected_spikes = [
            spike_times[unit_ids.index(unit_id)] for unit_id in double_peak_unit_ids
        ]
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
            for index, unit_id in enumerate(double_peak_unit_ids):
                second_peak = unit_responses[unit_id]["peak_row_15"]["peak_times"][1]
                second_peak += 0.015 if width_ms == 30 else 0.0
                unit_responses[unit_id].update(
                    {
                        f"peth_control_{width_ms}": peth[index],
                        f"second_peak_control_{width_ms}": _window_maximum(
                            peth[index], bin_centers, second_peak
                        ),
                        f"n_control_{width_ms}": len(alignment_times),
                    }
                )

    return {
        "subject": subject,
        "session": session,
        "units": excited_units,
        "unit_responses": unit_responses,
    }


def _plot_response(
    ax, unit_response: dict, show_control: bool = False, show_legend: bool = False
) -> None:
    bins = unit_response["bin_centers"]
    if not show_control:
        _plot_mean_sem_trace(ax, bins, unit_response["peth_15"], DISCOVERY_COLOR)
    else:
        for width_ms, label_y in ((15, 0.96), (30, 0.84)):
            color = PULSE_COLORS[width_ms]
            _plot_mean_sem_trace(
                ax, bins, unit_response[f"peth_control_{width_ms}"], color
            )
            if show_legend:
                ax.text(
                    0.98,
                    label_y,
                    f"{width_ms} ms (n={unit_response[f'n_control_{width_ms}']})",
                    color=color,
                    ha="right",
                    va="top",
                    transform=ax.transAxes,
                    fontsize=6,
                )
    ax.axvline(0, color="0.4", linestyle="--", linewidth=0.6)
    ax.set_xlabel("time from onset (s)")
    ax.set_ylabel("spikes s$^{-1}$")
    ax.set_ylim(bottom=0)


def _plot_second_peak_prediction_error(ax, unit_responses: list[dict]) -> None:
    """Plot observed minus onset-offset-predicted second-peak latency."""
    prediction_errors = []
    for unit_response in unit_responses:
        latency_15 = 1000 * unit_response["second_peak_control_15"]
        latency_30 = 1000 * unit_response["second_peak_control_30"]
        prediction_errors.append(latency_30 - (latency_15 + 15))
    x = np.linspace(-0.06, 0.06, len(prediction_errors))
    ax.scatter(x, prediction_errors, color="0.2", s=16, alpha=0.8, zorder=3)
    ax.axhline(0, color="0.6", linewidth=0.7)
    ax.set_xlim(-0.35, 0.35)
    ax.set_xticks([0], ["30 ms pulse"])
    ax.set_ylabel("Δ second-peak\nlatency (ms)")
    ax.margins(y=0.2)


def _plot_unit_property(ax, units: pd.DataFrame, column: str, ylabel: str) -> None:
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
    ax.set_xticks([0, 1], ["other\nexcited", "double\npeak"])
    ax.set_ylabel(ylabel)


def _make_figure(discovery_data: list[dict], pulse_width_data: dict):
    discovery_responses = [
        unit_response
        for session_result in discovery_data
        for unit_response in list(session_result["unit_responses"].values())[:2]
    ]
    pulse_width_responses = [
        unit_response
        for unit_response in pulse_width_data["unit_responses"].values()
        if "peth_control_30" in unit_response
    ]
    all_units = pd.concat(
        [session_result["units"] for session_result in discovery_data],
        ignore_index=True,
    )

    with plt.style.context("nature"), plt.rc_context(FIGURE_STYLE):
        fig = plt.figure(figsize=(7.2, 5.8))
        grid = fig.add_gridspec(3, 1, height_ratios=[1, 0.9, 1.05], hspace=0.7)

        discovery_grid = grid[0].subgridspec(1, 4, wspace=0.55)
        discovery_axes = [
            fig.add_subplot(discovery_grid[0, index]) for index in range(4)
        ]
        for ax, unit_response in zip(discovery_axes, discovery_responses):
            _plot_response(ax, unit_response)
        for ax in discovery_axes[len(discovery_responses) :]:
            ax.axis("off")

        properties = grid[1].subgridspec(1, 2, wspace=0.4)
        duration_ax = fig.add_subplot(properties[0, 0])
        _plot_unit_property(
            duration_ax, all_units, "spike_duration_ms", "spike duration (ms)"
        )
        duration_ax.axhline(NARROW_BROAD_MS, color="0.4", linestyle="--", linewidth=0.6)

        depth_ax = fig.add_subplot(properties[0, 1])
        _plot_unit_property(depth_ax, all_units, "depth", "depth from probe tip (µm)")

        pulse_grid = grid[2].subgridspec(1, 3, wspace=0.65)
        pulse_axes = [fig.add_subplot(pulse_grid[0, index]) for index in range(2)]
        for index, (ax, unit_response) in enumerate(
            zip(pulse_axes, pulse_width_responses)
        ):
            _plot_response(ax, unit_response, show_control=True, show_legend=index == 0)
        for ax in pulse_axes[len(pulse_width_responses) :]:
            ax.axis("off")

        prediction_error_ax = fig.add_subplot(pulse_grid[0, 2])
        _plot_second_peak_prediction_error(prediction_error_ax, pulse_width_responses)

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
        _collect_session(subject, session) for subject, session in DISCOVERY_SESSIONS
    ]
    pulse_width_data = _collect_session(*PULSE_WIDTH_SESSION)
    session_data = [*discovery_data, pulse_width_data]
    for session_result in session_data:
        print(
            f"{session_result['subject']} {session_result['session']}: "
            f"{len(session_result['unit_responses'])}/"
            f"{len(session_result['units'])} stable excited units are double peak"
        )
        unit_responses = session_result["unit_responses"]
        if unit_responses and "n_control_30" in next(iter(unit_responses.values())):
            unit_response = next(iter(unit_responses.values()))
            print(
                "Pulse-duration control: "
                f"{unit_response['n_control_15']} isolated 15 ms trials, "
                f"{unit_response['n_control_30']} isolated 30 ms trials"
            )
    fig = _make_figure(discovery_data, pulse_width_data)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Figure saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
