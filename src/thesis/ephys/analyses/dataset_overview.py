"""Show the raw spikes of the GLM sessions, before any model.

Question: what do single trials, single neurons and the whole population look
like around the flashes and the task events?

Comparison: none; this is a description of the data. Each figure has both mice
on the same scales.

- First flash (`dataset_first_flash.svg`): one example flash-excited neuron
  per mouse, spikes around the first flash of each trial (`FLASH_WINDOW_S`);
  below, the rate in 10-ms bins, mean ± SEM over trials. Only trials with the
  whole window inside the port: the mouse enters the port 24 ms or more before
  the first flash, so the trials with an entry less than 50 ms before it are
  left out. Later flashes of the train can fall in the window.
- First-flash population (`dataset_first_flash_population.svg`): every unit
  around the first flash of the same trials, the same measure as the flash
  population below.
- Flash raster (`dataset_flash_raster.svg`): the same neuron, spikes around
  every flash at least `LONG_GAP_S` after the flash before (not the first
  flash of a trial), in the port (still, blue) and between the center exit and
  the response-port entry (moving, orange), with the whole window
  (`FLASH_WINDOW_S`) inside its period; below, the rate in 10-ms bins,
  mean ± SEM over flashes.
- Population (`dataset_population.svg`): every unit around the first flash
  and around the center exit, 40-ms bins (one flash slot), as log2 of the rate over the unit's
  mean rate; units sorted by their mean from 150 to 500 ms after the first
  flash.
- Flash population (`dataset_flash_population.svg`): every unit around the
  in-port flashes of the flash raster, 10-ms bins, the rate minus the rate in
  the 40 ms before the flash, over the unit's mean rate; units sorted by their
  mean from 30 to 120 ms.

Units: the GLM units (`fetch_unit_table` with criteria 1 and stability 0,
without `fit.excluded_units`); flash-excited units from
`raw_video_trust_classification.json`. Example neuron: among the units that
are flash-excited by that table and by the flash weights of the GLM
(`flash_weight_check.json`), with a mean rate of at least
`MIN_EXAMPLE_RATE_HZ`, the unit whose response to in-port flashes is easiest to
see in single trials: (after − before) / √(after + before), the spikes per
flash in `PEAK_WINDOW_S` and those expected from `BASELINE_WINDOW_S`. Trials: the completed trials of the GLM. Mean rate
of a unit: its rate from 0.5 s before the first flash to 1.5 s after it, over
these trials. Display limits of the heatmaps: values past them take the end
color; none is removed.

    THESIS_CACHE_DIR=figures/cache uv run python -m \\
        thesis.ephys.analyses.dataset_overview
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thesis.ephys.analyses.glm import dmat
from thesis.ephys.analyses.glm.fit import analysis_unit_indices
from thesis.ephys.analyses.glm.flash_kernels import SESSIONS
from thesis.ephys.units import fetch_unit_table
from thesis.plotting import (
    CONCEPT_COLORS,
    FIGURE_STYLE,
    OBSERVED_COLOR,
    PERIOD_COLORS,
    label_panels,
    save_figure,
)

GLM = Path("results/glm")
OUT = Path("results/dataset_overview")
VISUAL = CONCEPT_COLORS["visual"]
STILL, MOVING = PERIOD_COLORS["in port"], PERIOD_COLORS["after exit"]
LONG_GAP_S = 0.08
MIN_EXAMPLE_RATE_HZ = 3.0
MEAN_WINDOW_S = (-0.5, 1.5)
FIRST_FLASH_WINDOW_S = (-0.3, 0.6)
EXIT_WINDOW_S = (-0.4, 0.6)
FLASH_WINDOW_S = (-0.05, 0.15)
SORT_WINDOW_S = (0.15, 0.5)
PEAK_WINDOW_S = (0.03, 0.12)
BASELINE_WINDOW_S = (-0.04, 0.0)
LOG2_LIMIT = 1.5
POPULATION_BIN_S = 0.04  # one flash slot
FLASH_LIMIT = 1.5


def counts_around(
    spikes: np.ndarray, events: np.ndarray, edges: np.ndarray
) -> np.ndarray:
    """Return spike counts (event, bin) in the bins `edges` (s) around each event."""
    index = np.searchsorted(spikes, events[:, None] + edges[None, :])
    return np.diff(index, axis=1)


def session_data(session: str, excited: set[int]) -> dict:
    subject, stamp = session.split("_", 1)
    prepared = dmat._load_windows(GLM / session / "stimulus_windows_me.npz")
    trials = dmat._load_selected_trials(prepared)
    units = fetch_unit_table(subject, stamp, 1, 0, include_metrics=False)
    units = units.iloc[analysis_unit_indices(units, "all")].reset_index(drop=True)
    first = np.array([float(t[0]) for t in trials["stim_pulse_times_s"]])
    exit_s = trials["center_exit_s"].to_numpy(float)
    response = trials["response_port_entry_s"].to_numpy(float)
    entry = trials["center_entry_s"].to_numpy(float)
    still, moving = [], []
    for times, widths, en, ex, rp in zip(
        trials["stim_pulse_times_s"],
        trials["stim_pulse_widths_ms"],
        entry,
        exit_s,
        response,
        strict=True,
    ):
        times = np.asarray(times, dtype=float)
        gaps = np.diff(times, prepend=-np.inf)
        long = (gaps >= LONG_GAP_S) & np.isfinite(gaps)
        long &= np.asarray(widths, dtype=float) == 15  # flashes only
        # The whole flash window lies inside its period.
        start, stop = times + FLASH_WINDOW_S[0], times + FLASH_WINDOW_S[1]
        still.extend(times[long & (start >= en) & (stop <= ex)])
        moving.extend(times[long & (start >= ex) & (stop <= rp)])
    still, moving = np.asarray(still), np.asarray(moving)
    spikes = [np.sort(np.asarray(s, dtype=float)) for s in units["spike_times_s"]]
    mean_rate = np.array(
        [
            counts_around(s, first, np.asarray(MEAN_WINDOW_S)).mean()
            / np.ptp(MEAN_WINDOW_S)
            for s in spikes
        ]
    )
    flash_edges = np.arange(FLASH_WINDOW_S[0], FLASH_WINDOW_S[1] + 1e-9, 0.01)
    flash_centers = (flash_edges[:-1] + flash_edges[1:]) / 2
    base = (flash_centers >= BASELINE_WINDOW_S[0]) & (
        flash_centers < BASELINE_WINDOW_S[1]
    )
    peak = (flash_centers >= PEAK_WINDOW_S[0]) & (flash_centers < PEAK_WINDOW_S[1])

    def rate_change(events):
        rate = np.array(
            [counts_around(s, events, flash_edges).mean(0) / 0.01 for s in spikes]
        )
        change = (rate - rate[:, base].mean(1, keepdims=True)) / np.maximum(
            mean_rate[:, None], 1e-9
        )
        return rate, change

    flash_rate, flash_response = rate_change(still)
    # The whole window of the first flash inside the port.
    first_in_port = first[
        (first + FLASH_WINDOW_S[0] >= entry) & (first + FLASH_WINDOW_S[1] <= exit_s)
    ]
    _, first_response = rate_change(first_in_port)
    ids = units["unit_id"].astype(int).to_numpy()
    candidates = [
        i
        for i in range(len(ids))
        if ids[i] in excited and mean_rate[i] >= MIN_EXAMPLE_RATE_HZ
    ]
    # Spikes per flash in the response window, and those expected from the rate
    # before the flash; their discriminability for Poisson counts.
    duration = np.ptp(PEAK_WINDOW_S)
    after = flash_rate[:, peak].mean(1) * duration
    before = flash_rate[:, base].mean(1) * duration
    visible = (after - before) / np.sqrt(np.maximum(after + before, 1e-9))
    example = max(candidates, key=lambda i: visible[i])
    return {
        "subject": subject,
        "units": units,
        "spikes": spikes,
        "ids": ids,
        "first": first,
        "first_in_port": first_in_port,
        "exit": exit_s,
        "still": still,
        "moving": moving,
        "mean_rate": mean_rate,
        "flash_centers": flash_centers,
        "flash_edges": flash_edges,
        "flash_response": flash_response,
        "first_response": first_response,
        "peak": peak,
        "example": example,
    }


def plot_flash_raster(sessions: list[dict]) -> Path:
    edges = np.arange(FLASH_WINDOW_S[0], FLASH_WINDOW_S[1] + 1e-9, 0.01)
    centers = 1000 * (edges[:-1] + edges[1:]) / 2
    with plt.rc_context(FIGURE_STYLE):
        figure, axes = plt.subplots(
            2,
            len(sessions),
            figsize=(7.0, 5.2),
            sharex=True,
            gridspec_kw={"height_ratios": (2, 1)},
        )
        for column, data in enumerate(sessions):
            spikes = data["spikes"][data["example"]]
            raster, psth = axes[0, column], axes[1, column]
            offset = 0
            for events, color in ((data["still"], STILL), (data["moving"], MOVING)):
                for row, event in enumerate(events):
                    s = (
                        spikes[
                            (spikes >= event + FLASH_WINDOW_S[0])
                            & (spikes < event + FLASH_WINDOW_S[1])
                        ]
                        - event
                    )
                    raster.plot(
                        1000 * s,
                        np.full(len(s), offset + row),
                        "|",
                        color=color,
                        markersize=1.2,
                    )
                offset += len(events)
                counts = counts_around(spikes, events, edges) / 0.01
                mean, sem = counts.mean(0), counts.std(0, ddof=1) / np.sqrt(len(events))
                psth.plot(centers, mean, color=color)
                psth.fill_between(
                    centers, mean - sem, mean + sem, color=color, alpha=0.2, linewidth=0
                )
            raster.set_ylim(0, offset)
            raster.set_ylabel("flashes")
            raster.axvline(0, color="0.6", linestyle="--", linewidth=0.7)
            psth.axvline(0, color="0.6", linestyle="--", linewidth=0.7)
            raster.text(
                0,
                1.02,
                f"{data['subject']}: one flash-excited neuron",
                transform=raster.transAxes,
                va="bottom",
            )
            # Each label right of the plot, at the middle of its block of rows.
            n_still = len(data["still"])
            for y, name, color in (
                (n_still / 2, "still", STILL),
                ((n_still + offset) / 2, "moving", MOVING),
            ):
                raster.annotate(
                    name,
                    (1.01, y),
                    xycoords=("axes fraction", "data"),
                    color=color,
                    va="center",
                    rotation=90,
                )
            psth.set_xlabel("time from flash (ms)")
            if column == 0:
                psth.set_ylabel("rate (spikes/s)")
        axes[1, 1].sharey(axes[1, 0])  # one rate scale for both mice
        axes[1, 0].autoscale()
        label_panels(axes.ravel(), x=-0.18, y=1.04)
        figure.tight_layout()
        return save_figure(figure, OUT / "dataset_flash_raster")


def plot_first_flash(sessions: list[dict]) -> Path:
    edges = np.arange(FLASH_WINDOW_S[0], FLASH_WINDOW_S[1] + 1e-9, 0.01)
    centers = 1000 * (edges[:-1] + edges[1:]) / 2
    with plt.rc_context(FIGURE_STYLE):
        figure, axes = plt.subplots(
            2,
            len(sessions),
            figsize=(7.0, 5.2),
            sharex=True,
            gridspec_kw={"height_ratios": (2, 1)},
        )
        for column, data in enumerate(sessions):
            spikes = data["spikes"][data["example"]]
            raster, psth = axes[0, column], axes[1, column]
            for row, event in enumerate(data["first_in_port"]):
                s = spikes[
                    (spikes >= event + FLASH_WINDOW_S[0])
                    & (spikes < event + FLASH_WINDOW_S[1])
                ]
                raster.plot(
                    1000 * (s - event),
                    np.full(len(s), row),
                    "|",
                    color=OBSERVED_COLOR,
                    markersize=1.2,
                )
            counts = counts_around(spikes, data["first_in_port"], edges) / 0.01
            mean = counts.mean(0)
            sem = counts.std(0, ddof=1) / np.sqrt(len(counts))
            psth.plot(centers, mean, color=OBSERVED_COLOR)
            psth.fill_between(
                centers,
                mean - sem,
                mean + sem,
                color=OBSERVED_COLOR,
                alpha=0.2,
                linewidth=0,
            )
            raster.set_ylim(0, len(data["first_in_port"]))
            raster.set_ylabel("trials")
            for axis in (raster, psth):
                axis.axvline(0, color=VISUAL, linewidth=0.8)
            raster.text(
                0,
                1.02,
                f"{data['subject']}: one flash-excited neuron",
                transform=raster.transAxes,
                va="bottom",
            )
            psth.set_xlabel("time from the first flash (ms)")
            if column == 0:
                psth.set_ylabel("rate (spikes/s)")
        axes[1, 1].sharey(axes[1, 0])  # one rate scale for both mice
        axes[1, 0].autoscale()
        label_panels(axes.ravel(), x=-0.18, y=1.04)
        figure.tight_layout()
        return save_figure(figure, OUT / "dataset_first_flash")


def _heat(axis, values, extent, limit, cmap):
    return axis.imshow(
        values,
        aspect="auto",
        cmap=cmap,
        vmin=-limit,
        vmax=limit,
        extent=extent,
        interpolation="nearest",
    )


def plot_population(sessions: list[dict]) -> tuple[Path, dict]:
    report = {}
    first_edges = np.arange(
        FIRST_FLASH_WINDOW_S[0], FIRST_FLASH_WINDOW_S[1] + 1e-9, POPULATION_BIN_S
    )
    exit_edges = np.arange(EXIT_WINDOW_S[0], EXIT_WINDOW_S[1] + 1e-9, POPULATION_BIN_S)
    first_centers = (first_edges[:-1] + first_edges[1:]) / 2
    sort = (first_centers >= SORT_WINDOW_S[0]) & (first_centers < SORT_WINDOW_S[1])
    heights = [len(d["ids"]) for d in sessions]
    with plt.rc_context(FIGURE_STYLE):
        figure, axes = plt.subplots(
            len(sessions),
            2,
            figsize=(6.8, 6.4),
            gridspec_kw={"height_ratios": heights},
            sharex="col",
        )
        for row, data in enumerate(sessions):
            mean = np.maximum(data["mean_rate"], 1e-9)[:, None]
            near_first = np.array(
                [
                    counts_around(s, data["first"], first_edges).mean(0)
                    / POPULATION_BIN_S
                    for s in data["spikes"]
                ]
            )
            exits = data["exit"][np.isfinite(data["exit"])]
            near_exit = np.array(
                [
                    counts_around(s, exits, exit_edges).mean(0) / POPULATION_BIN_S
                    for s in data["spikes"]
                ]
            )
            log_first = np.log2(np.maximum(near_first, 0.1) / mean)
            log_exit = np.log2(np.maximum(near_exit, 0.1) / mean)
            order = np.argsort(log_first[:, sort].mean(1))
            n = len(order)
            image = _heat(
                axes[row, 0],
                log_first[order],
                (*FIRST_FLASH_WINDOW_S, n, 0),
                LOG2_LIMIT,
                "RdBu_r",
            )
            _heat(
                axes[row, 1],
                log_exit[order],
                (*EXIT_WINDOW_S, n, 0),
                LOG2_LIMIT,
                "RdBu_r",
            )
            for axis in axes[row]:
                axis.axvline(0, color="0.2", linewidth=0.6)
            axes[row, 0].set_ylabel(f"{data['subject']}: {n} neurons")
            axes[row, 1].set_yticks([])
            report[data["subject"]] = {
                "units": n,
                "fraction_lower_150_500_ms": float(
                    np.mean(log_first[:, sort].mean(1) < 0)
                ),
                "values_past_the_color_limit": int(
                    np.sum(
                        np.abs(np.r_[log_first.ravel(), log_exit.ravel()]) > LOG2_LIMIT
                    )
                ),
            }
        axes[-1, 0].set_xlabel("time from the first flash (s)")
        axes[-1, 1].set_xlabel("time from the center exit (s)")
        bar = figure.colorbar(image, ax=axes, shrink=0.5, pad=0.02)
        bar.set_label("rate / mean rate (log2)")
        label_panels(axes[:, 0], x=-0.2, y=1.02)
        return save_figure(figure, OUT / "dataset_population"), report


def plot_flash_population(
    sessions: list[dict], key: str, events: str, name: str, xlabel: str
) -> tuple[Path, dict]:
    report = {}
    heights = [len(d["ids"]) for d in sessions]
    with plt.rc_context(FIGURE_STYLE):
        figure, axes = plt.subplots(
            len(sessions),
            1,
            figsize=(3.6, 6.4),
            gridspec_kw={"height_ratios": heights},
            sharex=True,
        )
        for axis, data in zip(axes, sessions, strict=True):
            response = data[key]
            order = np.argsort(response[:, data["peak"]].mean(1))
            n = len(order)
            edges = 1000 * data["flash_edges"]
            image = _heat(
                axis,
                response[order],
                (edges[0], edges[-1], n, 0),
                FLASH_LIMIT,
                "RdBu_r",
            )
            axis.axvline(0, color="0.2", linewidth=0.6)
            axis.set_ylabel(f"{data['subject']}: {n} neurons")
            report[data["subject"]] = {
                "units": n,
                "flashes": int(len(data[events])),
                "values_past_the_color_limit": int(
                    np.sum(np.abs(response) > FLASH_LIMIT)
                ),
            }
        axes[-1].set_xlabel(xlabel)
        bar = figure.colorbar(image, ax=axes, shrink=0.5, pad=0.04)
        bar.set_label("rate change / mean rate")
        label_panels(axes, x=-0.35, y=1.02)
        return save_figure(figure, OUT / name), report


def main() -> None:
    classes = json.loads((GLM / "raw_video_trust_classification.json").read_text())
    model = json.loads((GLM / "flash_weight_check.json").read_text())
    sessions = [
        session_data(
            s,
            set(classes[s])
            & set(model[s]["weights"]["units_by_model_class"]["excited"]),
        )
        for s in SESSIONS
    ]
    report = {
        d["subject"]: {
            "example_unit_id": int(d["ids"][d["example"]]),
            "example_mean_rate_hz": float(d["mean_rate"][d["example"]]),
            "still_flashes": int(len(d["still"])),
            "moving_flashes": int(len(d["moving"])),
        }
        for d in sessions
    }
    print(plot_flash_raster(sessions))
    path, population = plot_population(sessions)
    print(path)
    path, flashes = plot_flash_population(
        sessions,
        "flash_response",
        "still",
        "dataset_flash_population",
        "time from flash (ms)",
    )
    print(path)
    print(plot_first_flash(sessions))
    path, first = plot_flash_population(
        sessions,
        "first_response",
        "first_in_port",
        "dataset_first_flash_population",
        "time from the first flash (ms)",
    )
    print(path)
    report["population"] = population
    report["flash_population"] = flashes
    report["first_flash_population"] = first
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "dataset_overview.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
