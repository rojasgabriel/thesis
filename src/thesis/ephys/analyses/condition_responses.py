"""Do V1 neurons fire differently for the two values of a decoded variable?

Question: the stimulus category and the choice can be decoded from V1
(`category_decoding.py`). Before the decoder, what does each difference look
like in single neurons and in the population?

Comparison, in the same neurons (`--condition`):

- `choice`: left against right choices, aligned to the center exit, from
  `WINDOW_S[0]` to `WINDOW_S[1]` around it. After the exit, the choice is also
  the direction of the turn.
- `category`: slow against fast flash trains, aligned to the first flash. Only
  the time in the center port enters: each trial stops at its center exit
  (spikes after it are not drawn and its bins are left out of the means), so
  the movement after the exit and the choice it carries are not in the
  comparison. The first flash is the same in all trials. A bin is drawn only
  while at least `MIN_TRIALS_IN_BIN` of the trials of each group are still in
  the port; the heatmap shows the other bins in gray.

Common rules:

- Sessions: `SESSIONS`, the sessions of the decoders.
- Units: the stable single units (unit criteria 1, stability 0).
- Trials: the decoder trials (completed, a left or right response, not at the
  category boundary, with flashes); `load_decoding_data`. No balancing: the
  other variable is not equal across the two groups.
- Rates: spikes in `BIN_S` bins; no baseline is subtracted.
- Preference of a unit: (second − first group) mean rate over its mean rate
  in the window, in each bin (panels c, d), and a two-sided Mann–Whitney test
  of the trial rates in the preference window of the condition.
- Example neuron of each mouse: among the units with a mean rate of at least
  `MIN_RATE_HZ`, the one with the largest |d'| of those trial rates.
- Panels: a, b, the example neuron of each mouse: spikes of every trial, the
  first group above the second, and the mean rate (SEM over trials) of each
  group (second solid, first dashed); c, d, the preference of every unit,
  sorted by its value in the preference window. The heatmap limit is `LIMIT`;
  values past it take the end color, none is removed, and the report counts
  them. The sampling unit is the trial in a and b and the unit in c and d.

The script writes `figures/condition_responses/<condition>.svg` and
`results/condition_responses/<condition>.json`.

    THESIS_CACHE_DIR=figures/cache uv run python -m \\
        thesis.ephys.analyses.condition_responses --condition choice
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib
import numpy as np
from scipy.stats import mannwhitneyu

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thesis.ephys.analyses._decoding import load_decoding_data
from thesis.plotting import (
    CONCEPT_COLORS,
    FIGURE_STYLE,
    OBSERVED_COLOR,
    label_panels,
    save_figure,
)

OUT = Path("results/condition_responses")
SESSIONS = ("GRB006_20240821_121447", "GRB064_20261008_143359")
BIN_S = 0.02
MIN_RATE_HZ = 3.0
LIMIT = 2.0
MIN_TRIALS_IN_BIN = 0.5  # fraction of the trials of each group
# Each condition: its alignment, windows, the two groups (first, second) and
# their names, and one concept color; the group is the line style.
CONDITIONS = {
    "choice": {
        "align": "center exit",
        "window_s": (-0.5, 0.8),
        "preference_s": (0.0, 0.3),
        "in_port_only": False,
        "groups": (-1, 1),
        "names": ("left", "right"),
        "color": CONCEPT_COLORS["task"],
    },
    "category": {
        "align": "first flash",
        "window_s": (-0.3, 0.8),
        "preference_s": (0.1, 0.5),
        "in_port_only": True,
        "groups": (0, 1),
        "names": ("slow", "fast"),
        "color": CONCEPT_COLORS["visual"],
    },
}
STYLES = ("--", "-")


def session_responses(session: str, spec: dict) -> dict:
    subject, stamp = session.split("_", 1)
    units, trials, _ = load_decoding_data(
        subject,
        stamp,
        unit_criteria_id=1,
        stability_param_id=0,
        window_start_s=0.0,
        window_stop_s=1.0,
        include_mua=False,
    )
    exits = trials["center_exit_s"].to_numpy(float)
    first_flash = trials["stim_pulse_times_s"].str[0].to_numpy(float)
    align = exits if spec["align"] == "center exit" else first_flash
    if spec["groups"] == (-1, 1):
        group = trials["response"].to_numpy(int)
    else:
        group = trials["stim_category"].eq("high_rate").to_numpy(int)
    edges = np.arange(spec["window_s"][0], spec["window_s"][1] + 1e-9, BIN_S)
    spikes = [np.sort(np.asarray(s, dtype=float)) for s in units["spike_times_s"]]
    # (unit, trial, bin) rates around the alignment.
    rates = np.stack(
        [
            np.diff(np.searchsorted(s, align[:, None] + edges[None, :]), axis=1) / BIN_S
            for s in spikes
        ]
    )
    stop = np.full(len(align), np.inf)
    if spec["in_port_only"]:
        stop = exits - align  # the exit, in time from the alignment
        rates[:, edges[None, 1:] > stop[:, None]] = np.nan
    centers = (edges[:-1] + edges[1:]) / 2
    window = (centers >= spec["preference_s"][0]) & (centers < spec["preference_s"][1])
    with warnings_off():
        trial_rate = np.nanmean(rates[:, :, window], 2)  # (unit, trial)
        mean_rate = np.nanmean(rates, (1, 2))
        first, second = (group == g for g in spec["groups"])
        group_mean = [np.nanmean(rates[:, g], 1) for g in (first, second)]
    preference = (group_mean[1] - group_mean[0]) / np.maximum(mean_rate[:, None], 1e-9)
    # Draw a bin only while most trials of both groups are still in it.
    shown = (
        np.minimum(*(np.isfinite(rates[0][g]).mean(0) for g in (first, second)))
        >= MIN_TRIALS_IN_BIN
    )
    preference[:, ~shown] = np.nan
    kept = np.isfinite(trial_rate[0])  # trials still in the port in the window
    a, b = trial_rate[:, first & kept], trial_rate[:, second & kept]
    spread = np.sqrt((a.var(1, ddof=1) + b.var(1, ddof=1)) / 2)
    dprime = (b.mean(1) - a.mean(1)) / np.maximum(spread, 1e-9)
    p = np.array(
        [
            mannwhitneyu(y, x).pvalue if np.any(np.r_[x, y]) else 1.0
            for x, y in zip(a, b, strict=True)
        ]
    )
    active = np.flatnonzero(mean_rate >= MIN_RATE_HZ)
    example = int(active[np.argmax(np.abs(dprime[active]))])
    return {
        "subject": subject,
        "unit_ids": units["unit_id"].astype(int).tolist(),
        "spikes": spikes,
        "align": align,
        "stop": stop,
        "group": group,
        "trials_kept": [int((g & kept).sum()) for g in (first, second)],
        "centers": centers,
        "shown": shown,
        "rates": rates,
        "preference": preference,
        "dprime": dprime,
        "p": p,
        "example": example,
    }


def warnings_off():
    """Silence the empty-slice warnings of bins after every exit."""
    return warnings.catch_warnings(action="ignore", category=RuntimeWarning)


def draw_example(raster_axis, rate_axis, data: dict, spec: dict) -> None:
    unit = data["example"]
    spikes, align, stop = data["spikes"][unit], data["align"], data["stop"]
    color, window = spec["color"], spec["window_s"]
    row = 0
    # The second group at the bottom, the first above; in-port trials sorted
    # by the time of the exit.
    for index in (1, 0):
        start = row
        trials = np.flatnonzero(data["group"] == spec["groups"][index])
        trials = trials[np.argsort(stop[trials], kind="stable")]
        for trial in trials:
            end = min(window[1], stop[trial])
            s = spikes[
                (spikes >= align[trial] + window[0]) & (spikes < align[trial] + end)
            ]
            raster_axis.vlines(
                s - align[trial], row, row + 0.9, color=OBSERVED_COLOR, lw=0.8
            )
            row += 1
        if spec["in_port_only"]:
            exits = stop[trials]
            shown = exits < window[1]
            raster_axis.scatter(
                exits[shown],
                start + np.flatnonzero(shown) + 0.45,
                s=2,
                color=CONCEPT_COLORS["task"],
                lw=0,
            )
        raster_axis.text(
            1.01,
            (start + row) / 2,
            spec["names"][index],
            color=color,
            transform=raster_axis.get_yaxis_transform(),
            va="center",
        )
        if index == 1:
            raster_axis.axhline(row, color="0.6", lw=0.8)
    raster_axis.axvline(0, color=color, lw=0.8)
    raster_axis.set_ylim(0, row)
    raster_axis.set_ylabel("trials")
    raster_axis.text(
        0.0,
        1.02,
        f"{data['subject']}: unit {data['unit_ids'][unit]}",
        transform=raster_axis.transAxes,
        va="bottom",
    )
    if spec["in_port_only"]:
        raster_axis.text(
            1.0,
            1.02,
            "center exit",
            color=CONCEPT_COLORS["task"],
            transform=raster_axis.transAxes,
            ha="right",
            va="bottom",
        )
    centers = data["centers"]
    for index, value in enumerate(spec["groups"]):
        r = data["rates"][unit][data["group"] == value]
        n = np.sum(np.isfinite(r), 0)
        with warnings_off():
            mean = np.where(data["shown"], np.nanmean(r, 0), np.nan)
            sem = np.nanstd(r, 0, ddof=1) / np.sqrt(n)
        rate_axis.plot(centers, mean, color=color, lw=1.4, ls=STYLES[index])
        rate_axis.fill_between(
            centers, mean - sem, mean + sem, color=color, alpha=0.15, lw=0
        )
        y = 0.92 - 0.13 * index
        rate_axis.plot(
            [0.02, 0.09],
            [y, y],
            color=color,
            ls=STYLES[index],
            lw=1.4,
            transform=rate_axis.transAxes,
        )
        rate_axis.text(
            0.11,
            y,
            spec["names"][index],
            color=color,
            transform=rate_axis.transAxes,
            va="center",
        )
    rate_axis.axvline(0, color=color, lw=0.8)
    rate_axis.set_xlabel(f"time from the {spec['align']} (s)")
    rate_axis.set_ylabel("rate (spikes/s)")


def draw_population(axis, data: dict, spec: dict):
    centers = data["centers"]
    bins = (centers >= spec["preference_s"][0]) & (centers < spec["preference_s"][1])
    order = np.argsort(np.nanmean(data["preference"][:, bins], 1))
    image = axis.imshow(
        data["preference"][order],
        aspect="auto",
        cmap=matplotlib.colormaps["RdBu_r"].with_extremes(bad="0.8"),
        vmin=-LIMIT,
        vmax=LIMIT,
        extent=(centers[0] - BIN_S / 2, centers[-1] + BIN_S / 2, len(order), 0),
        interpolation="nearest",
    )
    axis.axvline(0, color=OBSERVED_COLOR, lw=0.8)
    axis.set_xlabel(f"time from the {spec['align']} (s)")
    axis.set_ylabel(f"{data['subject']}: {len(order)} neurons")
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--condition", choices=tuple(CONDITIONS), default="choice")
    name = parser.parse_args().condition
    spec = CONDITIONS[name]
    sessions = [session_responses(s, spec) for s in SESSIONS]
    first, second = spec["names"]
    with plt.rc_context(FIGURE_STYLE):
        figure = plt.figure(figsize=(9.0, 6.2), layout="constrained")
        grid = figure.add_gridspec(2, 2, width_ratios=(1.0, 0.8))
        panel_axes, image = [], None
        for row, data in enumerate(sessions):
            cell = grid[row, 0].subgridspec(2, 1, height_ratios=(1.3, 1.0), hspace=0.05)
            raster = figure.add_subplot(cell[0])
            rate = figure.add_subplot(cell[1], sharex=raster)
            raster.tick_params(labelbottom=False)
            draw_example(raster, rate, data, spec)
            heat = figure.add_subplot(grid[row, 1])
            image = draw_population(heat, data, spec)
            panel_axes += [raster, heat]
        figure.colorbar(
            image,
            ax=[a for a in panel_axes[1::2]],
            label=f"{second} − {first} (/ mean rate)",
        )
        label_panels([panel_axes[0], panel_axes[2], panel_axes[1], panel_axes[3]])
        path = save_figure(figure, OUT / name)
    report = {
        d["subject"]: {
            "units": len(d["unit_ids"]),
            f"trials_{first}_{second}": [
                int((d["group"] == g).sum()) for g in spec["groups"]
            ],
            "trials_in_the_preference_window": d["trials_kept"],
            "example_unit_id": d["unit_ids"][d["example"]],
            "example_dprime": float(d["dprime"][d["example"]]),
            "units_with_preference_p_below_0.05": int((d["p"] < 0.05).sum()),
            f"of_which_prefer_{second}": int(
                ((d["p"] < 0.05) & (d["dprime"] > 0)).sum()
            ),
            "preference_window_s": list(spec["preference_s"]),
            "in_port_only": spec["in_port_only"],
            "values_past_the_color_limit": int(np.sum(np.abs(d["preference"]) > LIMIT)),
            "values": int(np.isfinite(d["preference"]).sum()),
        }
        for d in sessions
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    print(path)


if __name__ == "__main__":
    main()
