"""Do V1 neurons fire differently for left and right choices?

Question: the choice can be decoded from V1 (`category_decoding.py`), most
of all after the mouse leaves the center port. Before the decoder, what does
that look like in single neurons and in the population?

Comparison: trials with a left choice against trials with a right choice, in
the same neurons, aligned to the center exit.

- Sessions: `SESSIONS`, the sessions of the decoders.
- Units: the stable single units (unit criteria 1, stability 0).
- Trials: the decoder trials (completed, a left or right response, not at the
  category boundary, with flashes); `load_decoding_data`.
- Rates: spikes in `BIN_S` bins from `WINDOW_S[0]` to `WINDOW_S[1]` around the
  center exit; no baseline is subtracted.
- Choice preference of a unit: (right − left) mean rate over its mean rate in
  the window, in each bin (panel c, d), and a two-sided Mann–Whitney test of
  the trial rates from `PREFERENCE_S[0]` to `PREFERENCE_S[1]` after the exit.
- Example neuron of each mouse: among the units with a mean rate of at least
  `MIN_RATE_HZ`, the one with the largest |d'| of those trial rates.
- Panels: a, b, the example neuron of each mouse: spikes of every trial, left
  choices above right choices, and the mean rate (SEM over trials) for each
  choice (right solid, left dashed); c, d, the choice preference of every unit, sorted by its value from
  `PREFERENCE_S`. The heatmap limit is `LIMIT`; values past it take the end
  color, none is removed, and the report counts them. The sampling unit is the trial in a and b and the
  unit in c and d.

The script writes `figures/choice_responses/choice_responses.svg` and
`results/choice_responses/choice_responses.json`.

    THESIS_CACHE_DIR=figures/cache uv run python -m \\
        thesis.ephys.analyses.choice_responses
"""

from __future__ import annotations

import json
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

OUT = Path("results/choice_responses")
SESSIONS = ("GRB006_20240821_121447", "GRB064_20261008_143359")
WINDOW_S = (-0.5, 0.8)
BIN_S = 0.02
PREFERENCE_S = (0.0, 0.3)
MIN_RATE_HZ = 3.0
LIMIT = 1.0
# The choice is a task variable: the task color; the side is the line style.
CHOICE_COLOR = CONCEPT_COLORS["task"]
CHOICE_STYLES = {-1: "--", 1: "-"}
CHOICE_NAMES = {-1: "left choice", 1: "right choice"}


def session_responses(session: str) -> dict:
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
    choice = trials["response"].to_numpy(int)
    edges = np.arange(WINDOW_S[0], WINDOW_S[1] + 1e-9, BIN_S)
    spikes = [np.sort(np.asarray(s, dtype=float)) for s in units["spike_times_s"]]
    # (unit, trial, bin) rates around the exit.
    rates = np.stack(
        [
            np.diff(np.searchsorted(s, exits[:, None] + edges[None, :]), axis=1) / BIN_S
            for s in spikes
        ]
    )
    centers = (edges[:-1] + edges[1:]) / 2
    window = (centers >= PREFERENCE_S[0]) & (centers < PREFERENCE_S[1])
    trial_rate = rates[:, :, window].mean(2)  # (unit, trial)
    mean_rate = rates.mean((1, 2))
    left, right = choice == -1, choice == 1
    preference = (rates[:, right].mean(1) - rates[:, left].mean(1)) / np.maximum(
        mean_rate[:, None], 1e-9
    )
    spread = np.sqrt(
        (trial_rate[:, left].var(1, ddof=1) + trial_rate[:, right].var(1, ddof=1)) / 2
    )
    dprime = (trial_rate[:, right].mean(1) - trial_rate[:, left].mean(1)) / np.maximum(
        spread, 1e-9
    )
    p = np.array(
        [
            mannwhitneyu(r[right], r[left]).pvalue if np.any(r) else 1.0
            for r in trial_rate
        ]
    )
    active = np.flatnonzero(mean_rate >= MIN_RATE_HZ)
    example = int(active[np.argmax(np.abs(dprime[active]))])
    return {
        "subject": subject,
        "unit_ids": units["unit_id"].astype(int).tolist(),
        "spikes": spikes,
        "exits": exits,
        "choice": choice,
        "centers": centers,
        "rates": rates,
        "preference": preference,
        "dprime": dprime,
        "p": p,
        "example": example,
    }


def draw_example(raster_axis, rate_axis, data: dict) -> None:
    unit = data["example"]
    spikes, exits, choice = data["spikes"][unit], data["exits"], data["choice"]
    row = 0
    for side in (1, -1):  # right choices at the bottom, left choices above
        start = row
        for trial in np.flatnonzero(choice == side):
            s = spikes[
                (spikes >= exits[trial] + WINDOW_S[0])
                & (spikes < exits[trial] + WINDOW_S[1])
            ]
            raster_axis.vlines(
                s - exits[trial], row, row + 0.9, color=OBSERVED_COLOR, lw=0.8
            )
            row += 1
        raster_axis.text(
            1.01,
            (start + row) / 2,
            CHOICE_NAMES[side].split()[0],
            color=CHOICE_COLOR,
            transform=raster_axis.get_yaxis_transform(),
            va="center",
        )
        if side == 1:
            raster_axis.axhline(row, color="0.6", lw=0.8)
    raster_axis.axvline(0, color=CHOICE_COLOR, lw=0.8)
    raster_axis.set_ylim(0, row)
    raster_axis.set_ylabel("trials")
    raster_axis.text(
        0.0,
        1.02,
        f"{data['subject']}: unit {data['unit_ids'][unit]}",
        transform=raster_axis.transAxes,
        va="bottom",
    )
    centers = data["centers"]
    for side in (-1, 1):
        r = data["rates"][unit][choice == side]
        mean, sem = r.mean(0), r.std(0, ddof=1) / np.sqrt(len(r))
        rate_axis.plot(
            centers, mean, color=CHOICE_COLOR, lw=1.4, ls=CHOICE_STYLES[side]
        )
        rate_axis.fill_between(
            centers, mean - sem, mean + sem, color=CHOICE_COLOR, alpha=0.15, lw=0
        )
        y = 0.92 - 0.13 * (side == 1)
        rate_axis.plot(
            [0.02, 0.09],
            [y, y],
            color=CHOICE_COLOR,
            ls=CHOICE_STYLES[side],
            lw=1.4,
            transform=rate_axis.transAxes,
        )
        rate_axis.text(
            0.11,
            y,
            CHOICE_NAMES[side],
            color=CHOICE_COLOR,
            transform=rate_axis.transAxes,
            va="center",
        )
    rate_axis.axvline(0, color=CHOICE_COLOR, lw=0.8)
    rate_axis.set_xlabel("time from the center exit (s)")
    rate_axis.set_ylabel("rate (spikes/s)")


def draw_population(axis, data: dict):
    order = np.argsort(data["preference"][:, _preference_bins(data)].mean(1))
    centers = data["centers"]
    image = axis.imshow(
        data["preference"][order],
        aspect="auto",
        cmap="RdBu_r",
        vmin=-LIMIT,
        vmax=LIMIT,
        extent=(centers[0] - BIN_S / 2, centers[-1] + BIN_S / 2, len(order), 0),
        interpolation="nearest",
    )
    axis.axvline(0, color=OBSERVED_COLOR, lw=0.8)
    axis.set_xlabel("time from the center exit (s)")
    axis.set_ylabel(f"{data['subject']}: {len(order)} neurons")
    return image


def _preference_bins(data: dict) -> np.ndarray:
    c = data["centers"]
    return (c >= PREFERENCE_S[0]) & (c < PREFERENCE_S[1])


def main() -> None:
    sessions = [session_responses(s) for s in SESSIONS]
    with plt.rc_context(FIGURE_STYLE):
        figure = plt.figure(figsize=(9.0, 6.2), layout="constrained")
        grid = figure.add_gridspec(2, 2, width_ratios=(1.0, 0.8))
        panel_axes, image = [], None
        for row, data in enumerate(sessions):
            cell = grid[row, 0].subgridspec(2, 1, height_ratios=(1.3, 1.0), hspace=0.05)
            raster = figure.add_subplot(cell[0])
            rate = figure.add_subplot(cell[1], sharex=raster)
            raster.tick_params(labelbottom=False)
            draw_example(raster, rate, data)
            heat = figure.add_subplot(grid[row, 1])
            image = draw_population(heat, data)
            panel_axes += [raster, heat]
        figure.colorbar(
            image, ax=[a for a in panel_axes[1::2]], label="right − left (/ mean rate)"
        )
        label_panels([panel_axes[0], panel_axes[2], panel_axes[1], panel_axes[3]])
        path = save_figure(figure, OUT / "choice_responses")
    report = {
        d["subject"]: {
            "units": len(d["unit_ids"]),
            "trials_left_right": [int((d["choice"] == s).sum()) for s in (-1, 1)],
            "example_unit_id": d["unit_ids"][d["example"]],
            "example_dprime": float(d["dprime"][d["example"]]),
            "units_with_choice_preference_p_below_0.05": int((d["p"] < 0.05).sum()),
            "of_which_prefer_right": int(((d["p"] < 0.05) & (d["dprime"] > 0)).sum()),
            "preference_window_s": list(PREFERENCE_S),
            "values_past_the_color_limit": int(np.sum(np.abs(d["preference"]) > LIMIT)),
            "values": int(d["preference"].size),
        }
        for d in sessions
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "choice_responses.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    print(path)


if __name__ == "__main__":
    main()
