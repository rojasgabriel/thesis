"""Test whether the hold-period suppression in V1 is explained by stillness.

Question: units fire less while the animal holds in the center port. Is that
because the animal is still, or is it specific to the center port? The side
ports have the same geometry, so the side port is also the control for "the
head is in a port". The stimulus plays while the animal is in the center port
and is off when it enters a side port.

Comparison: raw firing rates, no model fit. Each video frame in the trial
windows of the V1 GLM gets a state and a motion-energy group. Rates are
compared between states inside the same motion-energy group, so the animal
moves the same amount in both. The comparison is paired: each unit has a rate
in each state.

- States: "in center port" is from 150 ms after the first flash (the
  suppression has a latency) to the center exit. "moving to side port" is from
  100 ms after the center exit to the response-port entry. "at side port" is
  from 100 ms after the response-port entry to the end of the trial window.
- Motion energy: the total motion energy of the frame (`total` in the video
  features, the mean absolute frame difference). Groups are the quintiles over
  all frames that have a state.
- Rate: spikes in the frame interval, as a fraction of the unit's mean over all
  frames that have a state (the baseline). A state and group with fewer than
  `MIN_FRAMES` frames is not used.

Units, trials, alignment, and exclusions are those of `fit.py` (all eligible
units; completed trials; windows aligned to the first flash). The sampling unit
is the unit.

The script writes the rates to `port_vs_stillness.npz` and these statistics to
`port_vs_stillness.json`; it makes no figure. `design_optimization.py` draws
the rate against motion energy with `plot_rates`.

- Paired test: each unit's rate in the center port against its rate at the
  side port, in the group with the least motion energy (two-sided Wilcoxon
  signed-rank test).
- Slope test: the slope of each unit's rate over the motion-energy groups, in
  the center port and at the side port (two-sided Wilcoxon signed-rank tests
  of each slope against zero and of the difference of the two). If the low
  rate in the port were the low rate of a still animal, the rate in the port
  would go up with motion energy as it does at the side port.
- Prediction of the stillness account: if stillness caused the suppression,
  the units that movement excites most would be the most suppressed. Spearman
  correlation, over units, of the spike count's correlation with motion energy
  outside the center port against the rate in the center port relative to the
  rate outside it (all frames, not matched for motion energy).

Limits: motion energy comes from one camera view of the body and has no eye,
pupil, or whisker signal. The comparison does not separate the stimulus in
view from the task state (hold at the center port, reward at the side port).

    uv run python -m thesis.ephys.analyses.glm.port_vs_stillness
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
from scipy.stats import spearmanr, wilcoxon

from thesis.ephys.analyses.glm import cli, dmat
from thesis.ephys.analyses.glm.fit import analysis_unit_indices
from thesis.ephys.units import fetch_unit_table
from thesis.plotting import (
    CONCEPT_COLORS,
)

STATES = ("in center port", "moving to side port", "at side port")
STATE_COLORS = (CONCEPT_COLORS["task"], CONCEPT_COLORS["motion energy"], "0.35")
HOLD_LATENCY_S = 0.15
AFTER_EVENT_S = 0.1
GROUPS = 5
MIN_FRAMES = 50


def frame_table(root: Path) -> dict[str, np.ndarray]:
    """Return the time, state, and motion-energy group of each classified frame."""
    prepared = dmat._load_windows(root / "stimulus_windows_me.npz")
    trials = dmat._load_selected_trials(prepared)
    with np.load(root / "video_me_features.npz", allow_pickle=False) as video:
        times, rows, energy = (
            video["frame_times_s"],
            video["frame_trial_row"],
            video["total"],
        )
    column = lambda name: trials[name].to_numpy(dtype=float)[rows]  # noqa: E731
    first = trials["stim_pulse_times_s"].str[0].to_numpy(dtype=float)[rows]
    center_exit, entry = column("center_exit_s"), column("response_port_entry_s")
    state = np.full(len(times), -1)
    state[(times >= first + HOLD_LATENCY_S) & (times < center_exit)] = 0
    state[(times >= center_exit + AFTER_EVENT_S) & (times < entry)] = 1
    state[times >= entry + AFTER_EVENT_S] = 2
    keep = (rows >= 0) & (state >= 0)
    edges = np.quantile(energy[keep], np.linspace(0, 1, GROUPS + 1))
    group = np.clip(np.searchsorted(edges, energy, side="right") - 1, 0, GROUPS - 1)
    return {
        "times": times[keep],
        "state": state[keep],
        "group": group[keep],
        "energy": energy[keep],
        "interval": float(np.median(np.diff(times))),
        "metadata": prepared["metadata"],
    }


def unit_rates(frames: dict, units) -> dict[str, np.ndarray]:
    """Return each unit's rate and its error by state and group, and its movement sensitivity."""
    state, group = frames["state"], frames["group"]
    half = frames["interval"] / 2
    cells = [
        [(state == s) & (group == g) for g in range(GROUPS)] for s in range(len(STATES))
    ]
    sizes = np.asarray([[cell.sum() for cell in row] for row in cells])
    outside = state > 0
    rate, error, sensitivity, hold_ratio, unit_ids = [], [], [], [], []
    for _, unit in units.iterrows():
        spikes = np.asarray(unit["spike_times_s"], dtype=float)
        counts = np.searchsorted(spikes, frames["times"] + half) - np.searchsorted(
            spikes, frames["times"] - half
        )
        baseline = counts.mean()
        if baseline <= 0:
            continue
        sums = np.asarray([[counts[cell].sum() for cell in row] for row in cells])
        with np.errstate(invalid="ignore", divide="ignore"):
            rate.append(np.where(sizes >= MIN_FRAMES, sums / sizes / baseline, np.nan))
            error.append(
                np.where(sizes >= MIN_FRAMES, np.sqrt(sums) / sizes / baseline, np.nan)
            )
        sensitivity.append(
            spearmanr(frames["energy"][outside], counts[outside]).statistic
        )
        hold_ratio.append(counts[~outside].mean() / counts[outside].mean())
        unit_ids.append(int(unit["unit_id"]))
    return {
        "rate": np.asarray(rate),
        "error": np.asarray(error),
        "sensitivity": np.asarray(sensitivity),
        "hold_ratio": np.asarray(hold_ratio),
        "unit_ids": np.asarray(unit_ids),
        "frames": sizes,
        "group_energy": np.asarray(
            [np.median(frames["energy"][group == g]) for g in range(GROUPS)]
        ),
    }


def summarize(data: dict) -> dict:
    """Return the statistics of the paired test and of the stillness prediction."""
    rate = data["rate"]
    energy = data["group_energy"]
    center, side = rate[:, 0, 0], rate[:, 2, 0]
    paired = np.isfinite(center) & np.isfinite(side) & (center > 0) & (side > 0)
    ratio = center[paired] / side[paired]
    hold = np.isfinite(data["hold_ratio"]) & (data["hold_ratio"] > 0)
    sensitivity, hold_ratio = data["sensitivity"][hold], data["hold_ratio"][hold]
    relation = spearmanr(sensitivity, hold_ratio)
    summary = {
        "units": int(paired.sum()),
        "frames_by_state_and_group": data["frames"].tolist(),
        "group_median_motion_energy": energy.tolist(),
        "median_rate_by_state_and_group": np.nanmedian(rate, axis=0).tolist(),
        "least_motion_group": {
            "median_center_over_side": float(np.median(ratio)),
            "fraction_units_lower_in_center_port": float(np.mean(ratio < 1)),
            "wilcoxon_p": float(wilcoxon(center[paired], side[paired]).pvalue),
        },
        "sensitivity_against_suppression": {
            "spearman_rho": float(relation.statistic),
            "p": float(relation.pvalue),
        },
        "slope_against_motion_energy_group": _slopes(rate, data["frames"]),
    }
    return summary


def _slopes(rate: np.ndarray, frames: np.ndarray) -> dict:
    """Return the slope of each unit's rate over the motion-energy groups, by port.

    The slope is the least-squares line over the groups with `MIN_FRAMES` or
    more frames (rate per group). Two-sided Wilcoxon signed-rank tests: each
    slope against zero, and the side-port slope against the center-port slope.
    """
    slopes = {}
    for state, name in ((0, "center port"), (2, "side port")):
        groups = np.flatnonzero(frames[state] >= MIN_FRAMES)
        slopes[name] = np.asarray(
            [
                np.polyfit(groups[np.isfinite(row)], row[np.isfinite(row)], 1)[0]
                for row in rate[:, state, groups]
            ]
        )
    difference = slopes["side port"] - slopes["center port"]
    return {
        **{
            name: {
                "median": float(np.median(values)),
                "wilcoxon_p": float(wilcoxon(values).pvalue),
            }
            for name, values in slopes.items()
        },
        "side minus center": {
            "median": float(np.median(difference)),
            "units_higher_at_side_port": int(np.sum(difference > 0)),
            "units": len(difference),
            "wilcoxon_p": float(wilcoxon(difference).pvalue),
        },
    }


def plot_rates(axis, data: dict, states=(0, 1, 2)) -> None:
    """Draw rate against motion-energy group for the states in `states` (indices of `STATES`)."""
    rate = data["rate"]
    for index, (name, color) in enumerate(zip(STATES, STATE_COLORS, strict=True)):
        if index not in states:
            continue
        used = data["frames"][index] >= MIN_FRAMES
        low, median, high = np.nanpercentile(rate[:, index, used], [25, 50, 75], axis=0)
        positions = np.flatnonzero(used)
        axis.errorbar(
            positions + 0.06 * (index - 1),
            median,
            yerr=(median - low, high - median),
            color=color,
            marker="o",
            markersize=3.5,
            linewidth=1.2,
            elinewidth=0.7,
            capsize=1.5,
        )
        axis.text(  # outside the axis, so the axis ends at the data
            GROUPS - 0.7,
            median[-1] + (0.09, -0.09, 0)[(index + 2) % 3],
            name,
            color=color,
            va="center",
            clip_on=False,
        )
    centers = (np.arange(GROUPS) + 0.5) * 100 / GROUPS
    axis.set_xticks(range(GROUPS), [f"{center:.0f}" for center in centers])
    axis.set_xlim(-0.3, GROUPS - 0.8)
    axis.set_ylim(bottom=0)
    axis.set_xlabel("motion energy of the frame (percentile)")
    axis.set_ylabel("normalized firing rate")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root",
        type=Path,
        default=cli.PoissonGLM().root,
        help="Session artifact directory with the prepared windows and video.",
    )
    args = parser.parse_args()
    frames = frame_table(args.root)
    units = fetch_unit_table(
        frames["metadata"]["subject_name"],
        frames["metadata"]["session_name"],
        unit_criteria_id=1,
        stability_param_id=0,
        include_metrics=False,
    )
    data = unit_rates(frames, units.iloc[analysis_unit_indices(units, "all")])
    np.savez(args.root / "port_vs_stillness.npz", **data)
    summary = summarize(data)
    (args.root / "port_vs_stillness.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
