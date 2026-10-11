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

- Panel a (observation): rate against motion energy for each state; median and
  interquartile range over units.
- Panel b (the paired test): each unit's rate in the center port against its
  rate at the side port, in the group with the least motion energy. Bars are
  the Poisson standard error of each rate. Two-sided Wilcoxon signed-rank test.
- Panel c (prediction of the stillness account): if stillness caused the
  suppression, the units that movement excites most would be the most
  suppressed. x is the Spearman correlation of the spike count with motion
  energy outside the center port; y is the rate in the center port relative to
  the rate outside it, over all frames (not matched for motion energy).

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
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr, wilcoxon

from thesis.ephys.analyses.glm import cli, dmat
from thesis.ephys.analyses.glm.fit import analysis_unit_indices
from thesis.ephys.units import fetch_unit_table
from thesis.plotting import (
    CONCEPT_COLORS,
    FIGURE_STYLE,
    REFERENCE_COLOR,
    label_panels,
    save_figure,
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


def plot(data: dict, output: Path | None, axes=None) -> tuple[Path | None, dict]:
    """Draw the three panels and return the figure path and the statistics.

    With `axes`, draw into them and do not label or save (for a combined figure).
    """
    rate, error = data["rate"], data["error"]
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
    }

    with plt.rc_context(FIGURE_STYLE):
        figure = None
        if axes is None:
            figure, axes = plt.subplots(
                1, 3, figsize=(7.4, 2.5), width_ratios=(1.5, 1, 1), layout="constrained"
            )
        for index, (name, color) in enumerate(zip(STATES, STATE_COLORS, strict=True)):
            used = data["frames"][index] >= MIN_FRAMES
            low, median, high = np.nanpercentile(
                rate[:, index, used], [25, 50, 75], axis=0
            )
            positions = np.flatnonzero(used)
            axes[0].errorbar(
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
            axes[0].text(  # outside the axis, so the axis ends at the data
                GROUPS - 0.7,
                median[-1] + (0.09, -0.09, 0)[(index + 2) % 3],
                name,
                color=color,
                va="center",
                clip_on=False,
            )
        centers = (np.arange(GROUPS) + 0.5) * 100 / GROUPS
        axes[0].set_xticks(range(GROUPS), [f"{center:.0f}" for center in centers])
        axes[0].set_xlim(-0.3, GROUPS - 0.8)
        axes[0].set_ylim(bottom=0)
        axes[0].set_xlabel("motion energy of the frame (percentile)")
        axes[0].set_ylabel("rate / mean rate of the unit")

        limit = 1.05 * np.nanmax(np.r_[center[paired], side[paired]])
        axes[1].plot([0, limit], [0, limit], color=REFERENCE_COLOR, linewidth=0.8)
        axes[1].errorbar(
            side[paired],
            center[paired],
            xerr=error[paired, 2, 0],
            yerr=error[paired, 0, 0],
            fmt="o",
            color="black",
            markersize=2,
            elinewidth=0.4,
            alpha=0.6,
        )
        axes[1].set_xlim(0, limit)
        axes[1].set_ylim(0, limit)
        axes[1].set_xlabel("rate at side port")
        axes[1].set_ylabel("rate in center port")
        axes[1].text(
            0.5,
            1.02,
            f"lower in center port:\n{np.mean(ratio < 1):.0%} of {paired.sum()} units",
            transform=axes[1].transAxes,
            ha="center",
            va="bottom",
        )

        axes[2].axhline(1, color=REFERENCE_COLOR, linestyle="--", linewidth=0.8)
        axes[2].scatter(
            sensitivity, hold_ratio, s=6, color="black", alpha=0.6, linewidths=0
        )
        axes[2].set_yscale("log")
        axes[2].set_xlabel("movement sensitivity\noutside port (ρ)")
        axes[2].set_ylabel("rate in center port / outside it")
        axes[2].text(
            0.5,
            1.02,
            f"ρ = {relation.statistic:.2f}, p = {relation.pvalue:.2g}",
            transform=axes[2].transAxes,
            ha="center",
            va="bottom",
        )
        if figure is None:
            return None, summary
        label_panels(axes, -0.3, 1.14)
        return save_figure(figure, output), summary


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
    figure, summary = plot(data, args.root / "port_vs_stillness")
    (args.root / "port_vs_stillness.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({**summary, "figure": str(figure)}, indent=2))


if __name__ == "__main__":
    main()
