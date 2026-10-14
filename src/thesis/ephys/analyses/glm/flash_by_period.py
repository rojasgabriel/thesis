"""Compare the flash response when the animal is still and after it leaves the center port.

Question: the peak response of V1 units to a flash is higher after the animal
leaves the center port than in the fixation (`stim_responses_by_locomotion.py`).
Is that a change of the response to the flash (the flash × move terms of the
model), or a change of everything else at the exit?

Comparison: "still" is a flash in the fixation and "move" is a flash after the
center exit. These are task periods; the script does not measure locomotion,
and the exit also changes the view of the stimulus. One measure, computed in
the same way on three rates of the same flashes and units:

- observed: the spike counts;
- model: the rate of the final refit of the unit (all its terms, with the
  observed spike history; in-sample);
- model with no flash × move: the same rate with the weights of the flash × move
  and flash × move × gap terms set to 0.

The difference of the last two is the part of the change that the flash × move
terms carry.

- Flashes: in each trial, the last flash whose window (`BASELINE_MS[0]` to
  `PEAK_MS[1]`) is in the fixation, and the first flash at or after the center
  exit whose window ends before the response entry. Only trials with both are
  used, so the comparison is paired over the same trials.
- Measure (as in `stim_responses_by_locomotion.py`): the trial-mean rate in
  `peaks.PSTH_BIN_MS` bins; the baseline is the still rate from `BASELINE_MS`, and
  the same baseline is taken from both periods; the peak is the largest
  baseline-corrected rate with a bin center in `PEAK_MS`, found in each period
  apart (spikes/s).
- Units: the units with the "excited" classification of StimulusResponse
  (stimulus-response parameter 0) that the GLM fits, as in the raw figure. A
  peak has no meaning for a unit with no response to the flash.
- Units of both mice are pooled in the plots (standard practice; Parker et al.
  2022), and the caption gives the units and mice. The large dot is the pooled
  median with a hierarchical-bootstrap 95% interval (resample the mice, then
  the units of each; Saravanan et al. 2020); the small open circles are the
  median of each mouse. Statistics: the hierarchical bootstrap of the pooled
  median, and in each mouse a two-sided Wilcoxon signed-rank test, of move
  minus still in each rate, of the observed against the model change, and of
  the model against the model with no flash × move.
- Figure: a, b, observed and model peaks, still against move; c, move minus
  still in the three rates. A point outside the axis limits is drawn at the
  limit; it is in the statistics, and the count is in the output.

Trials, alignment (first flash), and exclusions are those of `fit.py`. The
sampling unit is the unit. The script writes `flash_by_period.svg` and
`flash_by_period.json` in `figures/glm/`.

    uv run python -m thesis.ephys.analyses.glm.flash_by_period
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

from thesis.ephys.analyses.glm import cli, dmat, peaks
from thesis.ephys.analyses.glm.flash_kernels import SESSIONS
from thesis.ephys.analyses.glm.peaks import BASELINE_MS, PEAK_MS
from thesis.plotting import (
    FIGURE_STYLE,
    MODEL_COLOR,
    OBSERVED_COLOR,
    PEAK_AXIS_LIMITS,
    REFERENCE_COLOR,
    hierarchical_interval,
    label_panels,
    median_interval,
    peak_axes,
    pooled_points,
    pooled_strip,
    save_figure,
)

INTERACTION = (dmat.FLASH_AFTER_EXIT, dmat.FLASH_MOVE_GAP)
RATES = ("observed", "model", "model, no flash × move")
COLORS = (OBSERVED_COLOR, MODEL_COLOR, "0.55")
PEAK_LIMITS = PEAK_AXIS_LIMITS
CHANGE_LIMITS = (-25, 95)


def paired_flashes(trials) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the trial row and the times of the still and the move flash."""
    before_s, after_s = -BASELINE_MS[0] / 1000, PEAK_MS[1] / 1000
    rows, still, move = [], [], []
    for row, (flashes, entry, leave, response) in enumerate(
        zip(
            trials["stim_pulse_times_s"],
            trials["center_entry_s"],
            trials["center_exit_s"],
            trials["response_port_entry_s"],
            strict=True,
        )
    ):
        flashes = np.asarray(flashes, dtype=float)
        inside = flashes[(flashes - before_s >= entry) & (flashes + after_s <= leave)]
        outside = flashes[(flashes >= leave) & (flashes + after_s <= response)]
        if len(inside) and len(outside):
            rows.append(row)
            still.append(inside[-1])
            move.append(outside[0])
    return np.asarray(rows), np.asarray(still), np.asarray(move)


def session_peaks(root: Path, unit_set: str) -> dict:
    """Return the still and move peaks of each unit: (unit, rate, period)."""
    model = cli.PoissonGLM(root=root)
    trials = dmat._load_selected_trials(dmat._load_windows(model.windows))
    rows, still, move = paired_flashes(trials)
    data = peaks.session_peaks(
        root, unit_set, [(rows, still), (rows, move)], zeroed=INTERACTION
    )
    return {**data, "trials": len(rows)}


def _change(values: np.ndarray) -> dict:
    median, low, high = median_interval(values)
    return {
        "units": len(values),
        "median": median,
        "median_95_interval": [low, high],
        "units_above_0": int((values > 0).sum()),
        "wilcoxon_p": float(wilcoxon(values).pvalue),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("figures/glm"))
    parser.add_argument("--units", type=cli._unit_set, default="all")
    args = parser.parse_args()
    sessions = [session_peaks(args.out / session, args.units) for session in SESSIONS]
    subjects = [data["subject"] for data in sessions]
    changes = [data["peaks"][:, :, 1] - data["peaks"][:, :, 0] for data in sessions]
    stats = {
        "units": {s: len(c) for s, c in zip(subjects, changes, strict=True)},
        "trials_with_both_flashes": {
            s: data["trials"] for s, data in zip(subjects, sessions, strict=True)
        },
    }
    for k, rate in enumerate(RATES):
        stats[f"{rate}: move minus still (spikes/s)"] = {
            "pooled, hierarchical bootstrap": hierarchical_interval(
                [c[:, k] for c in changes]
            ),
            **{s: _change(c[:, k]) for s, c in zip(subjects, changes, strict=True)},
        }
    for name, part in (
        ("model minus observed change", lambda c: c[:, 1] - c[:, 0]),
        ("part of the model change from flash × move", lambda c: c[:, 1] - c[:, 2]),
    ):
        stats[name] = {
            "pooled, hierarchical bootstrap": hierarchical_interval(
                [part(c) for c in changes]
            ),
            **{s: _change(part(c)) for s, c in zip(subjects, changes, strict=True)},
        }
    stats["units_outside_the_peak_axis"] = int(
        sum(
            np.sum((d["peaks"] < PEAK_LIMITS[0]) | (d["peaks"] > PEAK_LIMITS[1]))
            for d in sessions
        )
    )
    stats["units_outside_the_change_axis"] = int(
        sum(np.sum((c < CHANGE_LIMITS[0]) | (c > CHANGE_LIMITS[1])) for c in changes)
    )
    with plt.rc_context(FIGURE_STYLE):
        figure, axes = plt.subplots(
            1, 3, figsize=(7.4, 2.7), layout="constrained", width_ratios=(1, 1, 1.15)
        )
        for axis, k in zip(axes[:2], range(2), strict=True):
            peak_axes(axis)
            pooled_points(
                axis,
                [d["peaks"][:, k, 0] for d in sessions],
                [d["peaks"][:, k, 1] for d in sessions],
                subjects,
                COLORS[k],
                count_hidden=True,
            )
            axis.set_xlabel(f"{RATES[k]} peak, still\n(spikes/s)")
            axis.set_ylabel(f"{RATES[k]} peak, move (spikes/s)")
        change_axis = axes[2]
        for k, color in enumerate(COLORS):
            pooled_strip(
                change_axis,
                k,
                [np.clip(c[:, k], *CHANGE_LIMITS) for c in changes],
                subjects,
                color,
            )
        change_axis.axhline(0, color=REFERENCE_COLOR, lw=0.8)
        change_axis.set_xticks(
            range(len(RATES)), ["observed", "model", "model, no\nflash × move"]
        )
        for tick, color in zip(change_axis.get_xticklabels(), COLORS, strict=True):
            tick.set_color(color)
        change_axis.set_xlim(-0.5, len(RATES) - 0.2)
        change_axis.set_ylim(*CHANGE_LIMITS)
        change_axis.set_ylabel("peak, move minus still (spikes/s)")
        change_axis.set_title(
            f"{sum(len(c) for c in changes)} units, {len(changes)} mice",
            fontsize=6.5,
            loc="left",
        )
        label_panels(axes, x=-0.28, y=1.04)
        print(save_figure(figure, args.out / "flash_by_period"))
    (args.out / "flash_by_period.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
