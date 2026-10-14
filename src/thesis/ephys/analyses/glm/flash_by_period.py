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
  `PSTH_BIN_MS` bins; the baseline is the still rate from `BASELINE_MS`, and
  the same baseline is taken from both periods; the peak is the largest
  baseline-corrected rate with a bin center in `PEAK_MS`, found in each period
  apart (spikes/s).
- Units: the units with the "excited" classification of StimulusResponse
  (stimulus-response parameter 0) that the GLM fits, as in the raw figure. A
  peak has no meaning for a unit with no response to the flash.
- Each test is within one animal (`SESSIONS`); the units of the two animals are
  not pooled. Statistics: two-sided Wilcoxon signed-rank tests over units of
  move minus still in each rate, of the observed against the model change, and
  of the model against the model with no flash × move. Summary: the median of
  the units of each animal with its bootstrap 95% interval.
- Figure: a, b, observed and model peaks, still against move; c, move minus
  still in the three rates. Both animals in each axis, one marker for each
  animal. A point outside the axis limits is not drawn; it is in the
  statistics, and the count is in the output.

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

from thesis.ephys.analyses.glm import cli, dmat
from thesis.ephys.analyses.glm.fit import build_unit_counts, load_unit_records
from thesis.ephys.analyses.glm.flash_kernels import SESSIONS
from thesis.ephys.units import fetch_unit_table
from thesis.plotting import (
    ANIMAL_MARKERS,
    FIGURE_STYLE,
    MODEL_COLOR,
    OBSERVED_COLOR,
    REFERENCE_COLOR,
    animal_points,
    label_panels,
    median_interval,
    save_figure,
)

PSTH_BIN_MS = 10
BASELINE_MS = (-40, 0)
PEAK_MS = (30, 120)
STIM_RESPONSE_PARAM_ID = 0
INTERACTION = (dmat.FLASH_AFTER_EXIT, dmat.FLASH_MOVE_GAP)
RATES = ("observed", "model", "model, no flash × move")
COLORS = (OBSERVED_COLOR, MODEL_COLOR, "0.55")
PEAK_LIMITS = (-15, 110)
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


def excited_units(subject: str, session: str) -> set[int]:
    """Return the units with the "excited" classification of StimulusResponse."""
    from labdata_plugin.schema import StimulusResponse

    query = {
        "subject_name": subject,
        "session_name": session,
        "unit_criteria_id": 1,
        "stim_response_param_id": STIM_RESPONSE_PARAM_ID,
        "response_type": "excited",
    }
    return {int(unit) for unit in (StimulusResponse.Unit & query).fetch("unit_id")}


def peaks(trace: np.ndarray, lags: np.ndarray) -> np.ndarray:
    """Return the peak of each period: trace is (period, 1-ms bin) in spikes/s."""
    binned = trace.reshape(len(trace), -1, PSTH_BIN_MS).mean(-1)
    centers = lags.reshape(-1, PSTH_BIN_MS).mean(-1) + 0.5
    baseline = binned[
        0, (centers >= BASELINE_MS[0]) & (centers < BASELINE_MS[1])
    ].mean()
    window = (centers >= PEAK_MS[0]) & (centers < PEAK_MS[1])
    return (binned[:, window] - baseline).max(1)


def session_peaks(root: Path, unit_set: str) -> dict:
    """Return the peaks of each unit: (unit, rate, period) in spikes/s."""
    model = cli.PoissonGLM(root=root)
    prepared = dmat._load_windows(model.windows)
    trials = dmat._load_selected_trials(prepared)
    alignments, relative = prepared["alignments"], prepared["relative_centers"]
    grid = (len(alignments), len(relative))
    valid = dmat._valid_bin_mask(prepared)
    rows, still, move = paired_flashes(trials)
    lags = np.arange(BASELINE_MS[0], PEAK_MS[1])
    columns = np.stack(
        [
            np.rint((times - alignments[rows] - relative[0]) / dmat.BINWIDTH_S).astype(
                int
            )[:, None]
            + lags[None, :]
            for times in (still, move)
        ]
    )  # (period, trial, bin)
    if columns.min() < 0 or columns.max() >= grid[1]:
        raise ValueError("A response window is outside the model grid.")
    subject, session = (
        prepared["metadata"][key] for key in ("subject_name", "session_name")
    )
    units = fetch_unit_table(
        subject,
        session,
        unit_criteria_id=1,
        stability_param_id=0,
        include_metrics=False,
    )
    excited = excited_units(subject, session)
    metadata = json.loads(model.design.with_suffix(".json").read_text())
    interaction = [
        index
        for index, name in enumerate(metadata["column_names"])
        if name.startswith(tuple(f"{term}_basis" for term in INTERACTION))
    ]
    common = np.load(model.design, mmap_mode="r", allow_pickle=False)
    final = {
        int(item["unit_id"]): item
        for item in load_unit_records(
            model.fit_dir(unit_set) / "checkpoints", "unit_*_final.json"
        )
    }
    out, unit_ids = [], []
    for unit_id, spikes in zip(units["unit_id"], units["spike_times_s"], strict=True):
        if int(unit_id) not in final or int(unit_id) not in excited:
            continue
        record = final[int(unit_id)]
        spikes = np.asarray(spikes, dtype=float)
        observed = build_unit_counts(alignments, spikes).reshape(grid)
        width = int(metadata["base_columns"]) + int(record["components"])
        weights = np.asarray(record["coefficients"], dtype=float)
        history, _, _ = dmat.training_zscore(
            dmat.build_unit_history(alignments, spikes), valid
        )
        log_rate = (
            float(record["intercept"])
            + np.asarray(common[:, :width]) @ weights[:width]
            + history @ weights[width:]
        )
        without = log_rate - np.asarray(common[:, interaction]) @ weights[interaction]
        sources = (
            observed,
            np.exp(log_rate).reshape(grid),
            np.exp(without).reshape(grid),
        )
        traces = (
            np.stack(
                [
                    np.stack(
                        [source[rows[:, None], columns[p]].mean(0) for p in range(2)]
                    )
                    for source in sources
                ]
            )
            / dmat.BINWIDTH_S
        )  # (rate, period, bin)
        out.append([peaks(trace, lags) for trace in traces])
        unit_ids.append(int(unit_id))
    return {
        "subject": subject,
        "peaks": np.asarray(out),  # (unit, rate, period)
        "unit_ids": unit_ids,
        "trials": len(rows),
    }


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
    stats = {}
    for session, data in zip(SESSIONS, sessions, strict=True):
        values = data["peaks"]
        change = values[:, :, 1] - values[:, :, 0]  # (unit, rate)
        stats[session] = {
            "trials_with_both_flashes": data["trials"],
            "units": len(values),
            **{
                f"{rate}: move minus still (spikes/s)": _change(change[:, k])
                for k, rate in enumerate(RATES)
            },
            "model minus observed change": _change(change[:, 1] - change[:, 0]),
            "part of the model change from flash × move": _change(
                change[:, 1] - change[:, 2]
            ),
            "units_outside_the_peak_axis": int(
                np.sum((values < PEAK_LIMITS[0]) | (values > PEAK_LIMITS[1]))
            ),
            "units_outside_the_change_axis": int(
                np.sum((change < CHANGE_LIMITS[0]) | (change > CHANGE_LIMITS[1]))
            ),
        }
    with plt.rc_context(FIGURE_STYLE):
        figure, axes = plt.subplots(
            1, 3, figsize=(7.4, 2.7), layout="constrained", width_ratios=(1, 1, 1.15)
        )
        for axis, k in zip(axes[:2], range(2), strict=True):
            for data in sessions:
                values = data["peaks"][:, k]
                animal_points(
                    axis,
                    np.clip(values[:, 0], *PEAK_LIMITS),
                    np.clip(values[:, 1], *PEAK_LIMITS),
                    data["subject"],
                    COLORS[k],
                )
            axis.plot(PEAK_LIMITS, PEAK_LIMITS, color=REFERENCE_COLOR, lw=0.8, ls="--")
            axis.set(xlim=PEAK_LIMITS, ylim=PEAK_LIMITS, aspect="equal")
            axis.set_xlabel(f"{RATES[k]} peak, still\n(spikes/s)")
            axis.set_ylabel(f"{RATES[k]} peak, move (spikes/s)")
        change_axis = axes[2]
        rng = np.random.default_rng(0)
        for data in sessions:
            values = data["peaks"]
            change = values[:, :, 1] - values[:, :, 0]
            side = -0.17 if data["subject"] == sessions[0]["subject"] else 0.17
            marker = ANIMAL_MARKERS[data["subject"]]
            for k, color in enumerate(COLORS):
                shown = np.clip(change[:, k], *CHANGE_LIMITS)
                change_axis.scatter(
                    k + side + rng.uniform(-0.07, 0.07, len(shown)),
                    shown,
                    s=9,
                    marker=marker,
                    color=color,
                    alpha=0.3,
                    lw=0,
                )
                median, low, high = median_interval(change[:, k])
                change_axis.errorbar(
                    k + side,
                    median,
                    yerr=[[median - low], [high - median]],
                    fmt=marker,
                    ms=6,
                    color=color,
                    mec="white",
                    mew=0.6,
                    capsize=2,
                    zorder=4,
                )
        change_axis.axhline(0, color=REFERENCE_COLOR, lw=0.8)
        change_axis.set_xticks(
            range(len(RATES)), ["observed", "model", "model, no\nflash × move"]
        )
        for tick, color in zip(change_axis.get_xticklabels(), COLORS, strict=True):
            tick.set_color(color)
        change_axis.set_ylim(*CHANGE_LIMITS)
        change_axis.set_ylabel("peak, move minus still (spikes/s)")
        change_axis.set_title(
            "  ".join(
                f"{ANIMAL_MARKERS[data['subject']].replace('^', '△').replace('o', '○')}"
                f" {data['subject']}"
                for data in sessions
            ),
            fontsize=6.5,
            loc="left",
        )
        label_panels(axes, x=-0.28, y=1.04)
        print(save_figure(figure, args.out / "flash_by_period"))
    (args.out / "flash_by_period.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
