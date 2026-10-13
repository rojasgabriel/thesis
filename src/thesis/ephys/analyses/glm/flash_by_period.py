"""Compare the flash response when the animal is still and after it leaves the center port.

Question: the rate of a V1 unit after a flash is higher after the animal
leaves the center port than in the fixation
(`stim_responses_by_locomotion.py`). Is that a change of the response to the
flash, or a change of the rate level that is not specific to the flash?

Comparison: "still" is a flash in the fixation and "move" is a flash after the
center exit. These are task periods; the script does not measure locomotion,
and the exit also changes the view of the stimulus. Each unit has one value
for each period (paired), in two measures:

- Raw response (panel a). In each trial, two flashes: the last flash whose
  response window ends before the center exit, and the first flash at or
  after the center exit whose response window ends before the response entry.
  Only trials with both flashes are used. The response is the mean rate from
  `RESPONSE_MS` after the flash, over the trials (spikes/s), from the spike
  counts in the 1-ms bins of the model, with no baseline subtraction. So it
  has the flash response and also everything else that changes at the exit.
  Units with a response below `MIN_RATE` spikes/s in a period are left out
  (the log needs a rate above 0); the count is in the output.
- Flash kernel (panel b). The response of the flash kernel of the model, the
  mean of the kernel from `RESPONSE_MS` (Δ log rate), for a flash at a gap of
  80 ms or more in each period (`flash_kernels.npz`). The model takes the
  other things out: the fixation kernel (the low rate in the fixation), the
  withdrawal, the choice, motion energy, and spike history. So this is the
  change of the response to the flash alone. All fitted units are in.
- Uncertainty: for the raw response, the SD over `BOOTSTRAPS` resamples of
  the trials (`SEED`); for the kernel, the SD over the 10 cross-validation
  folds. Error bars are in both directions.
- Statistics: two-sided Wilcoxon signed-rank test over units of move minus
  still, for the log of the raw response and for the kernel response.
- Check of the model (in the output, not drawn): the change of the raw
  response that the final refit predicts for the same flashes, and the sum
  over units of the predicted over the observed response in each period.
- Figure: one row for each session. A point outside the axis limits is not
  drawn; it is in the statistics, and the count is in the output. In panel b
  the color is the kind of the unit (`flash_kernels.py`, from the response at
  the 40 ms gap, which is not one of the two compared responses).

Trials, alignment (first flash), and exclusions are those of `fit.py`. The
sampling unit is the unit. The script writes `flash_by_period.svg` and
`flash_by_period.json` in `figures/glm/`. Run `flash_kernels` first.

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
from thesis.ephys.analyses.glm.fit import (
    analysis_unit_indices,
    build_unit_counts,
    load_unit_records,
)
from thesis.ephys.analyses.glm.flash_kernels import (
    AXIS_LIMITS,
    KIND_COLORS,
    KIND_LIMIT,
    OTHER_COLOR,
    PAIRS,
)
from thesis.ephys.units import fetch_unit_table
from thesis.plotting import (
    FIGURE_STYLE,
    OBSERVED_COLOR,
    REFERENCE_COLOR,
    label_panels,
    save_figure,
)

SESSIONS = ("GRB006_20240821_121447", "GRB064_20260916_132939")
RESPONSE_MS = (40, 100)
MIN_RATE = 1.0
BOOTSTRAPS = 500
SEED = 0
RATE_LIMITS = (1, 200)


def paired_flashes(trials) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the trial row and the times of the still and the move flash."""
    window = RESPONSE_MS[1] / 1000
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
        before = flashes[(flashes >= entry) & (flashes + window <= leave)]
        after = flashes[(flashes >= leave) & (flashes + window <= response)]
        if len(before) and len(after):
            rows.append(row)
            still.append(before[-1])
            move.append(after[0])
    return np.asarray(rows), np.asarray(still), np.asarray(move)


def session_responses(root: Path, unit_set: str) -> dict:
    """Return the raw response of each unit: (unit, observed or model, period, trial)."""
    model = cli.PoissonGLM(root=root)
    prepared = dmat._load_windows(model.windows)
    trials = dmat._load_selected_trials(prepared)
    alignments, relative = prepared["alignments"], prepared["relative_centers"]
    grid = (len(alignments), len(relative))
    valid = dmat._valid_bin_mask(prepared)
    rows, still, move = paired_flashes(trials)
    lags = np.arange(*RESPONSE_MS)
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
    units = fetch_unit_table(
        prepared["metadata"]["subject_name"],
        prepared["metadata"]["session_name"],
        unit_criteria_id=1,
        stability_param_id=0,
        include_metrics=False,
    )
    units = units.iloc[analysis_unit_indices(units, unit_set)]
    metadata = json.loads(model.design.with_suffix(".json").read_text())
    common = np.load(model.design, mmap_mode="r", allow_pickle=False)
    final = {
        int(item["unit_id"]): item
        for item in load_unit_records(
            model.fit_dir(unit_set) / "checkpoints", "unit_*_final.json"
        )
    }
    responses, unit_ids = [], []
    for unit_id, spikes in zip(units["unit_id"], units["spike_times_s"], strict=True):
        if int(unit_id) not in final:
            continue
        record = final[int(unit_id)]
        spikes = np.asarray(spikes, dtype=float)
        observed = build_unit_counts(alignments, spikes).reshape(grid)
        width = int(metadata["base_columns"]) + int(record["components"])
        coefficients = np.asarray(record["coefficients"], dtype=float)
        history, _, _ = dmat.training_zscore(
            dmat.build_unit_history(alignments, spikes), valid
        )
        predicted = np.exp(
            float(record["intercept"])
            + np.asarray(common[:, :width]) @ coefficients[:width]
            + history @ coefficients[width:]
        ).reshape(grid)
        responses.append(
            [
                [source[rows[:, None], columns[p]].mean(1) for p in range(2)]
                for source in (observed, predicted)
            ]
        )
        unit_ids.append(int(unit_id))
    return {
        "responses": np.asarray(responses) / dmat.BINWIDTH_S,
        "unit_ids": unit_ids,
        "trials": len(rows),
    }


def _change(values: np.ndarray) -> dict:
    return {
        "units": len(values),
        "median": float(np.median(values)),
        "units_higher_in_move": int((values > 0).sum()),
        "wilcoxon_p": float(wilcoxon(values).pvalue),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("figures/glm"))
    parser.add_argument("--units", type=cli._unit_set, default="all")
    args = parser.parse_args()
    rng = np.random.default_rng(SEED)
    first, second, kind_from = PAIRS["period"]
    stats = {}
    with plt.rc_context(FIGURE_STYLE):
        figure, axes = plt.subplots(
            len(SESSIONS), 2, figsize=(6.2, 3.0 * len(SESSIONS)), layout="constrained"
        )
        for (rates, kernel_axis), session in zip(axes, SESSIONS):
            subject = session.split("_")[0]
            data = session_responses(args.out / session, args.units)
            responses = data["responses"]  # (unit, observed or model, period, trial)
            mean = responses.mean(3)
            used = (mean[:, 0] >= MIN_RATE).all(1)
            picks = rng.integers(0, data["trials"], (BOOTSTRAPS, data["trials"]))
            error = responses[used, 0][:, :, picks].mean(3).std(2)  # (unit, period)
            observed, predicted = mean[used, 0], mean[used, 1]
            log_change = np.log(mean[used, :, 1] / mean[used, :, 0])
            with np.load(args.out / session / "flash_kernels.npz") as saved:
                names = saved["conditions"].tolist()
                response = saved["response"]
                spread = saved["response_sd_over_folds"]
            x, y = response[:, names.index(first)], response[:, names.index(second)]
            x_error = spread[:, names.index(first)]
            y_error = spread[:, names.index(second)]
            reference = response[:, [names.index(name) for name in kind_from]].mean(1)
            outside = (np.minimum(x, y) < AXIS_LIMITS[0]) | (
                np.maximum(x, y) > AXIS_LIMITS[1]
            )
            stats[session] = {
                "trials_with_both_flashes": data["trials"],
                "raw response": {
                    "units_left_out_for_a_low_rate": int((~used).sum()),
                    "median_still_and_move_spikes_per_s": np.median(
                        observed, 0
                    ).tolist(),
                    "log(move / still)": _change(log_change[:, 0]),
                    "units_outside_the_axis_limits": int(
                        np.sum(
                            (observed < RATE_LIMITS[0]).any(1)
                            | (observed > RATE_LIMITS[1]).any(1)
                        )
                    ),
                },
                "flash kernel": {
                    "conditions": [first, second],
                    "move minus still": _change(y - x),
                    "units_outside_the_axis_limits": int(outside.sum()),
                },
                "check of the model": {
                    "median_predicted_log(move / still)": float(
                        np.median(log_change[:, 1])
                    ),
                    "predicted_over_observed_sum_still_and_move": (
                        predicted.sum(0) / observed.sum(0)
                    ).tolist(),
                },
            }
            style = {"fmt": "o", "ms": 2.5, "elinewidth": 0.4, "alpha": 0.8}
            rates.errorbar(
                observed[:, 0],
                observed[:, 1],
                xerr=error[:, 0],
                yerr=error[:, 1],
                color=OBSERVED_COLOR,
                **style,
            )
            rates.plot(RATE_LIMITS, RATE_LIMITS, color=REFERENCE_COLOR, lw=0.8, ls="--")
            rates.set(xscale="log", yscale="log", xlim=RATE_LIMITS, ylim=RATE_LIMITS)
            rates.set_aspect("equal")
            rates.set_xlabel("raw response, still (spikes/s)")
            rates.set_ylabel(f"{subject}\nraw response, move (spikes/s)")
            for members, color in (
                (np.abs(reference) <= KIND_LIMIT, OTHER_COLOR),
                (reference > KIND_LIMIT, KIND_COLORS["fire more"]),
                (reference < -KIND_LIMIT, KIND_COLORS["fire less"]),
            ):
                kernel_axis.errorbar(
                    x[members],
                    y[members],
                    xerr=x_error[members],
                    yerr=y_error[members],
                    color=color,
                    **style,
                )
            kernel_axis.plot(
                AXIS_LIMITS, AXIS_LIMITS, color=REFERENCE_COLOR, lw=0.8, ls="--"
            )
            kernel_axis.axhline(0, color=REFERENCE_COLOR, lw=0.6)
            kernel_axis.axvline(0, color=REFERENCE_COLOR, lw=0.6)
            kernel_axis.set(xlim=AXIS_LIMITS, ylim=AXIS_LIMITS, aspect="equal")
            kernel_axis.set_xlabel("flash kernel, still (Δ log rate)")
            kernel_axis.set_ylabel("flash kernel, move (Δ log rate)")
            for axis, entry in (
                (rates, stats[session]["raw response"]["log(move / still)"]),
                (kernel_axis, stats[session]["flash kernel"]["move minus still"]),
            ):
                axis.text(
                    0.04,
                    0.97,
                    f"higher in move:\n{entry['units_higher_in_move']} of "
                    f"{entry['units']} units",
                    transform=axis.transAxes,
                    va="top",
                    fontsize=6.5,
                )
        for name, height in (("fire more", 0.14), ("fire less", 0.05)):
            axes[0, 1].text(
                0.97,
                height,
                f"{name} after a flash",
                color=KIND_COLORS[name],
                transform=axes[0, 1].transAxes,
                ha="right",
                fontsize=6.5,
            )
        label_panels(axes[0], x=-0.3, y=1.05)
        print(save_figure(figure, args.out / "flash_by_period"))
    (args.out / "flash_by_period.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
