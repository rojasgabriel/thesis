"""Compare the flash response in the fixation and after the exit with the prediction of the GLM.

Question: the rate of a V1 unit after a flash differs between the time when
the animal is still in the center port and the time after it leaves
(`stim_responses_by_locomotion.py`). How much of that change do the other
terms of the model give (the end of the fixation effect, the withdrawal, the
choice, motion energy, spike history), and how much needs a flash response
that differs between the two periods (the flash × move kernel)?

Comparison: in each trial, two flashes. Still: the last flash whose response
window ends before the center exit. Moving: the first flash at or after the
center exit whose response window ends before the response entry. The
comparison is paired: only trials with both flashes are used, and each unit
has one value for each period. "Moving" is a task period; the script does not
measure locomotion, and the exit also changes the view of the stimulus.

- Response: the mean rate from `RESPONSE_MS` after the flash, over the trials
  (spikes/s). There is no baseline subtraction: the change of the rate level
  between the periods is a part of the question.
- Change of a unit: log(response move / response still).
- Observed: the spike counts in the 1-ms bins of the model.
- Model: the rate of the final refit in the same bins (all terms, with the
  observed spike history). The final refit is fit on all trials, so this is
  not a held-out prediction.
- Model without flash × move: the same weights with the weights of the
  flash × move kernel set to 0. The model is not fit again, so the other
  terms cannot take the place of the kernel. The kernel is scaled by a
  centered value, so without it each flash has the average flash kernel of
  the two periods.
- Units: every fitted unit (`--units`) with an observed response of
  `MIN_RATE` spikes/s or more in both periods (the log needs a rate above 0).
  The number of units that are left out is in the output.
- Uncertainty: the SD of the change over `BOOTSTRAPS` resamples of the trials
  (`SEED`), for the observed and for the predicted change (the error bars, in
  both directions).
- Statistics: two-sided Wilcoxon signed-rank test of the observed change
  against 0, and of the observed minus the predicted change against 0;
  Spearman correlation of the observed and the predicted change over units;
  and the fraction of the variance of the observed change that the
  prediction explains, 1 - SS(observed - predicted) / SS(observed - mean).
- Figure: one row for each session. a, the observed response, moving against
  still. b, the observed change against the change that the model predicts
  without the flash × move kernel. c, the same with the full model. A
  point outside the axis limits is not drawn; it is in the statistics, and
  the count is in the output.

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
from scipy.stats import spearmanr, wilcoxon

from thesis.ephys.analyses.glm import cli, dmat
from thesis.ephys.analyses.glm.fit import (
    analysis_unit_indices,
    build_unit_counts,
    load_unit_records,
)
from thesis.ephys.analyses.glm.unique_dev import block_slices
from thesis.ephys.units import fetch_unit_table
from thesis.plotting import (
    CONCEPT_COLORS,
    FIGURE_STYLE,
    MODEL_COLOR,
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
CHANGE_LIMITS = (-1.5, 2.0)
SOURCES = ("observed", "model without flash × move", "model")


def paired_flashes(trials) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the trial row and the times of the still and the moving flash."""
    window = RESPONSE_MS[1] / 1000
    rows, still, moving = [], [], []
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
            moving.append(after[0])
    return np.asarray(rows), np.asarray(still), np.asarray(moving)


def session_responses(root: Path, unit_set: str) -> dict:
    """Return the response of each unit: (unit, source, period, trial), in spikes/s."""
    model = cli.PoissonGLM(root=root)
    prepared = dmat._load_windows(model.windows)
    trials = dmat._load_selected_trials(prepared)
    alignments, relative = prepared["alignments"], prepared["relative_centers"]
    grid = (len(alignments), len(relative))
    valid = dmat._valid_bin_mask(prepared)
    rows, still, moving = paired_flashes(trials)
    lags = np.arange(*RESPONSE_MS)
    columns = np.stack(
        [
            np.rint((times - alignments[rows] - relative[0]) / dmat.BINWIDTH_S).astype(
                int
            )[:, None]
            + lags[None, :]
            for times in (still, moving)
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
        term = block_slices(metadata, int(record["components"]))[dmat.FLASH_AFTER_EXIT]
        coefficients = np.asarray(record["coefficients"], dtype=float)
        history, _, _ = dmat.training_zscore(
            dmat.build_unit_history(alignments, spikes), valid
        )
        log_rate = (
            float(record["intercept"])
            + np.asarray(common[:, :width]) @ coefficients[:width]
            + history @ coefficients[width:]
        )
        without = log_rate - np.asarray(common[:, term]) @ coefficients[term]
        sources = (
            observed,
            np.exp(without).reshape(grid),
            np.exp(log_rate).reshape(grid),
        )
        responses.append(
            [
                [source[rows[:, None], columns[p]].mean(1) for p in range(2)]
                for source in sources
            ]
        )
        unit_ids.append(int(unit_id))
    return {
        "responses": np.asarray(responses) / dmat.BINWIDTH_S,
        "unit_ids": unit_ids,
        "trials": len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("figures/glm"))
    parser.add_argument("--units", type=cli._unit_set, default="all")
    args = parser.parse_args()
    rng = np.random.default_rng(SEED)
    stats = {}
    with plt.rc_context(FIGURE_STYLE):
        figure, axes = plt.subplots(
            len(SESSIONS), 3, figsize=(8.6, 2.9 * len(SESSIONS)), layout="constrained"
        )
        for row_axes, session in zip(axes, SESSIONS):
            data = session_responses(args.out / session, args.units)
            responses = data["responses"]  # (unit, source, period, trial)
            mean = responses.mean(3)
            used = (mean[:, 0] >= MIN_RATE).all(1)
            mean, responses = mean[used], responses[used]
            change = np.log(mean[:, :, 1] / mean[:, :, 0])  # (unit, source)
            picks = rng.integers(0, data["trials"], (BOOTSTRAPS, data["trials"]))
            resampled = np.maximum(responses[:, :, :, picks].mean(4), 1e-3)
            error = np.log(resampled[:, :, 1] / resampled[:, :, 0]).std(2)
            entry = {
                "units": int(used.sum()),
                "units_left_out_for_a_low_rate": int((~used).sum()),
                "trials_with_both_flashes": data["trials"],
                "median_response_still_and_moving": np.median(mean[:, 0], 0).tolist(),
                "observed_change": {
                    "median": float(np.median(change[:, 0])),
                    "units_higher_when_moving": int((change[:, 0] > 0).sum()),
                    "wilcoxon_p": float(wilcoxon(change[:, 0]).pvalue),
                },
            }
            spread = np.sum((change[:, 0] - change[:, 0].mean()) ** 2)
            for index in (1, 2):
                rest = change[:, 0] - change[:, index]
                entry[SOURCES[index]] = {
                    "median_predicted_change": float(np.median(change[:, index])),
                    "median_observed_minus_predicted": float(np.median(rest)),
                    "wilcoxon_p_of_observed_minus_predicted": float(
                        wilcoxon(rest).pvalue
                    ),
                    "spearman_r": float(spearmanr(change[:, 0], change[:, index])[0]),
                    "variance_explained": float(1 - np.sum(rest**2) / spread),
                    "units_outside_the_axis_limits": int(
                        np.sum(
                            (change[:, [0, index]] < CHANGE_LIMITS[0]).any(1)
                            | (change[:, [0, index]] > CHANGE_LIMITS[1]).any(1)
                        )
                    ),
                }
            entry["units_outside_the_rate_limits"] = int(
                np.sum(
                    (mean[:, 0] < RATE_LIMITS[0]).any(1)
                    | (mean[:, 0] > RATE_LIMITS[1]).any(1)
                )
            )
            stats[session] = entry
            subject = session.split("_")[0]
            rates, without, full = row_axes
            rates.scatter(mean[:, 0, 0], mean[:, 0, 1], s=8, color=OBSERVED_COLOR, lw=0)
            rates.plot(RATE_LIMITS, RATE_LIMITS, color=REFERENCE_COLOR, lw=0.8, ls="--")
            rates.set(xscale="log", yscale="log", xlim=RATE_LIMITS, ylim=RATE_LIMITS)
            rates.set_aspect("equal")
            rates.set_xlabel("response, still (spikes/s)")
            rates.set_ylabel(f"{subject}\nresponse, move (spikes/s)")
            rates.text(
                0.04,
                0.96,
                f"{entry['units']} units\nhigher in move: "
                f"{entry['observed_change']['units_higher_when_moving']}",
                transform=rates.transAxes,
                va="top",
                fontsize=6.5,
            )
            for ax, index, color in (
                (without, 1, CONCEPT_COLORS["task"]),
                (full, 2, MODEL_COLOR),
            ):
                ax.errorbar(
                    change[:, index],
                    change[:, 0],
                    xerr=error[:, index],
                    yerr=error[:, 0],
                    fmt="o",
                    ms=2.5,
                    color=color,
                    elinewidth=0.4,
                    alpha=0.8,
                )
                ax.plot(
                    CHANGE_LIMITS, CHANGE_LIMITS, color=REFERENCE_COLOR, lw=0.8, ls="--"
                )
                ax.axhline(0, color=REFERENCE_COLOR, lw=0.6)
                ax.axvline(0, color=REFERENCE_COLOR, lw=0.6)
                ax.set(xlim=CHANGE_LIMITS, ylim=CHANGE_LIMITS, aspect="equal")
                ax.set_xlabel(f"predicted change,\n{SOURCES[index]}", color=color)
                ax.text(
                    0.04,
                    0.96,
                    f"variance explained {entry[SOURCES[index]]['variance_explained']:.2f}",
                    transform=ax.transAxes,
                    va="top",
                    fontsize=6.5,
                    color=color,
                )
            without.set_ylabel("observed change,\nlog(move / still)")
        label_panels(axes[0], x=-0.3, y=1.05)
        print(save_figure(figure, args.out / "flash_by_period"))
    (args.out / "flash_by_period.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
