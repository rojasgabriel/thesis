"""Compare the response to a flash in the fixation and after it, with no model.

Question: V1 units fire less in the fixation (first flash to center exit) than
after it. Is that the time in the stimulus train (slow adaptation), the
movement after the exit, or the period of the trial (the view from the center
port or the task state)?

Flashes: a flash is "in the fixation" when its whole window (`WINDOW_MS`) ends
before the center exit, and "after the exit" when it comes at or after the exit
and its window ends before the response entry. The first flash of a trial is
not used. Two groups of flashes are always matched on a grid: each cell with
flashes of both groups gets the same weight in both (the smaller of its two
counts). All comparisons are paired over units.

- Time in the train: the fixation against after the exit, matched for the time
  from the first flash (`TIME_EDGES_S`) and the gap from the flash before
  (`GAP_EDGES_S`). Slow adaptation predicts no difference at a matched time.
  This needs trials with different fixation lengths; the script gives the
  distribution of the length and the number of flashes that could be matched.
- The same comparison matched for the gap only (not for time), and for long
  and short gaps apart.
- Movement: among the flashes after the exit, the upper against the lower half
  of the motion energy of the video at the flash, matched for the time from
  the first flash, the gap, and the time from the exit. A gain that comes with
  the movement predicts a larger response with more motion energy.
- Side: among the flashes after the exit, trials with a response to the left
  port against to the right port (`response` of the trial table is 1 for the
  right port), with the same matching. The stimulus panel is above the center
  port, so it moves to the other side of the visual field when the animal
  turns. The recording is in left V1 (right visual field).
- Time from the exit: among the flashes after the exit, 0 to 100 ms against
  later, matched for the time from the first flash and the gap.
- Measures: the baseline (rate from `BASELINE_MS` around the flash, before the
  response), the response (rate from `RESPONSE_MS` after the flash), their
  log ratio between the two groups, and the difference of the two log ratios
  (is the response changed by more than the baseline). Peak latency: the time
  of the maximum of the smoothed response, for units with a response above
  the baseline in both groups (`MIN_RESPONSE_RATIO`).
- From the saved fit: is the fixation level of a unit (mean of its fixation
  kernel over `LEVEL_MS`) related to the unique ΔD² of its other regressors,
  to its rate, or to its depth (Spearman)? A change of the gain of the whole
  unit predicts no relation with the size of its flash response.

Units are those of `fit.py` (`--units`), and the trials are the trials of the
GLM. Bins after the response entry are not used. The sampling unit is the
unit; the units share the trials, so the tests over units do not show that a
result holds for other trials. The script writes `matched_flashes.json` and
`matched_flashes.npz` (the flash-aligned rates of each unit) in the session
directory.

    uv run python -m thesis.ephys.analyses.glm.matched_flashes
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr, wilcoxon

from thesis.ephys.analyses.glm import cli, dmat
from thesis.ephys.analyses.glm.fit import analysis_unit_indices, load_unit_records
from thesis.ephys.analyses.glm.hold_effect import LEVEL_MS
from thesis.ephys.analyses.glm.unique_dev import UNIQUE
from thesis.ephys.units import fetch_unit_table

WINDOW_MS = (-40, 150)
BASELINE_MS = (-20, 20)
RESPONSE_MS = (30, 120)
TIME_EDGES_S = np.arange(0.2, 0.91, 0.1)
GAP_EDGES_S = (0.0, 0.05, 0.1, np.inf)
SINCE_EXIT_EDGES_S = (0.0, 0.1, 0.25, np.inf)
MIN_RESPONSE_RATIO = 1.2
SMOOTH_MS = 5


def flash_table(source: Path) -> dict[str, np.ndarray]:
    """Return the time, period, and matching cell of every flash that is used."""
    prepared = dmat._load_windows(source / "stimulus_windows_me.npz")
    trials = dmat._load_selected_trials(prepared)
    with np.load(source / "video_me_features.npz", allow_pickle=False) as video:
        frame_times, frame_rows, energy = (
            video["frame_times_s"],
            video["frame_trial_row"],
            video["total"],
        )
    window_s = WINDOW_MS[1] / 1000
    out = {
        key: [] for key in ("time", "after", "cell", "since_exit", "energy", "right")
    }
    for row, (side, flashes, entry, leave, response) in enumerate(
        zip(
            trials["response"],
            trials["stim_pulse_times_s"],
            trials["center_entry_s"],
            trials["center_exit_s"],
            trials["response_port_entry_s"],
            strict=True,
        )
    ):
        flashes = np.asarray(flashes, dtype=float)
        flashes = flashes[(flashes >= entry) & (flashes <= response)]
        times, gaps = flashes[1:], np.diff(flashes)
        in_fixation = times + window_s <= leave
        after = (times >= leave) & (times + window_s <= response)
        time_cell = np.digitize(times - flashes[0], TIME_EDGES_S) - 1
        gap_cell = np.digitize(gaps, GAP_EDGES_S) - 1
        keep = (
            (in_fixation | after)
            & (time_cell >= 0)
            & (time_cell < len(TIME_EDGES_S) - 1)
        )
        frames = frame_rows == row
        out["time"].append(times[keep])
        out["after"].append(after[keep])
        out["cell"].append((time_cell * (len(GAP_EDGES_S) - 1) + gap_cell)[keep])
        out["since_exit"].append((times - leave)[keep])
        out["right"].append(np.full(keep.sum(), side == 1))
        out["energy"].append(
            np.interp(times[keep], frame_times[frames], energy[frames])
        )
    return {key: np.concatenate(values) for key, values in out.items()}


def matched_weights(cell: np.ndarray, first: np.ndarray, second: np.ndarray):
    """Return weights that give each cell the same total in the two groups."""
    weights = [np.zeros(len(cell)), np.zeros(len(cell))]
    for value in np.unique(cell):
        members = [(cell == value) & group for group in (first, second)]
        shared = min(member.sum() for member in members)
        for weight, member in zip(weights, members, strict=True):
            if shared:
                weight[member] = shared / member.sum()
    return weights


def flash_counts(spikes: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Return the spike count in each 1-ms bin of `WINDOW_MS` around each flash."""
    edges = times[:, None] + np.arange(WINDOW_MS[0], WINDOW_MS[1] + 1) / 1000
    return np.diff(np.searchsorted(spikes, edges), axis=1)


def _mean_rate(rates: np.ndarray, limits_ms: tuple[int, int]) -> np.ndarray:
    lags = np.arange(*WINDOW_MS)
    return rates[..., (lags >= limits_ms[0]) & (lags < limits_ms[1])].mean(-1)


def _paired(first: np.ndarray, second: np.ndarray) -> dict:
    """Summarize log(second / first) over the units where both are above zero."""
    used = (first > 0) & (second > 0)
    ratio = np.log(second[used] / first[used])
    return {
        "units": int(used.sum()),
        "median_log_ratio": float(np.median(ratio)),
        "median_ratio": float(np.exp(np.median(ratio))),
        "units_higher": int(np.sum(ratio > 0)),
        "wilcoxon_p": float(wilcoxon(ratio).pvalue),
    }


def _latency(values: np.ndarray) -> dict:
    """Compare the peak time of the smoothed response between the two groups."""
    lags = np.arange(*WINDOW_MS)
    smooth = gaussian_filter1d(values, SMOOTH_MS, axis=-1)
    searched = (lags >= 20) & (lags < 130)
    baseline = _mean_rate(values, BASELINE_MS)
    responsive = (smooth[..., searched].max(-1) > MIN_RESPONSE_RATIO * baseline).all(
        0
    ) & (baseline > 0).all(0)
    latency = lags[searched][smooth[..., searched].argmax(-1)][:, responsive]
    shift = latency[1] - latency[0]
    return {
        "units": int(responsive.sum()),
        "median_first_group": float(np.median(latency[0])),
        "median_second_group": float(np.median(latency[1])),
        "units_earlier_in_second_group": int(np.sum(shift < 0)),
        "units_later_in_second_group": int(np.sum(shift > 0)),
        "wilcoxon_p": float(wilcoxon(shift).pvalue),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=cli.PoissonGLM().root)
    parser.add_argument("--units", type=cli._unit_set, default="all")
    args = parser.parse_args()
    table = flash_table(args.root)
    after, fixation = table["after"], ~table["after"]
    gaps = len(GAP_EDGES_S) - 1
    gap = table["cell"] % gaps
    since = np.digitize(table["since_exit"], SINCE_EXIT_EDGES_S) - 1
    with_exit = table["cell"] * (len(SINCE_EXIT_EDGES_S) - 1) + since
    energy_split = np.median(table["energy"][after])
    # name: (grid to match on, first group, second group)
    comparisons = {
        "fixation against after exit, matched time and gap": (
            table["cell"],
            fixation,
            after,
        ),
        "fixation against after exit, matched gap": (gap, fixation, after),
        "fixation against after exit, long gaps": (
            gap,
            fixation & (gap == gaps - 1),
            after & (gap == gaps - 1),
        ),
        "fixation against after exit, short gaps": (
            gap,
            fixation & (gap == 0),
            after & (gap == 0),
        ),
        "after exit, lower against upper half of motion energy": (
            with_exit,
            after & (table["energy"] < energy_split),
            after & (table["energy"] >= energy_split),
        ),
        "after exit, response to the left against to the right": (
            with_exit,
            after & ~table["right"],
            after & table["right"],
        ),
        "after exit, first 100 ms against later": (
            table["cell"],
            after & (since == 0),
            after & (since > 0),
        ),
    }
    weights = {name: matched_weights(*groups) for name, groups in comparisons.items()}

    prepared = dmat._load_windows(args.root / "stimulus_windows_me.npz")
    trials = dmat._load_selected_trials(prepared)
    length = 1000 * (
        trials["center_exit_s"].to_numpy(dtype=float) - prepared["alignments"]
    )
    units = fetch_unit_table(
        prepared["metadata"]["subject_name"],
        prepared["metadata"]["session_name"],
        unit_criteria_id=1,
        stability_param_id=0,
        include_metrics=False,
    )
    units = units.iloc[analysis_unit_indices(units, args.units)]
    unit_ids = units["unit_id"].to_numpy(dtype=int)
    # rates[name][group, unit, lag]: the weighted mean rate in spikes/s.
    rates = {name: np.zeros((2, len(units), np.ptp(WINDOW_MS))) for name in comparisons}
    for position, spikes in enumerate(units["spike_times_s"]):
        counts = flash_counts(np.asarray(spikes, dtype=float), table["time"])
        for name, pair in weights.items():
            for group, weight in enumerate(pair):
                rates[name][group, position] = (
                    weight @ counts / weight.sum() / dmat.BINWIDTH_S
                )

    stats = {
        "units": len(units),
        "trials": len(trials),
        "fixation_length_ms_percentiles_1_5_50_95_99": np.percentile(
            length, [1, 5, 50, 95, 99]
        ).tolist(),
        "flashes_in_fixation": int(fixation.sum()),
        "flashes_after_exit": int(after.sum()),
    }
    for name, values in rates.items():
        baseline = _mean_rate(values, BASELINE_MS)
        response = _mean_rate(values, RESPONSE_MS)
        both = (baseline > 0).all(0) & (response > 0).all(0)
        difference = np.log(response[1, both] / response[0, both]) - np.log(
            baseline[1, both] / baseline[0, both]
        )
        first, second = weights[name]
        stats[name] = {
            "flashes_used": [int(np.sum(first > 0)), int(np.sum(second > 0))],
            "median_motion_energy": [
                float(np.median(table["energy"][first > 0])),
                float(np.median(table["energy"][second > 0])),
            ],
            "baseline, second / first group": _paired(baseline[0], baseline[1]),
            "response, second / first group": _paired(response[0], response[1]),
            "response more than baseline (difference of log ratios)": {
                "units": int(both.sum()),
                "median": float(np.median(difference)),
                "units_above_0": int(np.sum(difference > 0)),
                "wilcoxon_p": float(wilcoxon(difference).pvalue),
            },
            "peak latency (ms)": _latency(values),
        }
    lags = np.arange(*WINDOW_MS)

    # From the saved fit: the fixation level against the flash response and depth.
    fit_dir = cli.PoissonGLM(root=args.root).fit_dir(args.units)
    metadata = json.loads(
        cli.PoissonGLM(root=args.root).design.with_suffix(".json").read_text()
    )
    columns = [
        index
        for index, name in enumerate(metadata["column_names"])
        if name.startswith(dmat.HOLD)
    ]
    basis = dmat.task_temporal_bases()[dmat.HOLD]
    in_level = (1000 * basis.basis_time >= LEVEL_MS[0]) & (
        1000 * basis.basis_time < LEVEL_MS[1]
    )
    scale = np.asarray(metadata["training_scale"])[columns]
    final = {
        int(item["unit_id"]): item
        for item in load_unit_records(fit_dir / "checkpoints", "unit_*_final.json")
    }
    level = np.asarray(
        [
            (
                (np.asarray(final[unit]["coefficients"])[columns] / scale)
                @ basis.basis.T
            )[in_level].mean()
            for unit in unit_ids
        ]
    )
    records = {
        int(item["unit_id"]): item["groups"]
        for item in load_unit_records(fit_dir / "checkpoints" / "unique", "unit_*.json")
    }
    properties = {
        f"unique dD2, {group}": [records[unit][group][UNIQUE] for unit in unit_ids]
        for group in ("visual_flash", "flash_adaptation", "video", "history")
    } | {
        "firing rate": [final[unit]["fit_mean_count"] for unit in unit_ids],
        "depth": [final[unit]["depth"] for unit in unit_ids],
    }
    for name, values in properties.items():
        result = spearmanr(level, values)
        stats[f"fixation level against {name} (Spearman)"] = {
            "rho": float(result.statistic),
            "p": float(result.pvalue),
        }

    np.savez(
        args.root / "matched_flashes.npz",
        unit_ids=unit_ids,
        lags_ms=lags,
        fixation_level=level,
        flash_unique=np.asarray(properties["unique dD2, visual_flash"]),
        **{f"rate__{name}": values for name, values in rates.items()},
    )
    output = args.root / "matched_flashes.json"
    output.write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))
    print(output)


if __name__ == "__main__":
    main()
