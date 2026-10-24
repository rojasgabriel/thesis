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

- The fixation against after the exit, matched for the gap from the flash
  before (`GAP_EDGES_S`).
- Movement: among the flashes after the exit, the upper against the lower half
  of the motion energy of the video at the flash, matched for the time from
  the first flash (`TIME_EDGES_S`), the gap, and the time from the exit. A
  gain that comes with the movement predicts a larger response with more
  motion energy.
- Measures: the baseline (rate from `BASELINE_MS` around the flash, before the
  response), the response (rate from `RESPONSE_MS` after the flash), their
  log ratio between the two groups, and the difference of the two log ratios
  (is the response changed by more than the baseline).
- From the saved fit: is the fixation level of a unit (mean of its fixation
  kernel over `LEVEL_MS`) related to the unique ΔD² of its flash terms
  (Spearman)? A change of the gain of the whole unit predicts no relation.
  (The other comparisons of 2026-10-11 are in docs/glm_decisions.md and the
  git history.)

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
from scipy.stats import spearmanr, wilcoxon

from thesis.ephys.analyses.glm import cli, dmat
from thesis.ephys.analyses.glm.fit import analysis_unit_indices, load_unit_records
from thesis.ephys.analyses.glm.hold_effect import LEVEL_MS
from thesis.ephys.analyses.glm.unique_dev import UNIQUE_BITS, add_bits_per_spike
from thesis.ephys.units import fetch_unit_table

WINDOW_MS = (-40, 150)
BASELINE_MS = (-20, 20)
RESPONSE_MS = (30, 120)
TIME_EDGES_S = np.arange(0.2, 0.91, 0.1)
GAP_EDGES_S = (0.0, 0.05, 0.1, np.inf)
SINCE_EXIT_EDGES_S = (0.0, 0.1, 0.25, np.inf)
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
    out = {key: [] for key in ("time", "after", "cell", "since_exit", "energy")}
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
        "fixation against after exit, matched gap": (gap, fixation, after),
        "after exit, lower against upper half of motion energy": (
            with_exit,
            after & (table["energy"] < energy_split),
            after & (table["energy"] >= energy_split),
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
        # The spike count in each 1-ms bin of `WINDOW_MS` around each flash.
        edges = (
            table["time"][:, None] + np.arange(WINDOW_MS[0], WINDOW_MS[1] + 1) / 1000
        )
        counts = np.diff(
            np.searchsorted(np.asarray(spikes, dtype=float), edges), axis=1
        )
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
        for item in add_bits_per_spike(
            load_unit_records(fit_dir / "checkpoints" / "unique", "unit_*.json"),
            fit_dir / "checkpoints",
        )
    }
    # "flash": the flash terms that code the gap (with flash recovery), together.
    gap_terms = [*dmat.gap_flash_terms(), dmat.FLASH_RECOVERY]
    properties = {
        "unique bits per spike, flash": [
            sum(
                records[unit][group][UNIQUE_BITS]
                for group in gap_terms
                if group in records[unit]
            )
            for unit in unit_ids
        ]
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
        flash_unique=np.asarray(properties["unique bits per spike, flash"]),
        **{f"rate__{name}": values for name, values in rates.items()},
    )
    output = args.root / "matched_flashes.json"
    output.write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))
    print(output)


if __name__ == "__main__":
    main()
