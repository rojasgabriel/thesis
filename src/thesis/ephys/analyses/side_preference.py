"""Test whether the side preference of V1 units after the center exit needs the stimulus.

Question: after the animal leaves the center port, many units fire more for a
response to one side. The stimulus panel is above the center port, so it moves
in the visual field when the animal turns. Is the side preference a response
to that view of the stimulus, or a signal of the movement or of the place of
the animal?

Comparison: the side index of each unit in three windows of the completed
trials, with no model. The stimulus train is 1 s long and stops before the
response entry in most trials, and the arena is dark, so the time between the
last flash and the response entry has the movement with no light.

- Side index: (right − left) / (right + left) of the rate of a unit, where the
  rate is the spike count of the trials of a side over their time in the
  window. `response` of the trial table is 1 for the right port and −1 for the
  left port.
- Stimulus on: from `AFTER_EXIT_S` after the center exit to `AFTER_FLASH_ON_S`
  after the last flash (or the response entry).
- Stimulus off: from `AFTER_FLASH_OFF_S` after the last flash to the response
  entry.
- Fixation: from `AFTER_FIRST_FLASH_S` after the first flash to the center exit.
- A trial is in a window when the window is `MIN_WINDOW_S` or longer.
- Units: `fetch_unit_table` with quality criteria 1 and stability 0. The
  sampling unit is the unit.
- Statistics: Spearman correlation of the index between windows, over units.
  The split-half correlation of each window (random halves of its trials,
  `SPLIT_SEED`) is the limit that noise sets on those correlations.
- Limit: the window with the stimulus off is later in the movement than the
  window with the stimulus on. The units share the trials.

The script writes `side_preference.json` in
`figures/side_preference/<subject>_<session>/`.

    uv run python -m thesis.ephys.analyses.side_preference
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from thesis.ephys.trials import build_trial_table
from thesis.ephys.units import fetch_unit_table

AFTER_EXIT_S = 0.1
AFTER_FLASH_ON_S = 0.1
AFTER_FLASH_OFF_S = 0.15
AFTER_FIRST_FLASH_S = 0.15
MIN_WINDOW_S = 0.05
LARGE_INDEX = 0.2
SPLIT_SEED = 0
WINDOWS = ("stimulus on", "stimulus off", "fixation")


def trial_windows(trials) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Return the (start, stop) of each window and the side of each completed trial."""
    windows = {name: [] for name in WINDOWS}
    right = []
    for _, trial in trials.iterrows():
        flashes = np.atleast_1d(np.asarray(trial["stim_pulse_times_s"], dtype=float))
        flashes = flashes[np.isfinite(flashes)]
        leave, response = trial["center_exit_s"], trial["response_port_entry_s"]
        if (
            trial["early_withdrawal"] == 1
            or trial["response"] not in (-1, 1)
            or not len(flashes)
            or not np.isfinite(leave)
            or not np.isfinite(response)
        ):
            continue
        last = flashes[flashes <= response][-1]
        windows["stimulus on"].append(
            (leave + AFTER_EXIT_S, min(last + AFTER_FLASH_ON_S, response))
        )
        windows["stimulus off"].append((last + AFTER_FLASH_OFF_S, response))
        windows["fixation"].append((flashes[0] + AFTER_FIRST_FLASH_S, leave))
        right.append(trial["response"] == 1)
    return {name: np.asarray(values) for name, values in windows.items()}, np.asarray(
        right
    )


def side_index(counts, seconds, right, used) -> float:
    """Return (right − left) / (right + left) of the rate over the trials in `used`."""
    rates = [
        counts[used & side].sum() / max(seconds[used & side].sum(), 1e-9)
        for side in (right, ~right)
    ]
    return (rates[0] - rates[1]) / sum(rates) if sum(rates) > 0 else np.nan


def _correlation(first: np.ndarray, second: np.ndarray) -> dict:
    both = np.isfinite(first) & np.isfinite(second)
    result = spearmanr(first[both], second[both])
    return {
        "units": int(both.sum()),
        "rho": float(result.statistic),
        "p": float(result.pvalue),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--subject", default="GRB006")
    parser.add_argument("--session", default="20240821_121447")
    parser.add_argument("--out", type=Path, default=Path("figures/side_preference"))
    args = parser.parse_args()
    trials = build_trial_table(args.subject, args.session, include_frames=False)
    units = fetch_unit_table(
        args.subject,
        args.session,
        unit_criteria_id=1,
        stability_param_id=0,
        include_metrics=False,
    )
    windows, right = trial_windows(trials)
    rng = np.random.default_rng(SPLIT_SEED)
    index, halves = {}, {}
    stats = {
        "subject": args.subject,
        "session": args.session,
        "units": len(units),
        "completed_trials": len(right),
        "trials_to_the_right": int(right.sum()),
    }
    for name, window in windows.items():
        seconds = window[:, 1] - window[:, 0]
        used = seconds >= MIN_WINDOW_S
        half = rng.permutation(len(used)) % 2 == 0
        values = []
        for spikes in units["spike_times_s"]:
            spikes = np.asarray(spikes, dtype=float)
            counts = np.searchsorted(spikes, window[:, 1]) - np.searchsorted(
                spikes, window[:, 0]
            )
            values.append(
                [
                    side_index(counts, seconds, right, used & part)
                    for part in (np.ones_like(used), half, ~half)
                ]
            )
        values = np.asarray(values)
        index[name], halves[name] = values[:, 0], values[:, 1:]
        finite = values[np.isfinite(values[:, 0]), 0]
        stats[name] = {
            "trials": int(used.sum()),
            "median_window_ms": float(1000 * np.median(seconds[used])),
            "median_index": float(np.median(finite)),
            "median_size_of_index": float(np.median(np.abs(finite))),
            "fraction_units_with_large_index": float(
                np.mean(np.abs(finite) > LARGE_INDEX)
            ),
            "split_half": _correlation(*halves[name].T),
            "index_by_unit": values[:, 0].tolist(),
        }
    for first, second in (
        ("stimulus on", "stimulus off"),
        ("stimulus on", "fixation"),
        ("stimulus off", "fixation"),
    ):
        stats[f"{first} against {second}"] = _correlation(index[first], index[second])
    output = args.out / f"{args.subject}_{args.session}" / "side_preference.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(stats, indent=1))
    for name, values in stats.items():
        if isinstance(values, dict):
            values = {k: v for k, v in values.items() if k != "index_by_unit"}
        print(f"{name}: {values}")
    print(output)


if __name__ == "__main__":
    main()
