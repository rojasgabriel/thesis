"""Test whether the drop of V1 firing in the center port follows the first flash or the entry.

Question: V1 units fire less while the animal is in the center port with the
stimulus on. Does the drop start with the stimulus (the first flash) or with
the entry into the port? The time from the center entry to the first flash
differs between trials, so the two events can be separated.

Comparison: the trials are cut in two halves by the time from the center entry
to the first flash (short and long delay). The population rate is aligned to
the first flash and to the center entry, with no model. A drop that follows
the first flash comes at the same time from the first flash in both halves,
and later from the center entry in the long-delay half, by the difference of
the delays. A drop that follows the entry does the opposite.

- Trials: every trial with a flash, a center entry before the first flash
  (the last one before it, at most `MAX_DELAY_S` before), and a center exit
  `MIN_FIXATION_S` or more after the first flash. Early withdrawals are in.
- Population rate: for each trial, the mean over units of the rate in
  `BIN_S` bins, with the rate of each unit divided by its mean over the window.
- Time of the drop: the mean over trials is smoothed (Gaussian, `SMOOTH_BINS`).
  The drop is the first bin from `SEARCH_MS` where the rate is below the
  middle of its level before (`BEFORE_MS`) and its late level (`LATE_MS`) and
  stays below it for `HOLD_BINS` bins.
- Uncertainty: `BOOTSTRAPS` resamples of the trials of each half (percentile
  interval, `SEED`). The sampling unit is the trial.
- Units: `fetch_unit_table` with quality criteria 1 and stability 0.
- Limit: the measure needs a flat rate before the events. When the rate goes
  down before the first flash, the time of the drop is not defined; look at
  `mean_rate` in the output.
- Also given: the rate from `LATE_MS` over the rate before, in completed and in
  early-withdrawal trials.

The script writes `suppression_onset.json` in
`figures/suppression_onset/<subject>_<session>/`.

    uv run python -m thesis.ephys.analyses.suppression_onset
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

from thesis.ephys.trials import build_trial_table
from thesis.ephys.units import fetch_unit_table

BIN_S = 0.01
WINDOW_S = (-0.3, 0.3)
MAX_DELAY_S = 0.3
MIN_FIXATION_S = 0.25
BEFORE_MS = (-250, -120)
LATE_MS = (180, 290)
SEARCH_MS = (40, 280)
SMOOTH_BINS = 1.5
HOLD_BINS = 3
BOOTSTRAPS = 500
SEED = 0


def _finite(values) -> np.ndarray:
    values = np.atleast_1d(np.asarray(values, dtype=float))
    return values[np.isfinite(values)]


def trial_events(trials) -> np.ndarray:
    """Return (center entry, first flash, early withdrawal) of each trial that is used."""
    rows = []
    for _, trial in trials.iterrows():
        flashes = _finite(trial["stim_pulse_times_s"])
        entries = _finite(trial["center_port_entry_times_s"])
        exits = _finite(trial["center_port_exit_times_s"])
        if not len(flashes) or not np.any(entries < flashes[0]):
            continue
        entry = entries[entries < flashes[0]][-1]
        later = exits[exits > flashes[0]]
        if (
            len(later)
            and flashes[0] - entry < MAX_DELAY_S
            and later[0] - flashes[0] > MIN_FIXATION_S
        ):
            rows.append((entry, flashes[0], trial["early_withdrawal"] == 1))
    return np.asarray(rows, dtype=float)


def population_rate(units, times: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Return the normalized population rate of each trial in each bin."""
    total = np.zeros((len(times), len(edges) - 1))
    used = 0
    for spikes in units["spike_times_s"]:
        spikes = np.asarray(spikes, dtype=float)
        rate = np.diff(np.searchsorted(spikes, times[:, None] + edges), axis=1) / BIN_S
        if rate.mean() > 0:
            total += rate / rate.mean()
            used += 1
    return total / used


def drop_time(mean_rate: np.ndarray, centers: np.ndarray) -> float:
    """Return the time (ms) when the smoothed rate goes below the middle level."""
    smooth = gaussian_filter1d(mean_rate, SMOOTH_BINS)
    level = lambda limits: smooth[  # noqa: E731
        (centers > limits[0]) & (centers < limits[1])
    ].mean()
    middle = (level(BEFORE_MS) + level(LATE_MS)) / 2
    searched = np.flatnonzero((centers >= SEARCH_MS[0]) & (centers <= SEARCH_MS[1]))
    for index in searched:
        if np.all(smooth[index : index + HOLD_BINS] < middle):
            return float(centers[index])
    return np.nan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--subject", default="GRB006")
    parser.add_argument("--session", default="20240821_121447")
    parser.add_argument("--out", type=Path, default=Path("figures/suppression_onset"))
    args = parser.parse_args()
    trials = build_trial_table(args.subject, args.session, include_frames=False)
    units = fetch_unit_table(
        args.subject,
        args.session,
        unit_criteria_id=1,
        stability_param_id=0,
        include_metrics=False,
    )
    events = trial_events(trials)
    delay = 1000 * (events[:, 1] - events[:, 0])
    early = events[:, 2] == 1
    short = delay < np.median(delay)
    edges = np.arange(WINDOW_S[0], WINDOW_S[1] + BIN_S / 2, BIN_S)
    centers = 1000 * (edges[:-1] + edges[1:]) / 2
    rng = np.random.default_rng(SEED)
    stats = {
        "subject": args.subject,
        "session": args.session,
        "units": len(units),
        "trials": len(events),
        "early_withdrawal_trials": int(early.sum()),
        "delay_ms_percentiles_5_50_95": np.percentile(delay, [5, 50, 95]).tolist(),
        "median_delay_ms_short_and_long": [
            float(np.median(delay[short])),
            float(np.median(delay[~short])),
        ],
        "time_ms": centers.tolist(),
    }
    for name, column in (("first flash", 1), ("center entry", 0)):
        rate = population_rate(units, events[:, column], edges)
        result = {"mean_rate": {}}
        resampled = {}
        for half, members in (("short delay", short), ("long delay", ~short)):
            rows = np.flatnonzero(members)
            resampled[half] = np.asarray(
                [
                    drop_time(rate[rng.choice(rows, len(rows))].mean(0), centers)
                    for _ in range(BOOTSTRAPS)
                ]
            )
            result[half] = {
                "drop_ms": drop_time(rate[rows].mean(0), centers),
                "interval_95": np.nanpercentile(resampled[half], [2.5, 97.5]).tolist(),
            }
            result["mean_rate"][half] = rate[rows].mean(0).tolist()
        difference = resampled["long delay"] - resampled["short delay"]
        result["long minus short"] = {
            "drop_ms": result["long delay"]["drop_ms"]
            - result["short delay"]["drop_ms"],
            "interval_95": np.nanpercentile(difference, [2.5, 97.5]).tolist(),
        }
        stats[f"aligned to the {name}"] = result
        if column == 1:
            before = (centers > BEFORE_MS[0]) & (centers < BEFORE_MS[1])
            late = (centers > LATE_MS[0]) & (centers < LATE_MS[1])
            stats["late rate over the rate before"] = {
                kind: float(rate[rows][:, late].mean() / rate[rows][:, before].mean())
                for kind, rows in (("completed", ~early), ("early withdrawal", early))
                if rows.sum() >= 10
            }
    output = args.out / f"{args.subject}_{args.session}" / "suppression_onset.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(stats, indent=1))
    for name, values in stats.items():
        if isinstance(values, dict) and "mean_rate" in values:
            values = {k: v for k, v in values.items() if k != "mean_rate"}
        if name != "time_ms":
            print(f"{name}: {values}")
    print(output)


if __name__ == "__main__":
    main()
