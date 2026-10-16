"""Test whether V1 firing drops after the center entry, before the first flash.

Question: V1 units fire less while the animal is in the center port with the
stimulus on. Does the drop start with the stimulus (the first flash) or with
the entry into the port? The time from the center entry to the first flash
differs between trials, so the time between the two events has the port and
no stimulus.

Comparison: in the trials with a long delay from the center entry to the first
flash (the upper half of the delays), the rate of each unit from
`ENTRY_LAG_MS` after the entry to the first flash, against its rate in
`BEFORE_ENTRY_MS` before the entry, with no model. The comparison is paired
over units. A drop that needs the stimulus gives a ratio near 1; a drop that
starts at the entry gives a ratio below 1. The visual response to the first
flash starts later than the flash, so the window has no flash response.

- Trials: every trial with a flash, a center entry before the first flash
  (the last one before it, at most `MAX_DELAY_S` before), and a center exit
  `MIN_FIXATION_S` or more after the first flash. Early withdrawals are in.
- Rate of a unit in a window: its spikes in the window of all long-delay
  trials over the summed length of the windows. A unit with no spike before
  the entry has no ratio; the count is in the output.
- Reference: the rate in `LATE_MS` after the first flash (the full drop),
  over the same rate before the entry.
- Statistics: the median ratio and a two-sided Wilcoxon signed-rank test of
  the log ratio over units. The sampling unit is the unit; the units share the
  trials.
- Population rate (for the figure): for each trial, the mean over units of
  the rate in `BIN_S` bins, with the rate of each unit divided by its mean over
  the window, in the short and the long delays, aligned to the center entry
  and to the first flash.
- Units: `fetch_unit_table` with quality criteria 1 and stability 0.
- Limit: the rate can already change before the entry, while the animal
  comes to the port; the window before the entry ends `BEFORE_ENTRY_MS[1]` ms
  before it.

The script writes `suppression_onset.json` in
`figures/suppression_onset/<subject>_<session>/`.

    uv run python -m thesis.ephys.analyses.suppression_onset
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from thesis.ephys.trials import build_trial_table
from thesis.ephys.units import fetch_unit_table

BIN_S = 0.01
WINDOW_S = (-0.3, 0.3)
MAX_DELAY_S = 0.3
MIN_FIXATION_S = 0.25
BEFORE_ENTRY_MS = (-250, -100)
ENTRY_LAG_MS = 20
LATE_MS = (180, 290)


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


def window_rates(spikes: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> float:
    """Return the spikes in the windows over their summed length (spikes/s)."""
    counts = np.searchsorted(spikes, ends) - np.searchsorted(spikes, starts)
    return float(counts.sum() / np.sum(ends - starts))


def _paired(ratio: np.ndarray) -> dict:
    used = np.isfinite(ratio) & (ratio > 0)
    log_ratio = np.log(ratio[used])
    return {
        "units": int(used.sum()),
        "median_ratio": float(np.exp(np.median(log_ratio))),
        "units_lower": int(np.sum(log_ratio < 0)),
        "wilcoxon_p": float(wilcoxon(log_ratio).pvalue),
    }


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
        stats[f"aligned to the {name}"] = {
            "mean_rate": {
                half: rate[members].mean(0).tolist()
                for half, members in (("short delay", short), ("long delay", ~short))
            }
        }
    entry, flash = events[~short, 0], events[~short, 1]
    windows = {
        "before entry": (
            entry + BEFORE_ENTRY_MS[0] / 1000,
            entry + BEFORE_ENTRY_MS[1] / 1000,
        ),
        "entry to first flash": (entry + ENTRY_LAG_MS / 1000, flash),
        "late": (flash + LATE_MS[0] / 1000, flash + LATE_MS[1] / 1000),
    }
    rates = np.asarray(
        [
            [
                window_rates(np.asarray(spikes, dtype=float), *w)
                for w in windows.values()
            ]
            for spikes in units["spike_times_s"]
        ]
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        pre_flash = rates[:, 1] / rates[:, 0]
        late = rates[:, 2] / rates[:, 0]
    stats["short-delay trials"] = {
        "trials": int(short.sum()),
        "delay_ms_by_trial": (1000 * (events[short, 1] - events[short, 0])).tolist(),
    }
    stats["long-delay trials"] = {
        "trials": int((~short).sum()),
        "median_window_entry_to_flash_ms": float(
            1000 * np.median(flash - entry) - ENTRY_LAG_MS
        ),
        "delay_ms_by_trial": (1000 * (flash - entry)).tolist(),
        "units_with_no_spike_before_entry": int(np.sum(rates[:, 0] == 0)),
        "entry to first flash over before entry": _paired(pre_flash),
        "late over before entry": _paired(late),
        "ratio_by_unit": {
            "entry to first flash over before entry": np.where(
                np.isfinite(pre_flash), pre_flash, np.nan
            ).tolist(),
            "late over before entry": np.where(
                np.isfinite(late), late, np.nan
            ).tolist(),
        },
    }
    output = args.out / f"{args.subject}_{args.session}" / "suppression_onset.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(stats, indent=1))
    for name, values in stats.items():
        if isinstance(values, dict):
            values = {
                k: v
                for k, v in values.items()
                if k not in ("mean_rate", "ratio_by_unit", "delay_ms_by_trial")
            }
        if name != "time_ms":
            print(f"{name}: {values}")
    print(output)


if __name__ == "__main__":
    main()
