"""Compare the firing rate around the center exit with and without the stimulus.

Question: V1 units fire less while the animal is in the center port with the
stimulus on, and more after it leaves. Does the rise at the exit need the
stimulus? In a completed trial the flashes go on after the exit. In an
early-withdrawal trial the animal leaves before the go cue and the flashes
stop at the exit.

Comparison: the rate around the first center exit after the first flash, in
completed trials against early-withdrawal trials of one session, with no
model. The comparison is paired over units.

- Units: `fetch_unit_table` with quality criteria 1 and stability 0.
- Trials: all trials with at least one flash and a center exit after the
  first flash. A completed trial also has a response-port entry.
- Alignment: the center exit (`center_exit_s`; for an early withdrawal, the
  first time in `center_port_exit_times_s` after the first flash).
- Normalization: the rate of a unit is divided by its mean rate over the whole
  window around the first flash of all these trials.
- Measures: the rate from `BEFORE_MS` and from `AFTER_MS` around the exit, and
  their ratio; two-sided Wilcoxon signed-rank tests over units.
- Limits: the exit of an early withdrawal is earlier in the stimulus train, and
  an early withdrawal is followed by its punishment, a 2 s tone with no light
  (the arena stays dark). The
  units share the trials, so the tests do not show that a result holds for
  other trials.

The script writes `exit_without_stimulus.json` in
`figures/exit_without_stimulus/<subject>_<session>/`.

    uv run python -m thesis.ephys.analyses.exit_without_stimulus
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from thesis.ephys.trials import build_trial_table
from thesis.ephys.units import fetch_unit_table

BIN_S = 0.05
WINDOW_S = (-0.3, 0.5)
BEFORE_MS = (-200, 0)
AFTER_MS = (100, 400)


def _finite(values) -> np.ndarray:
    values = np.atleast_1d(np.asarray(values, dtype=float))
    return values[np.isfinite(values)]


def exit_times(trials) -> dict[str, np.ndarray]:
    """Return (first flash, center exit) of each completed and early trial."""
    events = {"completed": [], "early withdrawal": []}
    for _, trial in trials.iterrows():
        flashes = _finite(trial["stim_pulse_times_s"])
        if not len(flashes):
            continue
        if trial["early_withdrawal"] == 1:
            exits = _finite(trial["center_port_exit_times_s"])
            exits = exits[exits > flashes[0]]
            if len(exits):
                events["early withdrawal"].append((flashes[0], exits[0]))
        elif np.isfinite(trial["center_exit_s"]) and np.isfinite(
            trial["response_port_entry_s"]
        ):
            events["completed"].append((flashes[0], trial["center_exit_s"]))
    return {name: np.asarray(values) for name, values in events.items()}


def _rate(spikes: np.ndarray, times: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts = np.diff(np.searchsorted(spikes, times[:, None] + edges), axis=1)
    return counts.mean(0) / BIN_S


def _paired(first: np.ndarray, second: np.ndarray) -> dict:
    used = (first > 0) & (second > 0)
    ratio = np.log(second[used] / first[used])
    return {
        "units": int(used.sum()),
        "median_ratio": float(np.exp(np.median(ratio))),
        "units_higher": int(np.sum(ratio > 0)),
        "wilcoxon_p": float(wilcoxon(ratio).pvalue),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--subject", default="GRB064")
    parser.add_argument("--session", default="20260916_132939")
    parser.add_argument(
        "--out", type=Path, default=Path("figures/exit_without_stimulus")
    )
    args = parser.parse_args()
    trials = build_trial_table(args.subject, args.session, include_frames=False)
    units = fetch_unit_table(
        args.subject,
        args.session,
        unit_criteria_id=1,
        stability_param_id=0,
        include_metrics=False,
    )
    events = exit_times(trials)
    edges = np.arange(WINDOW_S[0], WINDOW_S[1] + BIN_S / 2, BIN_S)
    centers = 1000 * (edges[:-1] + edges[1:]) / 2
    first_flashes = np.concatenate([values[:, 0] for values in events.values()])
    rates = {name: [] for name in events}
    for spikes in units["spike_times_s"]:
        spikes = np.asarray(spikes, dtype=float)
        scale = _rate(spikes, first_flashes, edges).mean()
        if scale <= 0:
            continue
        for name, values in events.items():
            rates[name].append(_rate(spikes, values[:, 1], edges) / scale)
    rates = {name: np.asarray(values) for name, values in rates.items()}
    before = (centers >= BEFORE_MS[0]) & (centers < BEFORE_MS[1])
    after = (centers >= AFTER_MS[0]) & (centers < AFTER_MS[1])
    level = {
        name: (values[:, before].mean(1), values[:, after].mean(1))
        for name, values in rates.items()
    }
    stats = {
        "subject": args.subject,
        "session": args.session,
        "units": len(next(iter(rates.values()))),
        "time_from_exit_ms": centers.tolist(),
    }
    for name, values in rates.items():
        stats[name] = {
            "trials": len(events[name]),
            "median_first_flash_to_exit_ms": float(
                1000 * np.median(events[name][:, 1] - events[name][:, 0])
            ),
            "mean_normalized_rate": values.mean(0).tolist(),
            "median_rate_before_exit": float(np.median(level[name][0])),
            "median_rate_after_exit": float(np.median(level[name][1])),
            "after / before exit": _paired(*level[name]),
        }
    for index, period in enumerate(("before", "after")):
        stats[f"early withdrawal / completed, {period} exit"] = _paired(
            level["completed"][index], level["early withdrawal"][index]
        )
    output = args.out / f"{args.subject}_{args.session}" / "exit_without_stimulus.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))
    print(output)


if __name__ == "__main__":
    main()
