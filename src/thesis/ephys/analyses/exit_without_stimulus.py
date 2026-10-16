"""Compare the firing rate around the center exit with and without the stimulus.

Question: V1 units fire less while the animal is in the center port with the
stimulus on, and more after it leaves. Does the rise at the exit need the
stimulus? In a completed trial the flashes go on after the exit. In an
early-withdrawal trial the animal leaves before the go cue and the flashes
stop at the exit.

Comparison: the rate around the first center exit after the first flash, in
completed trials against early-withdrawal trials of one session, with no
model. The comparison is paired over units.

- Units: the units of the GLM (`fetch_unit_table` with quality criteria 1
  and stability 0, without the units of `fit.EXCLUDED_UNITS`).
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

- Panel a (observation): the rate against time from the center exit, in
  completed trials (the stimulus goes on) and in early withdrawals (the
  stimulus stops). Mean and SEM over units.
- Panel b (the paired comparison): the rate of each unit after the exit
  (`AFTER_MS`) in completed trials against early withdrawals. Error bars are
  the SD over `BOOTSTRAPS` resamples of the trials, in both directions. A unit
  with a rate of zero in a condition is not on the logarithmic axes; the
  number of those units is in the output.

The script writes `exit_without_stimulus.svg` and `exit_without_stimulus.json`
in `figures/exit_without_stimulus/<subject>_<session>/`.

    uv run python -m thesis.ephys.analyses.exit_without_stimulus
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem, wilcoxon

from thesis.ephys.analyses.glm.fit import analysis_unit_indices
from thesis.ephys.trials import build_trial_table
from thesis.ephys.units import fetch_unit_table
from thesis.plotting import (
    CONCEPT_COLORS,
    FIGURE_STYLE,
    REFERENCE_COLOR,
    label_panels,
    save_figure,
)

BIN_S = 0.05
WINDOW_S = (-0.3, 0.5)
BEFORE_MS = (-200, 0)
AFTER_MS = (100, 400)
BOOTSTRAPS = 200
SEED = 0
# One color for one concept: blue while the stimulus is on.
COLORS = {"completed": CONCEPT_COLORS["visual"], "early withdrawal": "0.35"}
LABELS = {
    "completed": "completed trial:\nthe stimulus goes on",
    "early withdrawal": "early withdrawal:\nthe stimulus stops",
}


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
    # The units of the GLM: its excluded units are not used here either.
    units = units.iloc[analysis_unit_indices(units, "all")]
    events = exit_times(trials)
    edges = np.arange(WINDOW_S[0], WINDOW_S[1] + BIN_S / 2, BIN_S)
    centers = 1000 * (edges[:-1] + edges[1:]) / 2
    first_flashes = np.concatenate([values[:, 0] for values in events.values()])
    rates = {name: [] for name in events}
    spread = {name: [] for name in events}
    rng = np.random.default_rng(SEED)
    window = np.asarray(AFTER_MS) / 1000
    for spikes in units["spike_times_s"]:
        spikes = np.asarray(spikes, dtype=float)
        scale = _rate(spikes, first_flashes, edges).mean()
        if scale <= 0:
            continue
        for name, values in events.items():
            rates[name].append(_rate(spikes, values[:, 1], edges) / scale)
            # SD of the rate after the exit over resamples of the trials.
            counts = np.diff(
                np.searchsorted(spikes, values[:, 1][:, None] + window), axis=1
            )[:, 0]
            draws = rng.integers(len(counts), size=(BOOTSTRAPS, len(counts)))
            spread[name].append(counts[draws].mean(1).std() / np.ptp(window) / scale)
    rates = {name: np.asarray(values) for name, values in rates.items()}
    spread = {name: np.asarray(values) for name, values in spread.items()}
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
            "sem_normalized_rate": sem(values, axis=0).tolist(),
            # One row for each unit, for figures that pool the units of mice.
            "normalized_rate_by_unit": values.tolist(),
            "median_rate_before_exit": float(np.median(level[name][0])),
            "median_rate_after_exit": float(np.median(level[name][1])),
            "after / before exit": _paired(*level[name]),
        }
    for index, period in enumerate(("before", "after")):
        stats[f"early withdrawal / completed, {period} exit"] = _paired(
            level["completed"][index], level["early withdrawal"][index]
        )
    x, y = level["completed"][1], level["early withdrawal"][1]
    shown = (x > 0) & (y > 0)
    stats["units_not_on_the_axes_of_panel_b"] = int((~shown).sum())
    folder = args.out / f"{args.subject}_{args.session}"
    with plt.rc_context(FIGURE_STYLE):
        figure, (trace, paired) = plt.subplots(
            1, 2, figsize=(6.6, 2.7), gridspec_kw={"width_ratios": (1.5, 1)}
        )
        for name, values in rates.items():
            mean, error = values.mean(0), sem(values, axis=0)
            trace.plot(centers, mean, color=COLORS[name])
            trace.fill_between(
                centers,
                mean - error,
                mean + error,
                color=COLORS[name],
                alpha=0.25,
                lw=0,
            )
        trace.text(
            0.02,
            0.98,
            LABELS["completed"],
            color=COLORS["completed"],
            transform=trace.transAxes,
            va="top",
        )
        trace.text(
            0.02,
            0.72,
            LABELS["early withdrawal"],
            color=COLORS["early withdrawal"],
            transform=trace.transAxes,
            va="top",
        )
        trace.axvline(0, color=REFERENCE_COLOR, linewidth=0.8)
        trace.set_ylim(bottom=0)
        trace.set_xlabel("time from center exit (ms)")
        trace.set_ylabel("normalized firing rate")
        paired.errorbar(
            x[shown],
            y[shown],
            xerr=spread["completed"][shown],
            yerr=spread["early withdrawal"][shown],
            fmt="o",
            color="black",
            markersize=2.5,
            elinewidth=0.5,
            alpha=0.6,
        )
        limits = (
            0.7 * min(x[shown].min(), y[shown].min()),
            1.4 * max(x[shown].max(), y[shown].max()),
        )
        paired.plot(
            limits, limits, color=REFERENCE_COLOR, linestyle="--", linewidth=0.8
        )
        paired.set_xscale("log")
        paired.set_yscale("log")
        paired.set_xlim(limits)
        paired.set_ylim(limits)
        paired.set_aspect("equal")
        paired.set_xlabel("rate after exit,\nthe stimulus goes on")
        paired.set_ylabel("rate after exit,\nthe stimulus stops")
        figure.tight_layout()
        label_panels((trace, paired), x=-0.22)
        print(save_figure(figure, folder / "exit_without_stimulus"))
    output = folder / "exit_without_stimulus.json"
    output.write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))
    print(output)


if __name__ == "__main__":
    main()
