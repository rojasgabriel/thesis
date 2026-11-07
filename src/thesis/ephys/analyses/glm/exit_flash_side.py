"""Does the flash response after the exit change with time as the head turns?

Questions:

1. Side: the flash × side term of the GLM finds a flash response after the
   exit that differs with the side of the response. If it comes from the LED
   panel moving in the visual field as the head turns, the side difference
   must be small just after the exit and grow later. The GLM kernel is the
   same at all times after the exit, so a growing observed difference against
   a flat model is the test.
2. First flash: the first flash after the exit is larger than an in-port flash
   at the same gap, by more than the gain of the rate before it
   (`exit_flash_responses.py`). If the head movement resets adaptation, the
   excess must grow with the delay of that flash after the exit.

Comparison: observed responses and the held-out GLM ("model + noise": the
mean peak of Poisson draws of the held-out rate, which has the same noise
bias as an observed peak); no new fit. Both mice are analyzed alone (the
flash × side kernels have opposite signs).

- Flashes: completed GLM trials; a window must lie fully inside the valid
  interval (before the response-port entry). Flashes at a gap of
  `dmat.short_gap_s()` or less are left out (the response to the flash before
  falls into their windows), as is the first flash of a trial.
- Side groups: flashes after the exit except the first one, by side of the
  response (right, left), by time from the exit (early: below
  `TIME_SPLIT_S`; late), and by stimulus category (slow, boundary, fast).
  The side is compared within each category (the category goes with the side
  and sets the gaps), so in the slow and fast categories it is also correct
  against error trials. A category counts when both sides have
  `MIN_FLASHES` flashes in that time group.
- First-flash groups: the first flash at or after the exit, by gap class
  (`GAPS`) and by its delay after the exit (below or above the median delay of
  that gap class); reference: in-port flashes of the same gap class.
- Measures, per unit and group, from the trial mean in 10-ms bins: the rate
  before the flash (-40 to 0 ms) and the peak above it (largest bin from 30 to
  120 ms), divided by the mean observed rate of the unit
  (`exit_flash_responses.session_values`).
- Side effect (gain controlled): log(peak right / peak left) minus log(rate
  before right / rate before left), from the means over units, averaged over
  the categories. Zero means the side changes the flash response only as
  much as it changes all activity; > 0, a larger flash response for a right
  turn. The log of the before ratio is also given (the side difference of
  all activity, which the turn term models).
- First-flash excess: log(peak first / peak in port) minus log(before first /
  before in port), at the same gap class.
- Units: the flash-excited units of `raw_video_trust_classification.json`
  (30 / 15). Intervals: bootstrap over units (`BOOTSTRAPS`, 95%), the same
  draws for every group, so late minus early is paired. Units with a peak or
  rate ≤ 0 have no log value and are not drawn as points; the summaries use
  means over all units. The sampling unit is the unit.

The script writes `figures/glm/exit_flash_side.svg` and
`results/glm/exit_flash_side.json`.

    THESIS_CACHE_DIR=figures/cache nice -n 10 uv run python -m \\
        thesis.ephys.analyses.glm.exit_flash_side
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thesis.ephys.analyses.glm import dmat, exit_flash_responses
from thesis.ephys.analyses.glm.flash_kernels import SESSIONS
from thesis.ephys.analyses.glm.peaks import LAGS_MS
from thesis.plotting import (
    FIGURE_STYLE,
    MODEL_COLOR,
    MOUSE_MARKERS,
    OBSERVED_COLOR,
    REFERENCE_COLOR,
    label_panels,
    save_figure,
)

OUT = Path("results/glm")
TIME_SPLIT_S = 0.25
TIMES = ("early", "late")
CATEGORIES = ("low_rate", "boundary", "high_rate")
SIDES = (1, -1)  # right, left
GAPS = exit_flash_responses.GAPS
GAP_LABELS = exit_flash_responses.GAP_LABELS
MIN_FLASHES = 8
BOOTSTRAPS = 2000
SEED = 7
SOURCES = {"observed": ("before", "peak"), "model": ("model_before", "model_peak")}
COLORS = {"observed": OBSERVED_COLOR, "model": MODEL_COLOR}


def grouping(trials, prepared) -> dict:
    """Return the window bins (flash, bin) of each side group, first-flash group and reference."""
    n_bins = len(prepared["relative_centers"])
    valid = dmat._valid_bin_mask(prepared).reshape(len(trials), -1)
    entries = []
    for row, (_, trial) in enumerate(trials.iterrows()):
        times = np.asarray(trial["stim_pulse_times_s"], dtype=float)
        leave = trial["center_exit_s"]
        gaps = np.diff(times, prepend=-np.inf)
        rank = np.cumsum(times >= leave)
        for time, gap, after in zip(times, gaps, rank, strict=True):
            start = time - prepared["alignments"][row] - prepared["relative_centers"][0]
            columns = int(np.rint(start / dmat.BINWIDTH_S)) + LAGS_MS
            if columns[0] < 0 or columns[-1] >= n_bins or not valid[row, columns].all():
                continue
            if not np.isfinite(gap) or gap <= dmat.short_gap_s():
                continue
            index = next(i for i, (lo, hi) in enumerate(GAPS) if lo < gap <= hi)
            window = row * n_bins + columns
            if time < leave:
                entries.append((("in port", index), window))
            elif after == 1:
                entries.append((("first", index, time - leave), window))
            else:
                time_group = TIMES[0] if time - leave < TIME_SPLIT_S else TIMES[1]
                key = (
                    "side",
                    trial["stim_category"],
                    int(trial["response"]),
                    time_group,
                )
                entries.append((key, window))
    found = {}
    for index in range(len(GAPS)):
        delays = [key[2] for key, _ in entries if key[0] == "first" and key[1] == index]
        if not delays:
            continue
        middle = float(np.median(delays))
        for key, window in entries:
            if key[0] == "first" and key[1] == index:
                half = "short delay" if key[2] < middle else "long delay"
                found.setdefault(("first", index, half), []).append(window)
        found[("median delay", index)] = middle
    for key, window in entries:
        if key[0] != "first":
            found.setdefault(key, []).append(window)
    delays = {k: v for k, v in found.items() if k[0] == "median delay"}
    groups = {
        k: np.asarray(v)
        for k, v in found.items()
        if k[0] != "median delay" and len(v) >= MIN_FLASHES
    }
    grouping.median_delay_s = {GAP_LABELS[k[1]]: v for k, v in delays.items()}
    return groups


def pooled(values: dict, key, name: str, units) -> float:
    """Return the mean over the given units of one measure of one group."""
    return float(np.mean(values[key][name][units]))


def side_effect(values: dict, time: str, source: str, units) -> dict:
    """Return the gain-controlled side effect and the log before ratio, averaged over categories."""
    before, peak = SOURCES[source]
    effects, rates, used = [], [], []
    for category in CATEGORIES:
        keys = [("side", category, side, time) for side in SIDES]
        if not all(k in values for k in keys):
            continue
        right, left = keys
        rate = np.log(
            pooled(values, right, before, units) / pooled(values, left, before, units)
        )
        response = np.log(
            pooled(values, right, peak, units) / pooled(values, left, peak, units)
        )
        effects.append(response - rate)
        rates.append(rate)
        used.append(category)
    if not used:
        return {"effect": np.nan, "before": np.nan, "categories": []}
    return {
        "effect": float(np.mean(effects)),
        "before": float(np.mean(rates)),
        "categories": used,
    }


def first_excess(values: dict, index: int, half: str, source: str, units) -> float:
    """Return log(peak first / in port) minus log(before first / in port) at one gap class."""
    before, peak = SOURCES[source]
    first, port = ("first", index, half), ("in port", index)
    if first not in values or port not in values:
        return np.nan
    return float(
        np.log(pooled(values, first, peak, units) / pooled(values, port, peak, units))
        - np.log(
            pooled(values, first, before, units) / pooled(values, port, before, units)
        )
    )


def unit_side_effects(values: dict, time: str) -> np.ndarray:
    """Return each unit's observed side effect (mean over categories); NaN where a log fails."""
    out = []
    for category in CATEGORIES:
        keys = [("side", category, side, time) for side in SIDES]
        if not all(k in values for k in keys):
            continue
        (right, left) = (values[k] for k in keys)
        with np.errstate(divide="ignore", invalid="ignore"):
            effect = np.log(right["peak"] / left["peak"]) - np.log(
                right["before"] / left["before"]
            )
        out.append(np.where(np.isfinite(effect), effect, np.nan))
    if not out:
        return np.array([])
    stacked = np.asarray(out)
    return np.where(np.isfinite(stacked).all(0), np.nanmean(stacked, 0), np.nan)


def with_interval(function, units: int, rng) -> dict:
    """Return a value over all units with its bootstrap interval over units."""
    draws = [function(rng.integers(0, units, units)) for _ in range(BOOTSTRAPS)]
    value = function(np.arange(units))
    return {
        "value": value,
        "interval_95": np.nanpercentile(draws, [2.5, 97.5]).tolist(),
    }


def session_stats(session: str, excited: list[int]) -> dict:
    """Return the side effects and first-flash excesses of one session."""
    data = exit_flash_responses.session_values(session, excited, False, grouping)
    values = data["values"]
    units = len(next(iter(values.values()))["peak"])
    stats = {
        "subject": data["subject"],
        "units": units,
        "flashes": data["flashes"],
        "median_first_flash_delay_ms": {
            k: 1000 * v for k, v in grouping.median_delay_s.items()
        },
        "side": {},
        "first flash": {},
    }
    for source in SOURCES:
        rng = np.random.default_rng(SEED)
        boot = [rng.integers(0, units, units) for _ in range(BOOTSTRAPS)]
        everyone = np.arange(units)
        entry = {}
        for time in TIMES:
            point = side_effect(values, time, source, everyone)
            draws = [side_effect(values, time, source, b) for b in boot]
            entry[time] = {
                **point,
                "effect_95": np.nanpercentile(
                    [d["effect"] for d in draws], [2.5, 97.5]
                ).tolist(),
                "before_95": np.nanpercentile(
                    [d["before"] for d in draws], [2.5, 97.5]
                ).tolist(),
                "_draws": np.asarray([d["effect"] for d in draws]),
            }
        late_minus_early = entry["late"]["_draws"] - entry["early"]["_draws"]
        entry["late minus early"] = {
            "effect": entry["late"]["effect"] - entry["early"]["effect"],
            "interval_95": np.nanpercentile(late_minus_early, [2.5, 97.5]).tolist(),
        }
        for time in TIMES:
            entry[time].pop("_draws")
        stats["side"][source] = entry
        first = {}
        for index, label in enumerate(GAP_LABELS):
            halves = {}
            for half in ("short delay", "long delay"):
                point = first_excess(values, index, half, source, everyone)
                draws = np.asarray(
                    [first_excess(values, index, half, source, b) for b in boot]
                )
                halves[half] = {
                    "excess": point,
                    "interval_95": np.nanpercentile(draws, [2.5, 97.5]).tolist(),
                    "_draws": draws,
                }
            if all(np.isfinite(h["excess"]) for h in halves.values()):
                diff = halves["long delay"]["_draws"] - halves["short delay"]["_draws"]
                halves["long minus short"] = {
                    "excess": halves["long delay"]["excess"]
                    - halves["short delay"]["excess"],
                    "interval_95": np.nanpercentile(diff, [2.5, 97.5]).tolist(),
                }
            for half in ("short delay", "long delay"):
                halves[half].pop("_draws")
            first[label] = halves
        stats["first flash"][source] = first
    stats["unit side effects"] = {
        time: unit_side_effects(values, time).tolist() for time in TIMES
    }
    return stats


def plot(sessions: list[dict]) -> Path:
    """Side effect by time from the exit (top) and first-flash excess by delay (bottom), per mouse."""
    with plt.rc_context(FIGURE_STYLE):
        figure, axes = plt.subplots(
            2, len(sessions), figsize=(3.4 * len(sessions), 5.8), squeeze=False
        )
        offsets = {"observed": -0.08, "model": 0.08}
        for column, stats in enumerate(sessions):
            subject = stats["subject"]
            marker = MOUSE_MARKERS.get(subject, "o")
            top, bottom = axes[0, column], axes[1, column]
            drawn = []
            for x, time in enumerate(TIMES):
                points = np.asarray(stats["unit side effects"][time], dtype=float)
                points = points[np.isfinite(points)]
                jitter = np.random.default_rng(x).uniform(-0.05, 0.05, len(points))
                drawn.append(len(points))
                top.scatter(
                    x - 0.3 + jitter,
                    points,
                    s=6,
                    color=OBSERVED_COLOR,
                    alpha=0.3,
                    lw=0,
                    marker=marker,
                )
            for source in SOURCES:
                entry = stats["side"][source]
                xs = np.arange(len(TIMES)) + offsets[source]
                ys = np.asarray([entry[t]["effect"] for t in TIMES])
                low = ys - np.asarray([entry[t]["effect_95"][0] for t in TIMES])
                high = np.asarray([entry[t]["effect_95"][1] for t in TIMES]) - ys
                top.errorbar(
                    xs,
                    ys,
                    yerr=[low, high],
                    color=COLORS[source],
                    marker=marker,
                    lw=1.2,
                    capsize=2,
                )
            top.axhline(0, color=REFERENCE_COLOR, ls="--", lw=0.8)
            top.set_xticks(range(len(TIMES)))
            top.set_xticklabels(
                [
                    f"{sign} {1000 * TIME_SPLIT_S:.0f} ms\n{n} / {stats['units']} units"
                    for sign, n in zip(("<", "≥"), drawn, strict=True)
                ]
            )
            top.set_xlim(-0.6, len(TIMES) - 0.4)
            top.set_xlabel("time of the flash from the center exit")
            top.set_ylabel(
                "flash response, right over left turn\n(log, gain controlled)"
            )
            top.set_title(
                f"{subject}: {stats['units']} flash-excited units", fontsize=8
            )
            for x_gap, label in enumerate(GAP_LABELS):
                for source in SOURCES:
                    halves = stats["first flash"][source].get(label, {})
                    ys = [
                        halves.get(h, {}).get("excess", np.nan)
                        for h in ("short delay", "long delay")
                    ]
                    if not np.isfinite(ys).any():
                        continue
                    xs = x_gap + np.array([-0.15, 0.15]) + offsets[source] / 2
                    low = [
                        y - halves[h]["interval_95"][0] if np.isfinite(y) else 0
                        for y, h in zip(ys, ("short delay", "long delay"), strict=True)
                    ]
                    high = [
                        halves[h]["interval_95"][1] - y if np.isfinite(y) else 0
                        for y, h in zip(ys, ("short delay", "long delay"), strict=True)
                    ]
                    bottom.errorbar(
                        xs,
                        ys,
                        yerr=[low, high],
                        color=COLORS[source],
                        marker=marker,
                        lw=1.2,
                        capsize=2,
                    )
            bottom.axhline(0, color=REFERENCE_COLOR, ls="--", lw=0.8)
            bottom.set_xticks(range(len(GAP_LABELS)))
            bottom.set_xticklabels([f"{g} ms" for g in GAP_LABELS])
            bottom.set_xlabel(
                "gap before the first flash after the exit\n(left: short delay after the exit; right: long)"
            )
            bottom.set_ylabel("first flash over in-port flash\n(log, gain controlled)")
        for row in range(2):
            lows, highs = zip(*(axis.get_ylim() for axis in axes[row]), strict=True)
            for axis in axes[row]:
                axis.set_ylim(min(lows), max(highs))
        for row, source in enumerate(SOURCES):
            label = "held-out model + noise" if source == "model" else source
            axes[0, 0].text(
                0.98,
                0.12 - 0.08 * row,
                label,
                color=COLORS[source],
                transform=axes[0, 0].transAxes,
                ha="right",
                fontsize=7,
            )
        figure.tight_layout()
        label_panels(axes.ravel())
        return save_figure(figure, OUT / "exit_flash_side")


def main() -> None:
    excited = json.loads((OUT / "raw_video_trust_classification.json").read_text())
    sessions = [session_stats(s, sorted(excited[s])) for s in SESSIONS]
    (OUT / "exit_flash_side.json").write_text(json.dumps(sessions, indent=1))
    print(
        json.dumps(
            [
                {k: v for k, v in s.items() if k != "unit side effects"}
                for s in sessions
            ],
            indent=1,
        )
    )
    print(plot(sessions))


if __name__ == "__main__":
    main()
