"""Explain how V1 flash responses change at the center exit.

Question: flash peaks are larger after the exit (result 3 of `docs/glm.md`).
How much of that is a gain on all activity (state), how much is the gap from
the flash before (adaptation), and what is left for the first flash after the
exit?

Comparison: observed responses, and the held-out GLM as a reference; no new
fit. Each flash is grouped by period and rank (in port; the first flash at or
after the exit; later flashes after the exit) and by the gap from the flash
before (`GAPS`). The first flash of a trial (no flash before) is the
unadapted reference. Short-gap flashes (`dmat.short_gap_s()` or less) are
left out: the response to the flash before falls into their windows.

- Trials, alignment and bins: the completed trials of the GLM on its 1-ms
  grid; a window must lie fully inside the valid interval.
- Measures, per unit and group, from the trial mean in 10-ms bins: the rate
  before the flash (mean from -40 to 0 ms), the peak above it (largest bin
  from 30 to 120 ms minus that rate), and the model miss (observed peak minus
  the mean peak of `DRAWS` Poisson draws of the held-out model rate of the
  same bins, which carries the same noise bias). Values are divided by the
  mean observed rate of the unit over all valid bins. Latency: the peak bin of
  the pooled mean trace.
- Gain test (panel c): for an after-exit group at a gap, the peak and the
  rate before the flash, each over the in-port value at the same gap. A pure
  gain on all activity puts the point on the identity line.
- Units: the flash-excited units of `raw_video_trust_classification.json`,
  pooled over both mice. Intervals: hierarchical bootstrap (mice, then units),
  95%; ratios are of pooled means. The sampling unit is the unit.
- Panels: a, flash peak by gap; b, latency by gap; c, the gain test; d, the
  model miss by gap. No axis limit hides a point.

The script writes `exit_flash_responses.svg` (figures/glm) and
`exit_flash_responses.json` (results/glm).

    THESIS_CACHE_DIR=figures/cache nice -n 10 uv run python -m \\
        thesis.ephys.analyses.glm.exit_flash_responses
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thesis.ephys.analyses.glm import dmat, flash_movement
from thesis.ephys.analyses.glm.fit import (
    build_unit_counts,
    cross_validation_partitions,
    load_unit_records,
)
from thesis.ephys.analyses.glm.flash_kernels import SESSIONS
from thesis.ephys.analyses.glm.peaks import LAGS_MS
from thesis.ephys.units import fetch_unit_table
from thesis.plotting import FIGURE_STYLE, REFERENCE_COLOR, label_panels, save_figure

OUT = Path("results/glm")
GAPS = ((0.05, 0.1), (0.1, 0.2), (0.2, np.inf))
GAP_LABELS = ("50–100", "100–200", "200+")
MARKERS = ("o", "s", "^")  # the gap in the gain test (panel c)
GROUPS = ("in port", "later after exit", "1st after exit")
# The period colors of the other figures: still (in the port) blue, moving
# (after the exit) orange, the first flash after the exit dark orange.
COLORS = {"in port": "C0", "later after exit": "#FDAE61", "1st after exit": "#B34700"}
FIRST = "first of trial"
DRAWS = 30
SEED = 6
BOOTSTRAPS = 2000
MIN_FLASHES = 12
CENTERS_S = (LAGS_MS.reshape(-1, 10).mean(-1) + 0.5) / 1000
BASELINE = (CENTERS_S >= -0.04) & (CENTERS_S < 0)
PEAK = (CENTERS_S >= 0.03) & (CENTERS_S < 0.12)


def trace(rate: np.ndarray) -> np.ndarray:
    """Return the event mean of (event, 1-ms bin) rates in 10-ms bins."""
    return rate.mean(0).reshape(-1, 10).mean(-1)


def before_and_peak(rate: np.ndarray) -> tuple[float, float]:
    """Return the rate before the flash and the peak above it."""
    values = trace(rate)
    before = values[BASELINE].mean()
    return float(before), float(values[PEAK].max() - before)


def events(trials, prepared) -> dict:
    """Return the window bins (event, bin) of each (group, gap index) and FIRST."""
    n_bins = len(prepared["relative_centers"])
    valid = dmat._valid_bin_mask(prepared).reshape(len(trials), -1)
    found = {}
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
            if not np.isfinite(gap):
                key = FIRST
            elif gap <= dmat.short_gap_s():
                continue
            else:
                index = next(i for i, (lo, hi) in enumerate(GAPS) if lo < gap <= hi)
                group = GROUPS[0] if time < leave else GROUPS[2 if after == 1 else 1]
                key = (group, index)
            found.setdefault(key, []).append(row * n_bins + columns)
    return {k: np.asarray(v) for k, v in found.items() if len(v) >= MIN_FLASHES}


def session_values(session: str, excited: list[int]) -> dict:
    """Return per-unit measures and traces of each event group of one session."""
    root = OUT / session
    metadata = json.loads((root / "common_design_me.json").read_text())
    records = {
        int(item["unit_id"]): item
        for item in load_unit_records(
            root / "all_fit_me/checkpoints", "unit_*_folds.json"
        )
    }
    subject, stamp = session.split("_", 1)
    table = fetch_unit_table(subject, stamp, 1, 0, include_metrics=False).set_index(
        "unit_id"
    )
    prepared = dmat._load_windows(root / "stimulus_windows_me.npz")
    trials = dmat._load_selected_trials(prepared)
    windows = events(trials, prepared)
    common = np.load(root / "common_design_me.npy", mmap_mode="r", allow_pickle=False)
    valid = dmat._valid_bin_mask(prepared)
    partitions = cross_validation_partitions(
        len(trials), len(prepared["relative_centers"]), valid
    )
    requested = np.zeros(len(valid), dtype=bool)
    requested[np.concatenate([w.ravel() for w in windows.values()])] = True
    rng = np.random.default_rng(SEED)
    values = {
        key: {"before": [], "peak": [], "miss": [], "trace": []} for key in windows
    }
    for unit_id in sorted(set(excited) & set(records)):
        spikes = table.loc[unit_id, "spike_times_s"]
        counts = build_unit_counts(prepared["alignments"], spikes)
        history = dmat.build_unit_history(prepared["alignments"], spikes)
        predicted = flash_movement.held_out_rates(
            counts, history, common, metadata, records[unit_id], partitions, requested
        )[0]
        scale = (counts * valid).sum() / (valid.sum() * dmat.BINWIDTH_S)
        to_rate = 1 / (dmat.BINWIDTH_S * scale)
        for key, window in windows.items():
            before, peak = before_and_peak(counts[window] * to_rate)
            noise = np.mean(
                [
                    before_and_peak(rng.poisson(predicted[window]) * to_rate)[1]
                    for _ in range(DRAWS)
                ]
            )
            values[key]["before"].append(before)
            values[key]["peak"].append(peak)
            values[key]["miss"].append(peak - noise)
            values[key]["trace"].append(trace(counts[window] * to_rate))
        print(f"{session}: unit {unit_id}", flush=True)
    return {
        "subject": subject,
        "flashes": {str(k): len(w) for k, w in windows.items()},
        "values": {
            key: {name: np.asarray(v) for name, v in measures.items()}
            for key, measures in values.items()
        },
    }


def draws(sessions: list[dict], seed: int = 0) -> list[list[np.ndarray]]:
    """Return hierarchical bootstrap indices: for each draw, the units of each mouse.

    A draw picks mice with replacement, then units of each picked mouse. The
    indices are shared by every group, so the groups stay paired.
    """
    rng = np.random.default_rng(seed)
    counts = [len(next(iter(s["values"].values()))["peak"]) for s in sessions]
    out = []
    for _ in range(BOOTSTRAPS):
        mice = rng.integers(0, len(sessions), len(sessions))
        out.append([(m, rng.integers(0, counts[m], counts[m])) for m in mice])
    return out


def pooled(sessions, key, name, picks=None) -> float:
    """Return the mean over the pooled units (or one bootstrap draw) of a measure."""
    if picks is None:
        return float(np.concatenate([s["values"][key][name] for s in sessions]).mean())
    return float(
        np.concatenate([sessions[m]["values"][key][name][i] for m, i in picks]).mean()
    )


def interval(sessions, function, boot) -> list[float]:
    """Return the 95% bootstrap interval of a function of a draw."""
    return np.percentile([function(picks) for picks in boot], [2.5, 97.5]).tolist()


def latency(sessions, key, picks=None) -> float:
    """Return the peak time (ms) of the pooled mean trace."""
    if picks is None:
        traces = np.concatenate([s["values"][key]["trace"] for s in sessions])
    else:
        traces = np.concatenate(
            [sessions[m]["values"][key]["trace"][i] for m, i in picks]
        )
    mean = traces.mean(0)
    return float(1000 * CENTERS_S[np.argmax(np.where(PEAK, mean, -np.inf))])


def main() -> None:
    excited = json.loads((OUT / "raw_video_trust_classification.json").read_text())
    sessions = [session_values(s, sorted(excited[s])) for s in SESSIONS]
    keys = [k for k in sessions[0]["values"] if all(k in s["values"] for s in sessions)]
    boot = draws(sessions)
    stats = {"flashes": {s["subject"]: s["flashes"] for s in sessions}, "groups": {}}
    for key in keys:
        entry = {}
        for name in ("before", "peak", "miss"):
            entry[name] = pooled(sessions, key, name)
            entry[f"{name}_interval"] = interval(
                sessions, lambda p, k=key, n=name: pooled(sessions, k, n, p), boot
            )
        entry["latency_ms"] = latency(sessions, key)
        entry["latency_interval_ms"] = interval(
            sessions, lambda p, k=key: latency(sessions, k, p), boot
        )
        stats["groups"][str(key)] = entry
    gain = {}
    for group in GROUPS[1:]:
        for index in range(len(GAPS)):
            key, port = (group, index), (GROUPS[0], index)
            if key not in keys or port not in keys:
                continue

            def ratio(picks, name, key=key, port=port):
                return pooled(sessions, key, name, picks) / pooled(
                    sessions, port, name, picks
                )

            gain[str(key)] = {
                name: {
                    "ratio": ratio(None, name),
                    "interval": interval(sessions, lambda p, n=name: ratio(p, n), boot),
                }
                for name in ("before", "peak")
            }
    stats["gain_test"] = gain
    gap_x = np.arange(len(GAPS))
    with plt.rc_context(FIGURE_STYLE):
        figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), layout="constrained")
        peak_axis, latency_axis, gain_axis, miss_axis = axes.flat
        for axis, measure in (
            (peak_axis, "peak"),
            (latency_axis, "latency"),
            (miss_axis, "miss"),
        ):
            for offset, group in zip((-0.12, 0, 0.12), GROUPS, strict=True):
                xs, ys, lo, hi = [], [], [], []
                for index in range(len(GAPS)):
                    entry = stats["groups"].get(str((group, index)))
                    if entry is None:
                        continue
                    name = "latency_ms" if measure == "latency" else measure
                    band = (
                        "latency_interval_ms"
                        if measure == "latency"
                        else f"{measure}_interval"
                    )
                    xs.append(index + offset)
                    ys.append(entry[name])
                    lo.append(entry[name] - entry[band][0])
                    hi.append(entry[band][1] - entry[name])
                axis.errorbar(
                    xs,
                    ys,
                    yerr=[lo, hi],
                    fmt="o-",
                    color=COLORS[group],
                    ms=4,
                    capsize=2,
                    lw=1,
                )
            first = stats["groups"].get(FIRST)
            if first is not None:
                name = "latency_ms" if measure == "latency" else measure
                band = (
                    "latency_interval_ms"
                    if measure == "latency"
                    else f"{measure}_interval"
                )
                axis.errorbar(
                    [len(GAPS) + 0.2],
                    [first[name]],
                    yerr=[
                        [first[name] - first[band][0]],
                        [first[band][1] - first[name]],
                    ],
                    fmt="s",
                    color=COLORS["in port"],
                    mfc="white",
                    ms=5,
                    capsize=2,
                    lw=1,
                )
            axis.set_xticks([*gap_x, len(GAPS) + 0.2], [*GAP_LABELS, "first\nof trial"])
            axis.set_xlabel("gap from the flash before (ms)")
        peak_axis.set_ylabel("flash peak (/ mean rate)")
        latency_axis.set_ylabel("peak latency (ms)")
        miss_axis.set_ylabel("observed − model + noise\npeak (/ mean rate)")
        miss_axis.axhline(0, color=REFERENCE_COLOR, lw=0.8)
        for row, group in enumerate(GROUPS):
            peak_axis.text(
                0.03,
                0.97 - 0.09 * row,
                group,
                color=COLORS[group],
                transform=peak_axis.transAxes,
                va="top",
            )
        x_limit = 1.15 * max(g["before"]["interval"][1] for g in gain.values())
        y_limit = 1.3 * max(g["peak"]["interval"][1] for g in gain.values())  # key room
        limit = max(x_limit, y_limit)
        gain_axis.plot([0, limit], [0, limit], color=REFERENCE_COLOR, lw=0.8, ls="--")
        for key_text, entry in gain.items():
            group = next(g for g in GROUPS if g in key_text)
            index = int(key_text.rstrip(")").split(",")[-1])
            x, y = entry["before"], entry["peak"]
            gain_axis.errorbar(
                [x["ratio"]],
                [y["ratio"]],
                xerr=[[x["ratio"] - x["interval"][0]], [x["interval"][1] - x["ratio"]]],
                yerr=[[y["ratio"] - y["interval"][0]], [y["interval"][1] - y["ratio"]]],
                fmt=MARKERS[index],
                color=COLORS[group],
                ms=4.5,
                capsize=2,
                lw=0.8,
            )
        gain_axis.set(xlim=(0, x_limit), ylim=(0, y_limit))
        gain_axis.set_xlabel("rate before the flash,\nafter exit / in port")
        gain_axis.set_ylabel("flash peak, after exit / in port")
        gain_axis.text(
            0.03,
            0.97,
            "dashed: one gain on all activity\ngap (ms): "
            + ", ".join(f"{s} {g}" for s, g in zip("○□△", GAP_LABELS, strict=True)),
            transform=gain_axis.transAxes,
            ha="left",
            va="top",
            fontsize=6,
        )
        label_panels(axes.flat, x=-0.22, y=1.05)
        print(save_figure(figure, OUT / "exit_flash_responses"))
    (OUT / "exit_flash_responses.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats["gain_test"], indent=1))


if __name__ == "__main__":
    main()
