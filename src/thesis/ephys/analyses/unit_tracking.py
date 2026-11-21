"""What following one V1 unit over weeks shows about its flash response.

Question: UnitMatch (`preprocessing/unit_match.py`) links units of sessions
that were sorted one by one, from their waveforms only. Is a linked unit the
same unit? And what does a unit followed over days show that one session
cannot: are its flash response, its response class and its movement
modulation properties of the unit that stay over weeks, and does pooling its
sessions resolve the movement modulation of single units?

Units: quality criterion 1 (UnitCount), the UnitMatch input. A track is one
UnitMatch "intermediate" ID; a track with two units in one session (a split
or a merge) is left out of the response analyses. Pairs:
- same unit: two sessions of one track (every pair of its sessions);
- neighbours: units of two different tracks in two sessions, with centroids
  less than `NEIGHBOUR_UM` apart (UnitMatch centroids, before drift
  correction): the closest units that are not the same unit.

Events and trials: first flash = the first flash of each saved task trial
(`task_events.fetch_task_stimulus_pulses`). Movement comparison: completed
trials (no early withdrawal, a left or right response); in each, the last
flash whose window (-40 to 120 ms) is in the port, and the first flash whose
window starts at or after the center exit and ends before the response entry,
so each flash has a baseline in its own period. Only trials with both
(paired).

Measures:
- class: the StimulusResponse rule (first flash; rate 30 to 120 ms against
  -40 to 0 ms, Wilcoxon, BH q < 0.05), computed here for every session with
  `StimulusResponse.compute`, so no table rows are needed. A track's class
  uses the first flashes of all its sessions together.
- first-flash response: trial-mean rate, 10-ms bins, -40 to 150 ms.
- similarity: Pearson r of two traces, each from one half of the trials
  (first or second half in time) or spikes (ACG), so every comparison uses the
  same amount of data: same session = half 1 against half 2; other session =
  half 1 of A against half 2 of B and the reverse, averaged.
- ACG: spike counts at lags of 1 to 50 ms, 1-ms bins.
- flash response for movement: window peak (flash-excited) or window dip
  (flash-suppressed), the mean over ±20 ms around the largest (smallest)
  10-ms bin from 30 to 120 ms (`peaks.window_peak`), minus the -40 to 0 ms
  baseline of the same flash, divided by the mean rate (0.5 s before to
  1.5 s after each first flash, as `exit_flash_pairs.mean_rate`).
  Movement modulation = after exit − in port. A deeper dip after the exit is
  negative.

Uncertainty: the sampling unit is the track. Intervals of pair statistics
resample tracks (`N_BOOT`); intervals of one unit resample its trials.
Mice: GRB064 (15 sessions over 68 days) and GRB061 (6 over 8 days), drawn
with their markers. Sessions without task pulses (GRB064 0826, 0921) have no
class or first-flash response; GRB064 0921 has no trial table either.

Figures (`figures/unit_tracking/`):
- `tracking.svg`, observation: a, every GRB064 unit found in two or more
  sessions, colored by its class in each session; b, c, a flash-excited and a
  flash-suppressed example (waveforms of all sessions overlaid, first-flash
  response in each session); d, share of a session's units found again
  against days between sessions; e, ACG similarity (not used by UnitMatch).
- `responses.svg`, characterization and prediction: a, first-flash
  similarity against days between sessions, for pairs with the same flash
  class (excited or suppressed) in both sessions, so that a neighbour is not
  less similar only because of its class; b, the class on another day (open
  triangles: each mouse);
  c, d, the movement modulation of flash-excited and flash-suppressed units
  in two consecutive sessions (density contours with 50% filled and 90% as a
  line; the cross is the median 95% interval of one session on both axes;
  axes at the 98th percentile of |modulation|, the pairs past them are
  counted and stay in ρ); e, movement modulation of single units with all
  their sessions pooled, against the one session with the most trials, with
  the hierarchical-bootstrap median (mice, then units) of each class.

The examples are GRB064 units with all sessions after the 27 aug → 16 sep
tracking break (a link across it is likely a false match). Pairs across the
break stay in the pair statistics. The first flash after the exit can follow
a shorter or longer gap than the last flash in the port; the paired
comparison does not match gaps (see `exit_flash_pairs.py`, by gap).

Needs `results/unit_match/<mouse>_ids.csv`, `_pairs.csv` and the waveform
files. Writes `results/unit_tracking/report.json`.

    THESIS_CACHE_DIR=figures/cache uv run python -m \\
        thesis.ephys.analyses.unit_tracking
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thesis.ephys.analyses.glm.peaks import window_peak
from thesis.ephys.cache import cached
from thesis.ephys.task_events import fetch_task_stimulus_pulses
from thesis.ephys.trials import build_trial_table
from thesis.ephys.units import fetch_unit_table
from thesis.plotting import (
    FIGURE_STYLE,
    REFERENCE_COLOR,
    density_contours,
    hierarchical_interval,
    label_panels,
    median_marker,
    mouse_marker,
    save_figure,
)

MATCH = Path("results/unit_match")
OUT = Path("results/unit_tracking")
MICE = ("GRB064", "GRB061")
# Same markers as the three-mouse figures (GRB006, GRB064, GRB061).
MARKERS = {"GRB064": mouse_marker("GRB064", 1), "GRB061": mouse_marker("GRB061", 2)}
BIN_S = 0.01
FF_EDGES = np.round(np.arange(-0.10, 0.15 + 1e-9, BIN_S), 3)
FF_CENTERS = (FF_EDGES[:-1] + FF_EDGES[1:]) / 2
FF_BASE = (FF_CENTERS >= -0.04) & (FF_CENTERS < 0)
FF_SIM = (FF_CENTERS >= -0.04) & (FF_CENTERS < 0.15)
MOV_EDGES = np.round(np.arange(-0.04, 0.12 + 1e-9, BIN_S), 3)
MOV_CENTERS_MS = 1000 * (MOV_EDGES[:-1] + MOV_EDGES[1:]) / 2
MOV_BASE = MOV_CENTERS_MS < 0
ACG_EDGES = np.arange(1, 52) / 1000
MEAN_RATE_S = np.array([-0.5, 1.5])
NEIGHBOUR_UM = 50
MIN_TRIALS = 20
N_BOOT = 1000
DAY_BINS = np.array([1, 4, 8, 15, 29, 71])
CLASSES = ("excited", "suppressed", "none")
CLASS_COLORS = {"excited": "#E69F00", "suppressed": "#332288", "none": "0.72"}
CLASS_NAMES = {
    "excited": "flash-excited",
    "suppressed": "flash-suppressed",
    "none": "no flash response",
}
SAME = "black"
GAIN_UV = 3.02734  # ProbeConfiguration.probe_gain of both probes, µV per count
SHANK_PITCH_UM = 250


def objects(items) -> np.ndarray:
    """A 1-D object array of `items` (arrays stay whole in one table cell)."""
    out = np.empty(len(items), dtype=object)
    for k, item in enumerate(items):
        out[k] = item
    return out


def counts(spikes: np.ndarray, events, edges: np.ndarray) -> np.ndarray:
    """Spike counts, (event, bin)."""
    events = np.asarray(events, float)
    return np.diff(np.searchsorted(spikes, events[:, None] + edges[None, :]), axis=1)


def acg(spikes: np.ndarray) -> np.ndarray:
    """Spike counts at lags `ACG_EDGES` after each spike."""
    total = np.array([np.searchsorted(spikes, spikes + e).sum() for e in ACG_EDGES])
    return np.diff(total)


def paired_flash_times(trials: pd.DataFrame):
    """In-port flash, after-exit flash and first flash of each completed trial with both."""
    done = trials[(trials["early_withdrawal"] == 0) & trials["response"].isin((-1, 1))]
    port, move, first = [], [], []
    for flashes, entry, leave, response in zip(
        done["stim_pulse_times_s"],
        done["center_entry_s"].to_numpy(float),
        done["center_exit_s"].to_numpy(float),
        done["response_port_entry_s"].to_numpy(float),
        strict=True,
    ):
        f = np.asarray(flashes, float)
        if not len(f) or not np.isfinite([entry, leave, response]).all():
            continue
        inside = f[(f + MOV_EDGES[0] >= entry) & (f + MOV_EDGES[-1] <= leave)]
        outside = f[(f + MOV_EDGES[0] >= leave) & (f + MOV_EDGES[-1] <= response)]
        if len(inside) and len(outside):
            port.append(inside[-1])
            move.append(outside[0])
            first.append(f[0])
    return np.array(port), np.array(move), np.array(first)


@cached(version="v1")
def session_units(subject: str, session: str) -> pd.DataFrame:
    """Per unit: class, first-flash counts, ACG halves and paired flash counts."""
    from labdata_plugin.schema import StimulusResponse

    table = fetch_unit_table(subject, session, 1, include_metrics=False)
    spikes = [np.sort(np.asarray(s, float)) for s in table["spike_times_s"]]
    ids = table["unit_id"].astype(int).to_list()
    n = len(ids)
    rows = pd.DataFrame({"unit_id": ids, "n_spikes": [len(s) for s in spikes]})
    try:
        pulses = fetch_task_stimulus_pulses(subject, session)
        first = np.sort(
            pulses.loc[pulses["first_in_train"], "timestamp"].to_numpy(float)
        )
    except ValueError as err:
        print(f"{subject} {session}: no first flashes ({err})")
        first = np.empty(0)
    if len(first):
        ff = np.stack([counts(s, first, FF_EDGES) for s in spikes])
        _, masks = StimulusResponse.compute(
            ff / BIN_S, FF_EDGES, ids, (-0.04, 0.0), (0.03, 0.12), 0.05
        )
        rows["cls"] = np.where(
            masks["excited"],
            "excited",
            np.where(masks["suppressed"], "suppressed", "none"),
        )
        rows["ff"] = objects(list(ff.astype(np.int32)))
        rows["ff_mean_rate"] = [
            counts(s, first, MEAN_RATE_S).mean() / np.ptp(MEAN_RATE_S) for s in spikes
        ]
        middle = np.median(first)
    else:
        rows["cls"] = "unknown"
        rows["ff"] = objects([None] * n)
        rows["ff_mean_rate"] = np.nan
        middle = np.median(np.concatenate(spikes))
    rows["acg"] = objects(
        [
            np.stack([acg(s[s < middle]), acg(s[s >= middle])]).astype(np.int32)
            for s in spikes
        ]
    )
    try:
        trials = build_trial_table(subject, session, include_frames=False)
        port_t, move_t, first_t = paired_flash_times(trials)
    except ValueError as err:
        print(f"{subject} {session}: no trial table ({err})")
        port_t = move_t = first_t = np.empty(0)
    rows["n_paired"] = len(port_t)
    if len(port_t):
        rows["port"] = objects(
            [counts(s, port_t, MOV_EDGES).astype(np.int32) for s in spikes]
        )
        rows["move"] = objects(
            [counts(s, move_t, MOV_EDGES).astype(np.int32) for s in spikes]
        )
        rows["mov_mean"] = objects(
            [counts(s, first_t, MEAN_RATE_S)[:, 0] for s in spikes]
        )
    else:
        for column in ("port", "move", "mov_mean"):
            rows[column] = objects([None] * n)
    return rows


def load_mouse(mouse: str):
    """Units of all sessions with their track, day, shank and features."""
    ids = pd.read_csv(MATCH / f"{mouse}_ids.csv", dtype={"session": str})
    pairs = pd.read_csv(
        MATCH / f"{mouse}_pairs.csv", dtype={"session_a": str, "session_b": str}
    )
    sessions = sorted(ids["session"].unique())
    features = pd.concat(
        [session_units(mouse, s).assign(session=s) for s in sessions],
        ignore_index=True,
    )
    units = ids.merge(
        features, on=["session", "unit_id"], how="left", validate="one_to_one"
    )
    if units["cls"].isna().any():
        raise ValueError(f"{mouse}: UnitMatch units missing from the unit table")
    dates = pd.to_datetime(units["session"].str[:8])
    units["day"] = (dates - dates.min()).dt.days
    units["date"] = dates
    units["shank"] = np.floor((units["x_um"] + 100) / SHANK_PITCH_UM).astype(int)
    units["mouse"] = mouse
    units["duplicate"] = units.groupby("track")["session"].transform(
        lambda s: s.duplicated().any()
    )
    return units, pairs


def r(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def halves(ff: np.ndarray) -> np.ndarray:
    """First-flash rate of the first and the second half of the trials (2, bins)."""
    n = len(ff) // 2
    return np.stack([ff[:n][:, FF_SIM].mean(0), ff[n:][:, FF_SIM].mean(0)])


def cross(h_a: np.ndarray, h_b: np.ndarray) -> float:
    """Similarity of two units or sessions with half the data on each side."""
    values = [v for v in (r(h_a[0], h_b[1]), r(h_a[1], h_b[0])) if np.isfinite(v)]
    return float(np.mean(values)) if values else np.nan


def track_boot(values, tracks, rng, stat=np.nanmedian) -> list[float]:
    """95% interval of a statistic of pairs, resampling tracks."""
    values, tracks = np.asarray(values, float), np.asarray(tracks)
    _, inverse = np.unique(tracks, return_inverse=True)
    groups = [np.flatnonzero(inverse == k) for k in range(inverse.max() + 1)]
    draws = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(groups), len(groups))
        draws.append(stat(values[np.concatenate([groups[p] for p in pick])]))
    return [float(v) for v in np.nanpercentile(draws, [2.5, 97.5])]


def pair_tables(units: pd.DataFrame, pairs: pd.DataFrame) -> dict:
    """Same-unit pairs (sessions of one track) and neighbour pairs, with similarities."""
    good = units[~units["duplicate"]].copy()
    good["ffh"] = objects([halves(f) if f is not None else None for f in good["ff"]])
    index = {(s, u): i for i, s, u in zip(good.index, good["session"], good["unit_id"])}
    rows = []
    for track, g in good.groupby("track"):
        g = g.sort_values("day")
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                rows.append((track, g.index[i], g.index[j]))
    same = pd.DataFrame(rows, columns=["track", "a", "b"])
    near = pairs[
        (pairs["session_a"] != pairs["session_b"])
        & (pairs["distance_um"] < NEIGHBOUR_UM)
    ]
    a = [index.get(k) for k in zip(near["session_a"], near["unit_a"])]
    b = [index.get(k) for k in zip(near["session_b"], near["unit_b"])]
    neighbours = pd.DataFrame({"a": a, "b": b}).dropna().astype(int)
    neighbours = neighbours[
        good.loc[neighbours["a"], "track"].to_numpy()
        != good.loc[neighbours["b"], "track"].to_numpy()
    ].reset_index(drop=True)
    neighbours["track"] = good.loc[neighbours["a"], "track"].to_numpy()
    for table in (same, neighbours):
        ua, ub = good.loc[table["a"]], good.loc[table["b"]]
        table["days"] = np.abs(ub["day"].to_numpy() - ua["day"].to_numpy())
        table["cls_a"] = ua["cls"].to_numpy()
        table["cls_b"] = ub["cls"].to_numpy()
        table["ff_r"] = [
            cross(x, y) if x is not None and y is not None else np.nan
            for x, y in zip(ua["ffh"], ub["ffh"], strict=True)
        ]
        table["acg_r"] = [
            cross(x, y) for x, y in zip(ua["acg"], ub["acg"], strict=True)
        ]
    self_rows = good.assign(
        ff_r=[within_r(h) for h in good["ffh"]],
        acg_r=[r(x[0], x[1]) for x in good["acg"]],
    )
    return {"good": good, "same": same, "neighbours": neighbours, "self": self_rows}


def within_r(h) -> float:
    return r(h[0], h[1]) if h is not None else np.nan


def response(c: np.ndarray, kind: str) -> float:
    """Window peak (excited) or dip (suppressed) minus the baseline of the same flash."""
    rate = c.mean(0) / BIN_S
    base = rate[MOV_BASE].mean()
    if kind == "excited":
        return float(window_peak(rate, MOV_CENTERS_MS)) - base
    return -float(window_peak(-rate, MOV_CENTERS_MS)) - base


def modulation(port, move, mean_counts, kind: str) -> float:
    """After exit − in port, divided by the mean rate."""
    mean_rate = np.mean(mean_counts) / np.ptp(MEAN_RATE_S)
    if mean_rate <= 0:
        return np.nan
    return (response(move, kind) - response(port, kind)) / mean_rate


def trial_boot(port, move, mean_counts, kind: str, rng) -> list[float]:
    idx = rng.integers(0, len(port), (N_BOOT, len(port)))
    draws = [modulation(port[i], move[i], mean_counts[i], kind) for i in idx]
    return [float(v) for v in np.nanpercentile(draws, [2.5, 97.5])]


def track_class(ffs) -> str:
    """The StimulusResponse rule on the first flashes of all sessions of a track."""
    from labdata_plugin.schema import StimulusResponse

    peth = np.concatenate(ffs)[None] / BIN_S
    _, masks = StimulusResponse.compute(
        peth, FF_EDGES, [0], (-0.04, 0.0), (0.03, 0.12), 0.05
    )
    return (
        "excited"
        if masks["excited"][0]
        else "suppressed"
        if masks["suppressed"][0]
        else "none"
    )


def movement(good: pd.DataFrame, rng) -> dict:
    """Per track: its class, per-session modulation, and the pooled modulation."""
    tracks = []
    for track, g in good.groupby("track"):
        g = g.sort_values("day")
        ffs = [f for f in g["ff"] if f is not None]
        if not ffs:
            continue
        kind = track_class(ffs)
        if kind == "none":
            continue
        use = g[g["n_paired"] >= MIN_TRIALS]
        sessions = []
        for _, u in use.iterrows():
            sessions.append(
                dict(
                    day=int(u["day"]),
                    n=int(u["n_paired"]),
                    mod=modulation(u["port"], u["move"], u["mov_mean"], kind),
                    odd=modulation(
                        u["port"][::2], u["move"][::2], u["mov_mean"][::2], kind
                    ),
                    even=modulation(
                        u["port"][1::2], u["move"][1::2], u["mov_mean"][1::2], kind
                    ),
                    ci=trial_boot(u["port"], u["move"], u["mov_mean"], kind, rng),
                    row=u,
                )
            )
        item = dict(track=int(track), kind=kind, sessions=sessions)
        if len(sessions) >= 3:
            port = np.concatenate([s["row"]["port"] for s in sessions])
            move = np.concatenate([s["row"]["move"] for s in sessions])
            mean_c = np.concatenate([s["row"]["mov_mean"] for s in sessions])
            best = max(sessions, key=lambda s: s["n"])["row"]
            item.update(
                pooled=modulation(port, move, mean_c, kind),
                pooled_ci=trial_boot(port, move, mean_c, kind, rng),
                pooled_n=len(port),
                single=modulation(best["port"], best["move"], best["mov_mean"], kind),
                single_ci=trial_boot(
                    best["port"], best["move"], best["mov_mean"], kind, rng
                ),
                single_n=int(best["n_paired"]),
            )
        tracks.append(item)
    return tracks


def spearman_ci(x, y, tracks, rng) -> tuple[float, list[float]]:
    x, y, tracks = np.asarray(x), np.asarray(y), np.asarray(tracks)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, tracks = x[ok], y[ok], tracks[ok]
    rho = float(spearmanr(x, y).statistic)
    idx = np.arange(len(x))

    def stat(i):
        i = i.astype(int)
        return spearmanr(x[i], y[i]).statistic

    return rho, track_boot(idx, tracks, rng, stat=stat)


def day_bin_medians(table, value, rng):
    """Median and track-bootstrap interval of `value` in each bin of days apart."""
    out = []
    for lo, hi in zip(DAY_BINS[:-1], DAY_BINS[1:]):
        t = table[(table["days"] >= lo) & (table["days"] < hi)].dropna(subset=[value])
        if len(t) < 5:
            continue
        out.append(
            (
                np.median(t["days"]),
                float(np.median(t[value])),
                track_boot(t[value], t["track"], rng),
                len(t),
            )
        )
    return out


# ---------------------------------------------------------------- figures


def track_map(ax, units: pd.DataFrame) -> dict:
    """One row per track seen in two or more sessions; dots colored by class."""
    seen = units.groupby("track")["session"].transform("nunique")
    multi = units[seen >= 2]
    order = (
        multi.groupby("track")
        .agg(shank=("shank", "median"), depth=("depth_um", "mean"))
        .sort_values(["shank", "depth"], ascending=[True, False])
    )
    row = {t: k for k, t in enumerate(order.index)}
    shank_start = order.reset_index().groupby("shank").head(1).index
    gap = 6
    y = {
        t: k + gap * int(np.searchsorted(shank_start, k, side="right") - 1)
        for t, k in row.items()
    }
    for t, g in multi.groupby("track"):
        ax.plot(
            [g["day"].min(), g["day"].max()], [y[t]] * 2, color="0.85", lw=0.4, zorder=1
        )
    for kind in (*CLASSES, "unknown"):
        q = multi[multi["cls"] == kind]
        ax.scatter(
            q["day"],
            q["track"].map(y),
            s=3.5,
            color=CLASS_COLORS.get(kind, "white"),
            edgecolor="0.6" if kind == "unknown" else "none",
            lw=0.3,
            zorder=2,
        )
    ax.invert_yaxis()
    ticks, names = [], []
    for shank, g in order.groupby("shank"):
        ticks.append(np.mean([y[t] for t in g.index]))
        names.append(f"shank {int(shank) + 1}")
    ax.set_yticks(ticks, names)
    ax.tick_params(axis="y", length=0)
    days = np.sort(units["day"].unique())
    dates = units.groupby("day")["date"].first()
    ax.set_xticks(days, minor=True)
    # The first session of each month gets a date label; every session a tick.
    months = [
        d
        for k, d in enumerate(days)
        if k == 0 or dates[d].month != dates[days[k - 1]].month
    ]
    ax.set_xticks(months, [dates[d].strftime("%-d %b").lower() for d in months])
    ax.set_xlabel("recording day")
    ax.set_ylabel("units followed over sessions (by shank, superficial up)")
    return {
        "tracks_drawn": int(multi["track"].nunique()),
        "units_seen_once": int((seen == 1).sum()),
    }


def example_unit(fig, spec, units: pd.DataFrame, track: int, mouse: str, kind: str):
    """Waveforms of all sessions overlaid, and the first-flash response of each session."""
    g = units[units["track"] == track].sort_values("day")
    sub = spec.subgridspec(1, 3, width_ratios=(0.9, 2.2, 0.08), wspace=0.08)
    ax_w = fig.add_subplot(sub[0])
    ax_h = fig.add_subplot(sub[1])
    ax_c = fig.add_subplot(sub[2])
    waves = []
    positions = None
    for _, u in g.iterrows():
        d = np.load(MATCH / f"{mouse}_{u['session']}.npz")
        i = int(np.flatnonzero(d["unit_ids"] == u["unit_id"])[0])
        waves.append(d["waveforms"][i].mean(-1) * GAIN_UV)
        positions = d["channel_pos"]
    mean_wave = np.mean(waves, axis=0)
    peak = int(np.argmax(np.ptp(mean_wave, axis=0)))
    near = np.argsort(np.linalg.norm(positions - positions[peak], axis=1))[:8]
    amplitude = np.ptp(mean_wave[:, peak])
    t_ms = np.arange(mean_wave.shape[0]) / 30.0
    for w in waves:
        for c in near:
            dx = (positions[c, 0] - positions[peak, 0]) / 32 * 3.4
            dy = (positions[c, 1] - positions[peak, 1]) / 15 * 0.55 * amplitude
            ax_w.plot(t_ms + dx, w[:, c] + dy, color=SAME, lw=0.5, alpha=0.35)
    bar_uv = 50 if amplitude < 300 else 100
    x0 = t_ms[0] - 0.4
    y0 = mean_wave[:, near].min() - 0.15 * amplitude
    ax_w.plot([x0, x0 + 1], [y0, y0], color="black", lw=1)
    ax_w.plot([x0, x0], [y0, y0 + bar_uv], color="black", lw=1)
    ax_w.text(x0 + 0.5, y0 - 0.1 * amplitude, "1 ms", ha="center", va="top", fontsize=6)
    ax_w.text(
        x0 - 0.15,
        y0 + bar_uv / 2,
        f"{bar_uv} µV",
        ha="right",
        va="center",
        fontsize=6,
        rotation=90,
    )
    ax_w.set_axis_off()
    ax_w.set_title(f"{len(waves)} sessions overlaid", fontsize=7, color=SAME)
    rows, labels = [], []
    for _, u in g.iterrows():
        if u["ff"] is None:
            continue
        rate = u["ff"].mean(0) / BIN_S
        rows.append((rate - rate[FF_BASE].mean()) / max(u["ff_mean_rate"], 1e-9))
        labels.append(u["date"].strftime("%-d %b").lower())
    heat = np.array(rows)
    limit = np.nanpercentile(np.abs(heat), 99)
    image = ax_h.imshow(
        heat,
        aspect="auto",
        cmap="PuOr_r",
        vmin=-limit,
        vmax=limit,
        extent=(1000 * FF_EDGES[0], 1000 * FF_EDGES[-1], len(rows) - 0.5, -0.5),
        interpolation="nearest",
    )
    ax_h.axvline(0, color="black", ls="--", lw=0.8)
    ax_h.set_yticks(range(len(labels)), labels, fontsize=5.5)
    ax_h.set_xlabel("time from first flash (ms)")
    shank = int(g["shank"].median()) + 1
    depth = g["depth_um"].mean()
    ax_h.set_title(
        f"{CLASS_NAMES[kind]} unit, shank {shank}, {depth:.0f} µm from the tip",
        fontsize=7,
        color=CLASS_COLORS[kind],
    )
    bar = fig.colorbar(image, cax=ax_c)
    bar.set_label("(rate − baseline) ÷ mean rate", fontsize=6)
    bar.ax.tick_params(labelsize=6)
    return ax_w


def found_again(ax, data: dict, breaks: dict) -> dict:
    """Share of a session's units whose track has a unit in a later session."""
    report = {}
    for mouse, (units, _) in data.items():
        tracks = units.groupby("session")["track"].apply(set)
        sessions = list(tracks.index)
        days = units.groupby("session")["day"].first()
        lo, hi = breaks[mouse]
        rows = []
        for i, a in enumerate(sessions):
            for j in range(i + 1, len(sessions)):
                b = sessions[j]
                rows.append(
                    (
                        days[b] - days[a],
                        len(tracks[a] & tracks[b]) / len(tracks[a]),
                        a <= lo and b >= hi,
                        j == i + 1,
                    )
                )
        t = pd.DataFrame(rows, columns=["days", "share", "across", "next"])
        m = MARKERS[mouse]
        q = t[~t["across"]]
        ax.scatter(q["days"], q["share"], s=14, marker=m, color=SAME, alpha=0.45, lw=0)
        q = t[t["across"]]
        ax.scatter(
            q["days"],
            q["share"],
            s=14,
            marker=m,
            facecolor="none",
            edgecolor=SAME,
            lw=0.6,
            alpha=0.7,
        )
        own = units.groupby("session")["self_prob"].apply(lambda p: np.mean(p > 0.5))
        ax.scatter(
            np.zeros(len(own)) - 1.5, own, s=14, marker=m, color=SAME, alpha=0.45, lw=0
        )
        report[mouse] = {
            "same_session_median": float(np.median(own)),
            "next_session_median": float(
                t.loc[t["next"] & ~t["across"], "share"].median()
            ),
            "next_session_across_break": t.loc[t["next"] & t["across"], "share"]
            .round(3)
            .tolist(),
            "longest_gap_days": int(t.loc[~t["across"], "days"].max()),
            "longest_gap_share": float(
                t.loc[~t["across"]].sort_values("days")["share"].iloc[-1]
            ),
        }
    ax.set_xlabel("days between sessions")
    ax.set_ylabel("share of units found again")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlim(-4, 72)
    ax.text(-1.5, 1.02, "same\nsession", ha="center", va="bottom", fontsize=6)
    ax.text(24, 0.16, "open: pairs across a tracking break", fontsize=6, color=SAME)
    return report


def mouse_key(ax, x: float, y: float) -> None:
    """The marker of each mouse, once per figure."""
    for k, mouse in enumerate(MICE):
        yy = y - 0.07 * k
        ax.plot(
            x,
            yy,
            MARKERS[mouse],
            ms=4,
            color=SAME,
            transform=ax.transAxes,
            clip_on=False,
        )
        ax.text(x + 0.04, yy, mouse, transform=ax.transAxes, va="center", fontsize=6)


def similarity_violins(ax, groups: dict, value: str) -> dict:
    """Distribution of a similarity in each pair group, with the median of each mouse."""
    report = {}
    for k, (name, (frame, color)) in enumerate(groups.items()):
        v = frame[value].dropna().to_numpy()
        parts = ax.violinplot(v, [k], widths=0.75, showextrema=False)
        for body in parts["bodies"]:
            body.set_facecolor(color)
            body.set_edgecolor("none")
            body.set_alpha(0.35)
        for mouse in MICE:
            q = frame[frame["mouse"] == mouse][value].dropna()
            if len(q):
                median_marker(ax, k, float(np.median(q)), MARKERS[mouse], color, size=6)
        report[name] = {
            "pairs": int(len(v)),
            "median": float(np.median(v)),
            **{
                m: float(np.median(frame[frame["mouse"] == m][value].dropna()))
                for m in MICE
            },
        }
    ax.set_xticks(range(len(groups)), list(groups))
    ax.set_xlim(-0.6, len(groups) - 0.4)
    return report


def plot_tracking(data, tables, examples, breaks) -> dict:
    report = {}
    with plt.rc_context(FIGURE_STYLE):
        fig = plt.figure(figsize=(11, 9))
        grid = fig.add_gridspec(
            3, 3, width_ratios=(1.15, 1, 1), hspace=0.55, wspace=0.45
        )
        ax_map = fig.add_subplot(grid[:, 0])
        report["map"] = track_map(ax_map, data["GRB064"][0])
        mouse = "GRB064"
        axes_ex = [
            example_unit(fig, grid[k, 1:], data[mouse][0], examples[kind], mouse, kind)
            for k, kind in enumerate(("excited", "suppressed"))
        ]
        ax_found = fig.add_subplot(grid[2, 1])
        report["found_again"] = found_again(ax_found, data, breaks)
        ax_acg = fig.add_subplot(grid[2, 2])
        same = pd.concat([t["same"].assign(mouse=m) for m, t in tables.items()])
        near = pd.concat([t["neighbours"].assign(mouse=m) for m, t in tables.items()])
        own = pd.concat([t["self"].assign(mouse=m) for m, t in tables.items()])
        report["acg"] = similarity_violins(
            ax_acg,
            {
                "same unit\nsame\nsession": (own, SAME),
                "same unit\nother\nsession": (same, SAME),
                "neighbour\nother\nsession": (near, REFERENCE_COLOR),
            },
            "acg_r",
        )
        ax_acg.set_ylabel("ACG similarity (r, half the spikes each)")
        ax_acg.set_ylim(-0.2, 1.02)
        mouse_key(ax_found, 0.62, 0.34)
        for k, kind in enumerate(CLASSES):
            ax_map.text(
                (0.0, 0.3, 0.68)[k],
                1.005,
                CLASS_NAMES[kind],
                color=CLASS_COLORS[kind] if kind != "none" else "0.45",
                transform=ax_map.transAxes,
                fontsize=6.5,
                va="bottom",
            )
        ax_map.text(
            0.02,
            -0.075,
            f"{report['map']['tracks_drawn']} units in ≥ 2 sessions; "
            f"{report['map']['units_seen_once']} seen once not drawn; "
            "open: session without flash times",
            transform=ax_map.transAxes,
            fontsize=6,
        )
        label_panels([ax_map, *axes_ex, ax_found, ax_acg], x=-0.12, y=1.03)
        fig.savefig(
            Path("figures/unit_tracking/tracking.png"), dpi=150, bbox_inches="tight"
        )
        print(save_figure(fig, OUT / "tracking"))
    return report


def scatter_two_sessions(ax, moves: list, kind: str, rng, lim: float) -> dict:
    """Modulation of one unit in two consecutive sessions, as density contours."""
    xs, ys, xh, yh, tr, odd, even, tro, widths = ([] for _ in range(9))
    for item in moves:
        if item["kind"] != kind:
            continue
        key = f"{item['mouse']}{item['track']}"
        s = item["sessions"]
        for x in s:
            odd.append(x["odd"])
            even.append(x["even"])
            tro.append(key)
            widths.append(np.diff(x["ci"])[0] / 2)
        for p, q in zip(s[:-1], s[1:]):
            xs.append(p["mod"])
            ys.append(q["mod"])
            xh.append(p["odd"])
            yh.append(q["even"])
            tr.append(key)
    full = spearman_ci(xs, ys, tr, rng)
    half_other = spearman_ci(xh, yh, tr, rng)
    half_same = spearman_ci(odd, even, tro, rng)
    color = CLASS_COLORS[kind]
    points = np.array([xs, ys], float)
    points = points[:, np.isfinite(points).all(0)]
    grid = np.linspace(-lim, lim, 150)
    ax.axhline(0, color="0.85", lw=0.6, zorder=0)
    ax.axvline(0, color="0.85", lw=0.6, zorder=0)
    ax.plot([-lim, lim], [-lim, lim], color=REFERENCE_COLOR, ls="--", lw=0.8)
    density_contours(ax, points, color, grid, grid)
    # The median 95% interval of one session, on both axes.
    half = float(np.nanmedian(widths))
    cx, cy = lim - half - 0.08 * lim, -lim + half + 0.08 * lim
    ax.plot([cx - half, cx + half], [cy, cy], color="black", lw=1)
    ax.plot([cx, cx], [cy - half, cy + half], color="black", lw=1)
    hidden = int((np.abs(points) > lim).any(0).sum())
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    units = len(set(tr))
    ax.set_title(
        f"{CLASS_NAMES[kind]}, {units} units\n"
        f"next session ρ = {full[0]:.2f} [{full[1][0]:.2f}, {full[1][1]:.2f}]\n"
        f"half the trials:\nother session {half_other[0]:.2f}, "
        f"same session {half_same[0]:.2f}",
        fontsize=6.5,
        color=color,
    )
    ax.set_xlabel(
        f"one session\n+: median 95% interval of one session\n"
        f"({hidden} of {points.shape[1]} pairs past the axes)"
    )
    ax.set_ylabel("next session")
    return {
        "units": units,
        "pairs": int(len(xs)),
        "pairs_past_axes": hidden,
        "next_session_rho": full[0],
        "next_session_ci": full[1],
        "half_trials_other_session_rho": half_other[0],
        "half_trials_other_session_ci": half_other[1],
        "half_trials_same_session_rho": half_same[0],
        "half_trials_same_session_ci": half_same[1],
        "median_session_interval_half_width": half,
    }


def pooled_units(ax, moves: list) -> dict:
    """Modulation of single units with all their sessions pooled, against one session."""
    stats = {}
    start = 0
    tops = []
    for kind in ("excited", "suppressed"):
        items = sorted(
            [m for m in moves if m["kind"] == kind and "pooled" in m],
            key=lambda m: m["pooled"],
        )
        color = CLASS_COLORS[kind]
        for k, m in enumerate(items):
            x = start + k
            ax.plot([x + 0.3] * 2, m["single_ci"], color="0.82", lw=1.1, zorder=1)
            ax.plot([x] * 2, m["pooled_ci"], color=color, lw=1.1, zorder=2)
            ax.plot(x, m["pooled"], MARKERS[m["mouse"]], ms=3, color=color, zorder=3)
        groups = [
            np.array([m["pooled"] for m in items if m["mouse"] == mouse])
            for mouse in MICE
        ]
        mice = [mouse for mouse, g in zip(MICE, groups, strict=True) if len(g)]
        groups = [g for g in groups if len(g)]
        summary = hierarchical_interval(groups)
        end = start + len(items) - 1
        ax.plot(
            [start, end], [summary["median"]] * 2, color=color, lw=1, ls=":", zorder=4
        )
        for k, (mouse, g) in enumerate(zip(mice, groups, strict=True)):
            median_marker(
                ax,
                end + 3 + 2.5 * k,
                float(np.median(g)),
                MARKERS[mouse],
                color,
                size=6,
            )

        def above(ci):
            return ci[0] > 0

        def below(ci):
            return ci[1] < 0

        counts_ = {
            "pooled_above_0": sum(above(m["pooled_ci"]) for m in items),
            "pooled_below_0": sum(below(m["pooled_ci"]) for m in items),
            "one_session_above_0": sum(above(m["single_ci"]) for m in items),
            "one_session_below_0": sum(below(m["single_ci"]) for m in items),
        }
        lo, hi = summary["interval_95"]
        per_mouse = ", ".join(
            f"{mouse} {len(g)}" for mouse, g in zip(mice, groups, strict=True)
        )
        ax.text(
            start if kind == "excited" else end,
            1.01,
            f"{CLASS_NAMES[kind]} ({per_mouse})\n"
            f"median {summary['median']:.2f} [{lo:.2f}, {hi:.2f}]\n"
            f"interval > 0: {counts_['pooled_above_0']} units, "
            f"< 0: {counts_['pooled_below_0']}\n"
            f"one session: > 0: {counts_['one_session_above_0']}, "
            f"< 0: {counts_['one_session_below_0']}",
            transform=ax.get_xaxis_transform(),
            ha="left" if kind == "excited" else "right",
            va="bottom",
            fontsize=6,
            color=color,
        )
        stats[kind] = {
            "units": len(items),
            "units_per_mouse": {
                mouse: len(g) for mouse, g in zip(mice, groups, strict=True)
            },
            "median": summary["median"],
            "median_ci_hierarchical": summary["interval_95"],
            "p_bootstrap": summary["p_bootstrap"],
            "median_of_each_mouse": dict(
                zip(mice, summary["median_of_each_mouse"], strict=True)
            ),
            **counts_,
            "median_interval_width_ratio_pooled_to_one_session": float(
                np.median(
                    [
                        np.diff(m["pooled_ci"])[0] / np.diff(m["single_ci"])[0]
                        for m in items
                    ]
                )
            ),
            "median_sessions": float(np.median([len(m["sessions"]) for m in items])),
            "median_trials_pooled": float(np.median([m["pooled_n"] for m in items])),
            "median_trials_one_session": float(
                np.median([m["single_n"] for m in items])
            ),
        }
        tops.extend(c for m in items for c in m["pooled_ci"])
        start = end + 14
    last = end
    span = max(tops) - min(tops)
    low, high = min(tops) - 0.08 * span, max(tops) + 0.08 * span
    ax.set_ylim(low, high)
    ax.set_xlim(-2, last + 9)
    cut = sum(
        m["single_ci"][0] < low or m["single_ci"][1] > high
        for m in moves
        if "pooled" in m
    )
    ax.axhline(0, color=REFERENCE_COLOR, ls="--", lw=0.8)
    ax.set_xticks([])
    ax.set_xlabel(
        "units followed in ≥ 3 sessions, by pooled modulation\n"
        "color: all sessions pooled; gray: the one session with the most trials\n"
        f"({cut} gray intervals pass the axes); dotted: pooled median; "
        "open: median of each mouse"
    )
    ax.set_ylabel(
        "movement modulation: flash response\nafter exit − in port (÷ mean rate)"
    )
    stats["gray_intervals_past_axes"] = int(cut)
    return stats


def plot_responses(tables, moves, rng) -> dict:
    report = {}
    same = pd.concat([t["same"].assign(mouse=m) for m, t in tables.items()])
    near = pd.concat([t["neighbours"].assign(mouse=m) for m, t in tables.items()])
    own = pd.concat([t["self"].assign(mouse=m) for m, t in tables.items()])
    responsive = ("excited", "suppressed")
    # The same flash class in both sessions, so that a neighbour is not less
    # similar only because it has another class.
    same_r = same[same["cls_a"].isin(responsive) & (same["cls_b"] == same["cls_a"])]
    near_r = near[near["cls_a"].isin(responsive) & (near["cls_b"] == near["cls_a"])]
    own_r = own[own["cls"].isin(responsive)]
    with plt.rc_context(FIGURE_STYLE):
        fig = plt.figure(figsize=(11, 8.4))
        grid = fig.add_gridspec(2, 4, height_ratios=(1, 1.1), hspace=0.5, wspace=0.55)
        ax = fig.add_subplot(grid[0, :2])
        axes = [ax]
        for mouse in MICE:
            q = same_r[same_r["mouse"] == mouse]
            ax.scatter(
                q["days"],
                q["ff_r"],
                s=4,
                marker=MARKERS[mouse],
                color=SAME,
                alpha=0.12,
                lw=0,
            )
        stats = {}
        for name, table, color in (
            ("same unit", same_r, SAME),
            ("neighbour", near_r, REFERENCE_COLOR),
        ):
            bins = day_bin_medians(table, "ff_r", rng)
            ax.plot(
                [b[0] for b in bins],
                [b[1] for b in bins],
                color=color,
                lw=1.6,
                marker="o",
                ms=4,
            )
            for b in bins:
                ax.plot([b[0]] * 2, b[2], color=color, lw=1.2)
            stats[name] = [
                {"days_median": float(b[0]), "median": b[1], "ci": b[2], "pairs": b[3]}
                for b in bins
            ]
        # The median of each mouse in each bin of days (same unit).
        for k, mouse in enumerate(MICE):
            q = same_r[same_r["mouse"] == mouse]
            per_bin = []
            for lo, hi in zip(DAY_BINS[:-1], DAY_BINS[1:], strict=False):
                b = q[(q["days"] >= lo) & (q["days"] < hi)].dropna(subset=["ff_r"])
                if len(b) < 5:
                    continue
                x = float(np.median(b["days"])) + (1.4, -1.4)[k]
                y = float(np.median(b["ff_r"]))
                median_marker(ax, x, y, MARKERS[mouse], SAME, size=5)
                per_bin.append(
                    {"days_median": x - (1.4, -1.4)[k], "median": y, "pairs": len(b)}
                )
            stats.setdefault("same unit, each mouse", {})[mouse] = per_bin
        own_ci = track_boot(own_r["ff_r"], own_r["track"], rng)
        own_med = float(np.nanmedian(own_r["ff_r"]))
        ax.plot([-3, -3], own_ci, color=SAME, lw=1.2)
        ax.plot(-3, own_med, "o", color="white", mec=SAME, mew=1.4, ms=6)
        ax.text(
            -3,
            own_ci[0] - 0.03,
            "same\nsession",
            fontsize=6,
            color=SAME,
            ha="center",
            va="top",
        )
        stats["same session"] = {
            "median": own_med,
            "ci": own_ci,
            "units": int(own_r["ff_r"].notna().sum()),
        }
        last = stats["same unit"][-1]
        ax.text(
            last["days_median"],
            last["median"] + 0.08,
            "same unit",
            fontsize=6.5,
            color=SAME,
            ha="center",
        )
        lastn = stats["neighbour"][-1]
        ax.text(
            lastn["days_median"],
            lastn["median"] - 0.12,
            "neighbour, same class (< 50 µm)",
            fontsize=6.5,
            color="0.45",
            ha="center",
        )
        ax.axhline(0, color=REFERENCE_COLOR, lw=0.6, ls="--")
        ax.set_xlim(-6, 50)
        ax.set_xticks([0, 10, 20, 30, 40, 50])
        ax.set_xlabel(
            "days between sessions\n(pairs with the same flash response class in both sessions)"
        )
        ax.set_ylabel("first-flash response similarity\n(r, half the trials each)")
        ax.set_ylim(-0.45, 1.0)
        stats["same session"]["each mouse"] = {
            mouse: float(np.nanmedian(own_r.loc[own_r["mouse"] == mouse, "ff_r"]))
            for mouse in MICE
        }
        report["first_flash_similarity"] = stats

        ax = fig.add_subplot(grid[0, 2:])
        axes.append(ax)
        both = same[same["cls_a"].isin(CLASSES) & same["cls_b"].isin(CLASSES)]
        nboth = near[near["cls_a"].isin(CLASSES) & near["cls_b"].isin(CLASSES)]
        stats = {}
        for k, kind in enumerate(CLASSES):
            color = CLASS_COLORS[kind] if kind != "none" else "0.45"
            for offset, table, filled, name in (
                (-0.12, both, True, "same unit"),
                (0.12, nboth, False, "neighbour"),
            ):
                q = table[table["cls_a"] == kind]
                hit = (q["cls_b"] == kind).astype(float)
                ci = track_boot(hit, q["track"], rng, stat=np.mean)
                value = float(hit.mean())
                ax.plot(
                    [k + offset] * 2, [100 * ci[0], 100 * ci[1]], color=color, lw=1.2
                )
                ax.plot(
                    k + offset,
                    100 * value,
                    "o",
                    ms=7,
                    color=color if filled else "white",
                    mec=color,
                    mew=1.4,
                )
                stats.setdefault(kind, {})[name] = {
                    "percent": 100 * value,
                    "ci": [100 * c for c in ci],
                    "pairs": int(len(q)),
                }
            stats[kind]["share of all"] = 100 * float(np.mean(both["cls_b"] == kind))
            for j, mouse in enumerate(MICE):
                q = both[(both["cls_a"] == kind) & (both["mouse"] == mouse)]
                if len(q) < 5:
                    continue
                value = 100 * float(np.mean(q["cls_b"] == kind))
                median_marker(
                    ax, k - 0.3 - 0.08 * j, value, MARKERS[mouse], color, size=5
                )
                stats[kind].setdefault("same unit, each mouse", {})[mouse] = {
                    "percent": value,
                    "pairs": int(len(q)),
                }
        ax.set_xticks(range(3), [CLASS_NAMES[k] for k in CLASSES])
        for tick, kind in zip(ax.get_xticklabels(), CLASSES, strict=True):
            tick.set_color(CLASS_COLORS[kind] if kind != "none" else "0.45")
        ax.set_xlabel("class in the first session of the pair")
        ax.set_ylabel("same class in the other session (%)")
        ax.set_ylim(0, 100)
        ax.set_xlim(-0.5, 2.5)
        ax.text(
            -0.45,
            4,
            "filled: same unit; open circle: neighbour\nopen triangles: same unit, each mouse",
            fontsize=6.5,
            va="bottom",
        )
        report["class_stability"] = stats

        values = [x["mod"] for item in moves for x in item["sessions"]]
        lim = float(np.nanpercentile(np.abs(values), 98))
        report["movement_two_sessions"] = {}
        for k, kind in enumerate(responsive):
            ax = fig.add_subplot(grid[1, k])
            axes.append(ax)
            report["movement_two_sessions"][kind] = scatter_two_sessions(
                ax, moves, kind, rng, lim
            )
        ax = fig.add_subplot(grid[1, 2:])
        axes.append(ax)
        report["movement_pooled"] = pooled_units(ax, moves)
        mouse_key(ax, 0.84, 0.16)
        label_panels([*axes[:2], None, None, None], x=-0.12, y=1.02)
        label_panels([None, None, *axes[2:4], None], x=-0.25, y=1.42)
        label_panels([None, None, None, None, axes[4]], x=-0.12, y=1.27)
        fig.savefig(
            Path("figures/unit_tracking/responses.png"), dpi=150, bbox_inches="tight"
        )
        print(save_figure(fig, OUT / "responses"))
    return report


def pick_examples(units: pd.DataFrame, moves: list, first_session: str) -> dict:
    """The excited and the suppressed track in the most sessions, then the largest response.

    Only tracks with all sessions from `first_session` on: a link across the
    tracking break is likely a false match.
    """
    good = units[~units["duplicate"]]
    after = good.groupby("track")["session"].min() >= first_session
    n = good.groupby("track")["session"].nunique()
    picks = {}
    for kind in ("excited", "suppressed"):
        candidates = [
            m["track"] for m in moves if m["kind"] == kind and m["mouse"] == "GRB064"
        ]
        scored = []
        for t in candidates:
            if not after[t]:
                continue
            g = good[(good["track"] == t) & good["ff"].notna()]
            if not len(g):
                continue
            agree = np.mean(g["cls"] == kind)
            rate = np.concatenate(list(g["ff"])).mean(0) / BIN_S
            size = abs(
                rate[(FF_CENTERS >= 0.03) & (FF_CENTERS < 0.12)].mean()
                - rate[FF_BASE].mean()
            )
            scored.append((n[t], agree, size, t))
        scored.sort(reverse=True)
        picks[kind] = scored[0][3]
    return picks


def main() -> None:
    rng = np.random.default_rng(0)
    Path("figures/unit_tracking").mkdir(parents=True, exist_ok=True)
    data = {m: load_mouse(m) for m in MICE}
    tables = {m: pair_tables(*data[m]) for m in MICE}
    moves = []
    for m in MICE:
        for item in movement(tables[m]["good"], rng):
            item["mouse"] = m
            moves.append(item)
    # The last session before and the first after the step that few units cross.
    breaks = {
        "GRB064": ("20260827_143334", "20260916_132939"),
        "GRB061": ("20261113_133142", "20261116_133457"),
    }
    examples = pick_examples(data["GRB064"][0], moves, breaks["GRB064"][1])
    report = {
        "units": {m: int(len(d[0])) for m, d in data.items()},
        "tracks": {m: int(d[0]["track"].nunique()) for m, d in data.items()},
        "tracks_left_out_split_or_merge": {
            m: int(d[0].loc[d[0]["duplicate"], "track"].nunique())
            for m, d in data.items()
        },
        "examples": {k: int(v) for k, v in examples.items()},
    }
    report["tracking"] = plot_tracking(data, tables, examples, breaks)
    report["responses"] = plot_responses(tables, moves, rng)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main()
