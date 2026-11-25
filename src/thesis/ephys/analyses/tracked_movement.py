"""How movement reshapes the flash response of V1 units followed over sessions.

Question: UnitMatch tracks (`unit_tracking.py`) give each unit many sessions of
trials. (a) Is the change of a unit's flash response after the center exit
large against the change of the same response over weeks (drift)? (b) Does
movement change the timing of the response, and does that follow the turn
side, that is, where the LED panel is in the visual field (b–d)? (e) Does the change depend
on the speed of the movement, inside the same units? (f) Do narrow- and
broad-spiking units change differently? In the report only, exploratory: do
sessions with a different movement time have a different flash response
(Sadeh & Clopath 2022, eLife 11:e77907)?

Units and tracks: as `unit_tracking.py` (quality criterion 1, UnitMatch
"intermediate" tracks, no track with two units in one session). A track's
class is the StimulusResponse rule on all its first flashes; only
flash-excited and flash-suppressed tracks. Sessions: GRB064 from 16 sep on
(earlier sessions are before a tracking break; 21 sep has no trial table),
all GRB061 sessions. No pair of sessions across a tracking break (GRB061 13
nov → 16 nov). A unit-session needs `MIN_TRIALS` paired trials.

Flashes (gap-matched, paired in each completed trial): the last flash whose
window (-40 to 120 ms) is in the port, and the first flash whose window
starts at or after the center exit and ends before the response entry; both
at least 80 ms after the flash before (the first flash of a train has no
flash before and is not used). Only trials with both. Baseline: -40 to 0 ms
of the same flash, so each flash has a baseline in its own period.

Amplitude, linear in spike counts: one window per unit for both periods,
±20 ms around the largest (flash-excited) or smallest (flash-suppressed)
10-ms bin from 30 to 120 ms of the mean of its in-port and after-exit traces
(all sessions pooled). Response = mean rate in that window − baseline, divided
by the unit's mean rate over all its sessions (0.5 s before to 1.5 s after
each first flash). Movement change = after exit − in port; for
flash-suppressed units a negative change is a deeper dip. The report also
gives the change with the full 30–120 ms window and with a window found in
each period apart, because the response timing changes (b).

a. GRB064 only (GRB061 tracks are short and less reliable; its values are in
   the report). Each term is the difference of two independent half-trial
   estimates (odd and even trials): noise = in port, odd − even; movement =
   after exit in one half − in port in the other half, same session; drift =
   in port in one session (one half) − in port in a later session (other
   half). True root-mean-square change over units = sqrt(mean(term²) −
   mean(noise²)), with the noise of the same sessions and periods. Intervals
   resample tracks.
b. Mean flash response (baseline-corrected, ÷ mean rate, all sessions pooled
   per unit; standard error over units) in the port, after a left turn and
   after a right turn (units with 20 or more paired trials of each side).
   In both mice the probe is in left V1 (aimed at the upper field and the
   binocular zone by coordinates only; no retinotopy), which sees the right
   visual field; the LED panel is above the center port, so after a left turn
   it lies to the right of the head (contralateral), after a right turn to
   the left (ipsilateral).
c. Timing: centre of mass (ms) of the baseline-corrected response from 30 to
   120 ms (its positive part for flash-excited units, its negative part for
   flash-suppressed units), after exit − in port in the same trials, for each
   turn side. One point per unit (60 or more paired trials). A slow change of
   the firing rate after the exit moves the two classes in opposite
   directions, so it cannot move both earlier. Caution: the in-port trace of
   flash-excited units has a dip at 20–30 ms (likely the end of the response
   to the flash before, which peaks late in the port); it moves their in-port
   centre of mass later, so read c with b and d.
d. The fast part of the response (20–40 ms), left − right turn. In the port
   (split by the upcoming choice) the head is in the same place for both
   sides: the control. After the exit the head has turned. Fast trains are
   rewarded on the right, so the report also tests the turn side within each
   stimulus rate and in error trials (all units of a session, population
   rate; side shuffled within rate). The turn side cannot separate where the
   panel is in the visual field from the direction of the turn itself.
   The report also gives the timing change with both sides pooled and the
   change in an early (30–60 ms) and a late (60–120 ms) part.
e. Movement time (center exit to response entry), split in thirds in each
   session (the edges of each session), then pooled per third (units in 3+
   sessions). The median latency of the after-exit flash and the share of
   completed trials with a pair are given for each third.
f. Spike duration: the labdata waveform metric, median over the unit's
   sessions. Narrow and broad at the minimum of its density from 0.2 to 0.6 ms
   (its histogram is in the report as the threshold only).

Pooled medians: hierarchical bootstrap (mice, then units); the median of
each mouse is an open marker. No GLM fits, no video. The exit comes with
sounds (go cue before it); they do not change the flash measure, whose
baseline is the 40 ms before each flash.

Writes `figures/unit_tracking/movement.svg` and
`results/unit_tracking/movement.json`.

    THESIS_CACHE_DIR=figures/cache uv run python -m \\
        thesis.ephys.analyses.tracked_movement
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, mannwhitneyu

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thesis.ephys.analyses.unit_tracking import (
    BIN_S,
    CLASS_COLORS,
    CLASS_NAMES,
    DAY_BINS,
    MARKERS,
    MEAN_RATE_S,
    MICE,
    MIN_TRIALS,
    MOV_BASE,
    MOV_CENTERS_MS,
    MOV_EDGES,
    counts,
    load_mouse,
    objects,
    pair_tables,
    track_boot,
    track_class,
)
from thesis.ephys.cache import cached
from thesis.ephys.trials import build_trial_table
from thesis.ephys.units import fetch_unit_table
from thesis.plotting import (
    FIGURE_STYLE,
    REFERENCE_COLOR,
    hierarchical_interval,
    label_panels,
    median_marker,
    save_figure,
)

OUT = Path("results/unit_tracking")
FIRST_SESSION = {"GRB064": "20260916_132939", "GRB061": "20261109_134551"}
# Last session before and first session after a tracking break.
BREAKS = {"GRB061": ("20261113_133142", "20261116_133457")}
GAP_MIN_MS = 75  # gaps are 40-ms steps with a few ms of jitter: >= 80 ms
THIRDS = ("fast", "middle", "slow")
KINDS = ("excited", "suppressed")
N_PERMUTATIONS = 9999


@cached(version="v2")
def gap_matched(subject: str, session: str) -> pd.DataFrame:
    """Per unit: counts around gap-matched paired flashes, with trial facts."""
    table = fetch_unit_table(subject, session, 1, include_metrics=True)
    spikes = [np.sort(np.asarray(s, float)) for s in table["spike_times_s"]]
    n = len(spikes)
    try:
        trials = build_trial_table(subject, session, include_frames=False)
    except ValueError as err:
        print(f"{subject} {session}: no trial table ({err})")
        trials = pd.DataFrame(
            columns=["early_withdrawal", "response", "rewarded", "stim_pulse_times_s"]
            + ["center_entry_s", "center_exit_s", "response_port_entry_s"]
        )
    done = trials[(trials["early_withdrawal"] == 0) & trials["response"].isin((-1, 1))]
    port_t, move_t, first_t, move_time, latency, side = [], [], [], [], [], []
    all_time, paired = [], []
    for flashes, entry, leave, response, choice in zip(
        done["stim_pulse_times_s"],
        done["center_entry_s"].to_numpy(float),
        done["center_exit_s"].to_numpy(float),
        done["response_port_entry_s"].to_numpy(float),
        done["response"].to_numpy(float),
        strict=True,
    ):
        f = np.asarray(flashes, float)
        if not len(f) or not np.isfinite([entry, leave, response]).all():
            continue
        gap = np.diff(f, prepend=np.nan) * 1000
        ok = gap >= GAP_MIN_MS
        inside = ok & (f + MOV_EDGES[0] >= entry) & (f + MOV_EDGES[-1] <= leave)
        outside = ok & (f + MOV_EDGES[0] >= leave) & (f + MOV_EDGES[-1] <= response)
        all_time.append(response - leave)
        paired.append(bool(inside.any() and outside.any()))
        if paired[-1]:
            port_t.append(f[inside][-1])
            move_t.append(f[outside][0])
            first_t.append(f[0])
            move_time.append(response - leave)
            latency.append(f[outside][0] - leave)
            side.append(int(choice))
    rows = pd.DataFrame(
        {
            "unit_id": table["unit_id"].astype(int).to_numpy(),
            "spike_duration_ms": table["spike_duration_ms"].to_numpy(float),
        }
    )
    session_facts = {
        "n_trials": len(trials),
        "n_early": int((trials["early_withdrawal"] == 1).sum()),
        "n_completed": len(done),
        "pct_correct": float(100 * done["rewarded"].mean()) if len(done) else np.nan,
    }
    for key, value in session_facts.items():
        rows[key] = value
    for key, value in (
        ("gm_move_time", np.array(move_time)),
        ("gm_latency", np.array(latency)),
        ("gm_side", np.array(side, dtype=int)),
        ("all_move_time", np.array(all_time)),
        ("all_paired", np.array(paired)),
    ):
        rows[key] = objects([value] * n)
    rows["n_gm"] = len(port_t)
    if port_t:
        rows["gm_port"] = objects(
            [counts(s, port_t, MOV_EDGES).astype(np.int32) for s in spikes]
        )
        rows["gm_move"] = objects(
            [counts(s, move_t, MOV_EDGES).astype(np.int32) for s in spikes]
        )
        rows["gm_mean"] = objects(
            [counts(s, first_t, MEAN_RATE_S)[:, 0] for s in spikes]
        )
    else:
        for column in ("gm_port", "gm_move", "gm_mean"):
            rows[column] = objects([None] * n)
    return rows


def window_for(trace: np.ndarray, kind: str) -> np.ndarray:
    """Bins within 20 ms of the largest (excited) or smallest (suppressed) 30–120 ms bin."""
    rate = trace.mean(0) / BIN_S
    search = np.flatnonzero((MOV_CENTERS_MS > 30 - 1e-6) & (MOV_CENTERS_MS < 120))
    pick = np.argmax(rate[search]) if kind == "excited" else np.argmin(rate[search])
    at = MOV_CENTERS_MS[search[pick]]
    return (np.abs(MOV_CENTERS_MS - at) <= 20 + 1e-6) & (MOV_CENTERS_MS < 120)


def response(c: np.ndarray, window: np.ndarray, norm: float) -> float:
    rate = c.mean(0) / BIN_S
    return float((rate[window].mean() - rate[MOV_BASE].mean()) / norm)


def build_tracks(mouse: str) -> tuple[list, pd.DataFrame, dict]:
    """Tracks with a flash class and their gap-matched trials in each session."""
    units, pairs = load_mouse(mouse)
    sessions = sorted(s for s in units["session"].unique() if s >= FIRST_SESSION[mouse])
    gm = pd.concat(
        [gap_matched(mouse, s).assign(session=s) for s in sessions], ignore_index=True
    )
    units = units.merge(
        gm, on=["session", "unit_id"], how="left", validate="one_to_one"
    )
    use = units[~units["duplicate"] & (units["session"] >= FIRST_SESSION[mouse])]
    tracks = []
    for track, g in use.groupby("track"):
        g = g.sort_values("day")
        ffs = [f for f in g["ff"] if f is not None]
        if not ffs:
            continue
        kind = track_class(ffs)
        if kind == "none":
            continue
        g = g[g["n_gm"].fillna(0) >= MIN_TRIALS]
        if not len(g):
            continue
        both = np.concatenate([*g["gm_port"], *g["gm_move"]])
        window = window_for(both, kind)
        norm = np.concatenate(list(g["gm_mean"])).mean() / np.ptp(MEAN_RATE_S)
        if norm <= 0:
            continue
        tracks.append(
            dict(
                mouse=mouse,
                track=f"{mouse}{track}",
                kind=kind,
                window=window,
                norm=norm,
                spike_duration_ms=float(np.nanmedian(g["spike_duration_ms"])),
                sessions=[
                    dict(
                        session=u["session"],
                        day=int(u["day"]),
                        port=u["gm_port"],
                        move=u["gm_move"],
                        move_time=u["gm_move_time"],
                        latency=u["gm_latency"],
                        side=u["gm_side"],
                    )
                    for _, u in g.iterrows()
                ],
            )
        )
    return tracks, units, pairs


def across_break(mouse: str, a: str, b: str) -> bool:
    if mouse not in BREAKS:
        return False
    last, first = BREAKS[mouse]
    return a <= last and b >= first


def movement_and_drift(tracks: list) -> pd.DataFrame:
    """Noise-corrected squared changes: movement (same session) and drift (two sessions)."""
    rows = []
    for t in tracks:
        w, norm = t["window"], t["norm"]
        halves = []
        for s in t["sessions"]:
            odd, even = slice(0, None, 2), slice(1, None, 2)
            r = {
                (period, half): response(s[period][sl], w, norm)
                for period in ("port", "move")
                for half, sl in (("odd", odd), ("even", even))
            }
            noise_port = (r["port", "odd"] - r["port", "even"]) ** 2
            noise_move = (r["move", "odd"] - r["move", "even"]) ** 2
            square = (
                (r["move", "odd"] - r["port", "even"]) ** 2
                + (r["move", "even"] - r["port", "odd"]) ** 2
            ) / 2
            rows.append(
                dict(
                    track=t["track"],
                    mouse=t["mouse"],
                    kind=t["kind"],
                    term="movement",
                    days=0,
                    value=square - (noise_port + noise_move) / 2,
                    noise=noise_port,
                    signed=(r["move", "odd"] + r["move", "even"]) / 2
                    - (r["port", "odd"] + r["port", "even"]) / 2,
                )
            )
            halves.append((s, r, noise_port))
        for i in range(len(halves)):
            for j in range(i + 1, len(halves)):
                (s1, r1, n1), (s2, r2, n2) = halves[i], halves[j]
                if across_break(t["mouse"], s1["session"], s2["session"]):
                    continue
                square = (
                    (r2["port", "odd"] - r1["port", "even"]) ** 2
                    + (r2["port", "even"] - r1["port", "odd"]) ** 2
                ) / 2
                rows.append(
                    dict(
                        track=t["track"],
                        mouse=t["mouse"],
                        kind=t["kind"],
                        term="drift",
                        days=abs(s2["day"] - s1["day"]),
                        value=square - (n1 + n2) / 2,
                        noise=np.nan,
                        signed=np.nan,
                    )
                )
    return pd.DataFrame(rows)


def rms(values) -> float:
    return float(np.sqrt(max(np.mean(values), 0.0)))


def movement_thirds(tracks: list) -> list:
    """Per track with 3+ sessions: the movement change in each third of movement time."""
    out = []
    for t in tracks:
        if len(t["sessions"]) < 3:
            continue
        pooled = {
            k: {"port": [], "move": [], "latency": [], "time": []} for k in THIRDS
        }
        for s in t["sessions"]:
            edges = np.quantile(s["move_time"], [1 / 3, 2 / 3])
            label = np.digitize(s["move_time"], edges)
            for k, name in enumerate(THIRDS):
                q = label == k
                for key, value in (
                    ("port", s["port"][q]),
                    ("move", s["move"][q]),
                    ("latency", s["latency"][q]),
                    ("time", s["move_time"][q]),
                ):
                    pooled[name][key].append(value)
        changes = {}
        for name, p in pooled.items():
            port, move = np.concatenate(p["port"]), np.concatenate(p["move"])
            changes[name] = response(move, t["window"], t["norm"]) - response(
                port, t["window"], t["norm"]
            )
        out.append(
            dict(
                track=t["track"],
                mouse=t["mouse"],
                kind=t["kind"],
                changes=changes,
                latency={
                    k: float(np.median(np.concatenate(p["latency"])))
                    for k, p in pooled.items()
                },
                time={
                    k: float(np.median(np.concatenate(p["time"])))
                    for k, p in pooled.items()
                },
            )
        )
    return out


def pair_share(units: pd.DataFrame, mouse: str) -> dict:
    """Share of completed trials with a gap-matched pair, in each third of movement time."""
    shares = {k: [] for k in THIRDS}
    one = units[units["session"] >= FIRST_SESSION[mouse]].drop_duplicates("session")
    for _, u in one.iterrows():
        times, paired = u["all_move_time"], u["all_paired"]
        if not isinstance(times, np.ndarray) or len(times) < 15:
            continue
        label = np.digitize(times, np.quantile(times, [1 / 3, 2 / 3]))
        for k, name in enumerate(THIRDS):
            shares[name].append(float(np.mean(paired[label == k])))
    return {k: float(np.median(v)) for k, v in shares.items() if v}


def narrow_threshold(durations: np.ndarray) -> float:
    """The minimum of the density of spike durations from 0.2 to 0.6 ms."""
    grid = np.linspace(0.2, 0.6, 401)
    density = gaussian_kde(durations[np.isfinite(durations)])(grid)
    return float(grid[np.argmin(density)])


def partial_mantel(neural, behaviour, days, rng) -> dict:
    """Correlation of neural and behavioural distances with days regressed out of both."""
    n = neural.shape[0]
    iu = np.triu_indices(n, 1)

    def residual(matrix):
        y = matrix[iu]
        x = np.c_[np.ones(len(y)), days[iu]]
        return y - x @ np.linalg.lstsq(x, y, rcond=None)[0]

    rb = residual(behaviour)

    def stat(matrix):
        return float(np.corrcoef(residual(matrix), rb)[0, 1])

    observed = stat(neural)
    null = []
    for _ in range(N_PERMUTATIONS):
        p = rng.permutation(n)
        null.append(stat(neural[np.ix_(p, p)]))
    null = np.asarray(null)
    plain = float(np.corrcoef(neural[iu], behaviour[iu])[0, 1])
    return {
        "sessions": int(n),
        "pairs": int(len(iu[0])),
        "r_plain": plain,
        "r_partial_days": observed,
        "p_permutation": float(
            (1 + np.sum(np.abs(null) >= abs(observed))) / (1 + len(null))
        ),
    }


def behaviour_drift(units: pd.DataFrame, pairs: pd.DataFrame, rng) -> dict:
    """GRB064: first-flash change between sessions against movement-time change."""
    use = units[units["session"] >= FIRST_SESSION["GRB064"]]
    tables = pair_tables(use, pairs)
    same, good = tables["same"], tables["good"]
    keep = same[same["cls_a"].isin(KINDS) & (same["cls_b"] == same["cls_a"])].copy()
    keep["session_a"] = good.loc[keep["a"], "session"].to_numpy()
    keep["session_b"] = good.loc[keep["b"], "session"].to_numpy()
    sessions = sorted(
        s
        for s, u in use.drop_duplicates("session").set_index("session").iterrows()
        if isinstance(u["all_move_time"], np.ndarray) and len(u["all_move_time"]) >= 15
    )
    facts = use.drop_duplicates("session").set_index("session")
    move = np.array([np.median(facts.loc[s, "all_move_time"]) for s in sessions])
    day = np.array([facts.loc[s, "day"] for s in sessions], float)
    n = len(sessions)
    neural = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i + 1, n):
            q = keep[
                (
                    (keep["session_a"] == sessions[i])
                    & (keep["session_b"] == sessions[j])
                )
                | (
                    (keep["session_a"] == sessions[j])
                    & (keep["session_b"] == sessions[i])
                )
            ]["ff_r"].dropna()
            if len(q) >= 5:
                neural[i, j] = neural[j, i] = 1 - float(np.median(q))
    ok = np.array([np.sum(~np.isnan(neural[k])) == n - 1 for k in range(n)])
    sessions = [s for s, o in zip(sessions, ok, strict=True) if o]
    neural, move, day = neural[np.ix_(ok, ok)], move[ok], day[ok]
    np.fill_diagonal(neural, 0)
    behaviour = np.abs(move[:, None] - move[None, :]) * 1000
    days = np.abs(day[:, None] - day[None, :])
    result = partial_mantel(neural, behaviour, days, rng)
    iu = np.triu_indices(len(sessions), 1)
    result.update(
        sessions_used=[s[:8] for s in sessions],
        median_move_time_ms={
            s[:8]: round(float(m) * 1000) for s, m in zip(sessions, move, strict=True)
        },
        points={
            "behaviour_ms": behaviour[iu].tolist(),
            "neural": neural[iu].tolist(),
            "days": days[iu].tolist(),
        },
    )
    return result


RESP = (MOV_CENTERS_MS > 30) & (MOV_CENTERS_MS < 120)
EARLY = (MOV_CENTERS_MS > 30) & (MOV_CENTERS_MS < 60)
LATE = (MOV_CENTERS_MS > 60) & (MOV_CENTERS_MS < 120)


def centre(c: np.ndarray, kind: str) -> float:
    """Centre of mass (ms) of the baseline-corrected response, 30–120 ms."""
    rate = c.mean(0) / BIN_S
    r = rate[RESP] - rate[MOV_BASE].mean()
    w = np.clip(r if kind == "excited" else -r, 0, None)
    return (
        float(np.sum(MOV_CENTERS_MS[RESP] * w) / np.sum(w)) if w.sum() > 0 else np.nan
    )


SIDES = {"left turn": -1, "right turn": 1}
# 20–40 ms after the flash: the fast transient after a left turn.
FAST = (MOV_CENTERS_MS > 20) & (MOV_CENTERS_MS < 40)
MIN_SIDE_TRIALS = 20


def trace(c: np.ndarray, norm: float) -> np.ndarray:
    rate = c.mean(0) / BIN_S
    return (rate - rate[MOV_BASE].mean()) / norm


def timing(tracks: list, rng, n_boot: int = 500) -> list:
    """Per unit (all sessions pooled): response timing and the fast part, by turn side.

    In the port the head is in the same place whatever the next choice is, so
    the split of in-port flashes by the upcoming side is the control.
    """
    out = []
    for t in tracks:
        port = np.concatenate([s["port"] for s in t["sessions"]])
        move = np.concatenate([s["move"] for s in t["sessions"]])
        side = np.concatenate([s["side"] for s in t["sessions"]])
        if len(port) < 60:
            continue
        kind, norm = t["kind"], t["norm"]
        a, b = centre(port, kind), centre(move, kind)
        if not np.isfinite(a) or not np.isfinite(b):
            continue
        idx = rng.integers(0, len(port), (n_boot, len(port)))
        draws = [centre(move[i], kind) - centre(port[i], kind) for i in idx]
        item = dict(
            track=t["track"],
            mouse=t["mouse"],
            kind=kind,
            trials=len(port),
            port=a,
            move=b,
            shift=b - a,
            ci=[float(v) for v in np.nanpercentile(draws, [2.5, 97.5])],
            early=response(move, EARLY, norm) - response(port, EARLY, norm),
            late=response(move, LATE, norm) - response(port, LATE, norm),
            full=response(move, RESP, norm) - response(port, RESP, norm),
            traces={"in port": trace(port, norm)},
        )
        sides_ok = all(np.sum(side == v) >= MIN_SIDE_TRIALS for v in SIDES.values())
        if sides_ok:
            for name, v in SIDES.items():
                q = side == v
                item["traces"][f"after exit, {name}"] = trace(move[q], norm)
                item[f"shift, {name}"] = centre(move[q], kind) - centre(port[q], kind)
                item[f"fast in port, {name}"] = response(port[q], FAST, norm)
                item[f"fast after exit, {name}"] = response(move[q], FAST, norm)
                item[f"full in port, {name}"] = response(port[q], RESP, norm)
                item[f"full after exit, {name}"] = response(move[q], RESP, norm)
            item["fast left − right, in port"] = (
                item["fast in port, left turn"] - item["fast in port, right turn"]
            )
            item["fast left − right, after exit"] = (
                item["fast after exit, left turn"] - item["fast after exit, right turn"]
            )
            for name in SIDES:
                item[f"full change, {name}"] = (
                    item[f"full after exit, {name}"] - item[f"full in port, {name}"]
                )
        out.append(item)
    return out


def strip(ax, x: float, groups: dict, color, seed: int) -> dict:
    """Units of each mouse as light points in its marker, open medians, pooled median ± interval."""
    rng = np.random.default_rng(seed)
    groups = {m: np.asarray(v, float)[np.isfinite(v)] for m, v in groups.items()}
    for k, (mouse, values) in enumerate(groups.items()):
        if not len(values):
            continue
        ax.scatter(
            x - 0.12 + 0.24 * k + rng.uniform(-0.07, 0.07, len(values)),
            values,
            s=7,
            marker=MARKERS[mouse],
            color=color,
            alpha=0.35,
            lw=0,
        )
        median_marker(
            ax, x + 0.38, float(np.median(values)), MARKERS[mouse], color, size=5
        )
    stats = hierarchical_interval([v for v in groups.values() if len(v)])
    ax.plot([x - 0.3, x + 0.3], [stats["median"]] * 2, color=color, lw=2.2)
    ax.plot([x, x], stats["interval_95"], color=color, lw=1.3)
    return stats


def mouse_key(ax, x: float, y: float) -> None:
    """The marker of each mouse, once per figure."""
    for k, mouse in enumerate(MICE):
        ax.plot(
            x,
            y - 0.06 * k,
            MARKERS[mouse],
            ms=4,
            color="black",
            transform=ax.transAxes,
            clip_on=False,
        )
        ax.text(
            x + 0.035,
            y - 0.06 * k,
            mouse,
            transform=ax.transAxes,
            va="center",
            fontsize=6,
        )


def plot(
    report: dict,
    change: pd.DataFrame,
    thirds: list,
    tracks: list,
    times: list,
    threshold: float,
) -> None:
    with plt.rc_context(FIGURE_STYLE):
        fig = plt.figure(figsize=(11, 7.8))
        grid = fig.add_gridspec(2, 3, hspace=0.62, wspace=0.48)
        axes = [fig.add_subplot(grid[r, c]) for r in range(2) for c in range(3)]

        # a: movement against drift, GRB064
        ax = axes[0]
        for kind in KINDS:
            color = CLASS_COLORS[kind]
            r = report["movement_vs_drift"]["GRB064"][kind]
            x = [b["days_median"] for b in r["drift"]]
            ax.plot(
                x,
                [b["rms"] for b in r["drift"]],
                color=color,
                lw=1.5,
                marker="o",
                ms=3.5,
            )
            for b in r["drift"]:
                ax.plot([b["days_median"]] * 2, b["ci"], color=color, lw=1.0)
            m = r["movement"]
            ax.plot([-3, -3], m["ci"], color=color, lw=1.3)
            ax.plot(-3, m["rms"], "D", color=color, ms=5)
        ax.set_xlim(-6, 40)
        ax.set_ylim(bottom=0)
        ax.set_xticks([0, 10, 20, 30, 40])
        ax.text(
            -3,
            -0.06,
            "after exit\nvs in port",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=6,
        )
        ax.set_xlabel("\nin port, days between sessions (GRB064)")
        ax.set_ylabel(
            "true change of the flash response\n(root mean square over units, ÷ mean rate)"
        )
        for kind, y in zip(KINDS, (0.95, 0.85), strict=True):
            ax.text(
                0.98,
                y,
                CLASS_NAMES[kind],
                color=CLASS_COLORS[kind],
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=6.5,
            )

        # b: mean flash response, in port and after each turn
        sub = axes[1].get_subplotspec().subgridspec(2, 1, hspace=0.12)
        axes[1].remove()
        axes[1] = fig.add_subplot(sub[0])
        ax_low = fig.add_subplot(sub[1], sharex=axes[1])
        styles = {
            "in port": "-",
            "after exit, left turn": "--",
            "after exit, right turn": ":",
        }
        for ax, kind in ((axes[1], "excited"), (ax_low, "suppressed")):
            color = CLASS_COLORS[kind]
            items = [t for t in times if t["kind"] == kind and len(t["traces"]) == 3]
            for name, ls in styles.items():
                traces = np.array([t["traces"][name] for t in items])
                mean = traces.mean(0)
                sem = traces.std(0, ddof=1) / np.sqrt(len(traces))
                ax.plot(MOV_CENTERS_MS, mean, color=color, ls=ls, lw=1.4)
                ax.fill_between(
                    MOV_CENTERS_MS,
                    mean - sem,
                    mean + sem,
                    color=color,
                    alpha=0.12,
                    lw=0,
                )
            ax.axvline(0, color="black", ls="--", lw=0.8)
            ax.axhline(0, color=REFERENCE_COLOR, lw=0.6)
            ax.text(
                0.02,
                0.95,
                f"{CLASS_NAMES[kind]}, {len(items)} units",
                color=color,
                transform=ax.transAxes,
                va="top",
                fontsize=6.5,
            )
        axes[1].tick_params(labelbottom=False)
        axes[1].text(
            0.98,
            0.95,
            "solid: in port\ndashed: after a left turn\ndotted: after a right turn",
            transform=axes[1].transAxes,
            ha="right",
            va="top",
            fontsize=6,
        )
        ax_low.set_xlabel("time from flash (ms)")
        axes[1].set_ylabel("flash response\n((rate − baseline) ÷ mean rate)")
        axes[1].yaxis.set_label_coords(-0.16, 0.0)

        # c: timing after each turn
        ax = axes[2]
        names = [
            ("excited", "left turn"),
            ("excited", "right turn"),
            ("suppressed", "left turn"),
            ("suppressed", "right turn"),
        ]
        values = []
        for x, (kind, side) in enumerate(names):
            groups = {
                m: np.array(
                    [
                        t[f"shift, {side}"]
                        for t in times
                        if t["kind"] == kind
                        and t["mouse"] == m
                        and f"shift, {side}" in t
                    ]
                )
                for m in MICE
            }
            strip(ax, x, groups, CLASS_COLORS[kind], seed=x)
            values.extend(np.concatenate(list(groups.values())))
        ax.axhline(0, color=REFERENCE_COLOR, ls="--", lw=0.8)
        ax.set_xticks(
            range(4), ["left\nturn", "right\nturn", "left\nturn", "right\nturn"]
        )
        for tick, (kind, _) in zip(ax.get_xticklabels(), names, strict=True):
            tick.set_color(CLASS_COLORS[kind])
        values = np.asarray(values)
        lim = float(np.nanpercentile(np.abs(values), 98))
        ax.set_ylim(-lim, lim)
        ax.set_xlim(-0.6, 3.7)
        ax.set_ylabel("response timing, after exit − in port\n(centre of mass, ms)")
        ax.set_xlabel(
            f"one point per unit ({int(np.sum(np.abs(values) > lim))} past the axes)"
        )
        mouse_key(ax, 0.62, 0.16)

        # d: the fast part (20–40 ms), left − right turn: in port (control) and after exit
        ax = axes[3]
        names = [
            ("excited", "in port"),
            ("excited", "after exit"),
            ("suppressed", "in port"),
            ("suppressed", "after exit"),
        ]
        values = []
        for x, (kind, period) in enumerate(names):
            key = f"fast left − right, {period}"
            groups = {
                m: np.array(
                    [
                        t[key]
                        for t in times
                        if t["kind"] == kind and t["mouse"] == m and key in t
                    ]
                )
                for m in MICE
            }
            strip(ax, x, groups, CLASS_COLORS[kind], seed=30 + x)
            values.extend(np.concatenate(list(groups.values())))
        ax.axhline(0, color=REFERENCE_COLOR, ls="--", lw=0.8)
        ax.set_xticks(
            range(4),
            [
                "in port\n(next\nchoice)",
                "after\nexit",
                "in port\n(next\nchoice)",
                "after\nexit",
            ],
        )
        for tick, (kind, _) in zip(ax.get_xticklabels(), names, strict=True):
            tick.set_color(CLASS_COLORS[kind])
        values = np.asarray(values)
        lim = float(np.nanpercentile(np.abs(values), 98))
        ax.set_ylim(-lim, lim)
        ax.set_xlim(-0.6, 3.7)
        ax.set_ylabel("20–40 ms response, left − right turn\n(÷ mean rate)")
        ax.set_xlabel(
            f"one point per unit ({int(np.sum(np.abs(values) > lim))} past the axes)"
        )

        # e: movement time
        ax = axes[4]
        for k, kind in enumerate(KINDS):
            color = CLASS_COLORS[kind]
            for t in [t for t in thirds if t["kind"] == kind]:
                ax.plot(
                    np.arange(3) + 0.08 * (k - 0.5),
                    [t["changes"][n] for n in THIRDS],
                    color=color,
                    alpha=0.1,
                    lw=0.6,
                )
            r = report["movement_time"][kind]
            ax.errorbar(
                np.arange(3) + 0.08 * (k - 0.5),
                [r["median"][n] for n in THIRDS],
                yerr=np.abs(
                    np.array([r["ci"][n] for n in THIRDS]).T
                    - np.array([r["median"][n] for n in THIRDS])
                ),
                color=color,
                marker="o",
                ms=4,
                lw=1.6,
                capsize=2,
            )
        ax.set_xticks(
            range(3),
            [f"{n}\n{report['movement_time']['time_ms'][n]:.0f} ms" for n in THIRDS],
            fontsize=6.5,
        )
        ax.axhline(0, color=REFERENCE_COLOR, ls="--", lw=0.8)
        every = np.array([t["changes"][n] for t in thirds for n in THIRDS])
        lim = float(np.nanpercentile(np.abs(every), 98))
        ax.set_ylim(-lim, lim)
        report["movement_time"]["unit_values_past_axes"] = int(
            np.sum(np.abs(every) > lim)
        )
        lat = report["movement_time"]["latency_ms"]
        ax.set_xlabel(
            "third of movement time (median)\n"
            f"flash {min(lat.values()):.0f}–{max(lat.values()):.0f} ms after exit; "
            f"{report['movement_time']['unit_values_past_axes']} values past the axes"
        )
        ax.set_ylabel(
            "movement change of the flash response\n(after exit − in port, ÷ mean rate)"
        )

        # f: spike type
        ax = axes[5]
        for k, kind in enumerate(KINDS):
            color = CLASS_COLORS[kind]
            for j, width in enumerate(("narrow", "broad")):
                entry = report["spike_duration"][kind][width]
                groups = {m: np.array(entry["by_mouse"][m]) for m in MICE}
                strip(ax, 2 * k + j, groups, color, seed=10 + 2 * k + j)
            p = report["spike_duration"][kind]["p_mann_whitney"]
            ax.text(
                2 * k + 0.5,
                1.02,
                f"p = {p:.2g}",
                transform=ax.get_xaxis_transform(),
                ha="center",
                fontsize=6.5,
                color=color,
            )
        ax.axhline(0, color=REFERENCE_COLOR, ls="--", lw=0.8)
        ax.set_xticks(range(4), ["narrow", "broad", "narrow", "broad"])
        for tick, kind in zip(
            ax.get_xticklabels(),
            ("excited", "excited", "suppressed", "suppressed"),
            strict=True,
        ):
            tick.set_color(CLASS_COLORS[kind])
        lim = report["spike_duration"]["display_limit"]
        ax.set_ylim(-lim, lim)
        ax.set_xlim(-0.6, 3.7)
        ax.set_ylabel(
            "movement change of the flash response\n(all sessions pooled, ÷ mean rate)"
        )
        ax.set_xlabel(
            f"spike duration < or ≥ {threshold:.2f} ms "
            f"({report['spike_duration']['past_axes']} units past the axes)"
        )
        label_panels(axes, x=-0.14, y=1.05)
        fig.savefig(
            Path("figures/unit_tracking/movement.png"), dpi=150, bbox_inches="tight"
        )
        print(save_figure(fig, OUT / "movement"))


def turn_side_within_rate(mouse: str, sessions: list, rng) -> dict:
    """Population control: the fast after-exit response by turn side within each rate.

    All units of each session; the gap-matched after-exit flash of each
    completed trial; rate 20–40 ms after it minus -40–0 ms before it, in
    spikes/s per unit. Fast trains are rewarded on the right, so the turn side
    goes with the rate; the left-turn effect is fitted with the rate as a
    factor and tested by shuffling the side within each rate.
    """
    rows = []
    for session in sessions:
        try:
            trials = build_trial_table(mouse, session, include_frames=False)
        except ValueError:
            continue
        units = fetch_unit_table(mouse, session, 1, include_metrics=False)
        spikes = np.sort(
            np.concatenate([np.asarray(x, float) for x in units["spike_times_s"]])
        )
        done = trials[
            (trials["early_withdrawal"] == 0) & trials["response"].isin((-1, 1))
        ]
        for _, r in done.iterrows():
            f = np.asarray(r["stim_pulse_times_s"], float)
            ends = [r["center_entry_s"], r["center_exit_s"], r["response_port_entry_s"]]
            if not len(f) or not np.isfinite(ends).all():
                continue
            gap = np.diff(f, prepend=np.nan) * 1000
            outside = (
                (gap >= GAP_MIN_MS)
                & (f + MOV_EDGES[0] >= ends[1])
                & (f + MOV_EDGES[-1] <= ends[2])
            )
            if not outside.any():
                continue
            x = f[outside][0]
            base = np.diff(np.searchsorted(spikes, [x - 0.04, x]))[0] / 0.04
            fast = np.diff(np.searchsorted(spikes, [x + 0.02, x + 0.04]))[0] / 0.02
            rows.append(
                (
                    float(r["visual_stim_rate_hz"]),
                    int(r["response"]),
                    int(r["rewarded"]),
                    (fast - base) / len(units),
                    float(gap[outside][0]),
                    1000 * float(x - ends[1]),
                )
            )
    d = pd.DataFrame(
        rows, columns=["rate", "side", "rewarded", "fast", "gap_ms", "latency_ms"]
    )
    x = pd.get_dummies(d["rate"].astype(str)).astype(float)
    x["left"] = (d["side"] == -1).astype(float)
    effect = float(
        np.linalg.lstsq(x.to_numpy(), d["fast"].to_numpy(), rcond=None)[0][-1]
    )
    null = []
    for _ in range(N_PERMUTATIONS // 5):
        shuffled = d.groupby("rate")["side"].transform(
            lambda v: rng.permutation(v.to_numpy())
        )
        x["left"] = (shuffled.to_numpy() == -1).astype(float)
        null.append(
            np.linalg.lstsq(x.to_numpy(), d["fast"].to_numpy(), rcond=None)[0][-1]
        )
    within = {}
    for rate, q in d.groupby("rate"):
        a, b = q[q["side"] == -1]["fast"], q[q["side"] == 1]["fast"]
        if len(a) >= 5 and len(b) >= 5:
            within[f"{rate:g} Hz"] = {
                "left − right": float(a.mean() - b.mean()),
                "n left": len(a),
                "n right": len(b),
            }
    errors = d[d["rewarded"] == 0]
    return {
        "trials": int(len(d)),
        "left_turn_effect_rate_as_factor": effect,
        "p_side_shuffled_within_rate": float(
            (1 + np.sum(np.abs(null) >= abs(effect))) / (1 + len(null))
        ),
        "within_rate": within,
        "error_trials_mean_by_side": {
            "left": float(errors.loc[errors["side"] == -1, "fast"].mean()),
            "right": float(errors.loc[errors["side"] == 1, "fast"].mean()),
            "n left": int((errors["side"] == -1).sum()),
            "n right": int((errors["side"] == 1).sum()),
        },
        "median_gap_ms_by_side": {
            "left": float(d.loc[d["side"] == -1, "gap_ms"].median()),
            "right": float(d.loc[d["side"] == 1, "gap_ms"].median()),
        },
        "median_latency_ms_by_side": {
            "left": float(d.loc[d["side"] == -1, "latency_ms"].median()),
            "right": float(d.loc[d["side"] == 1, "latency_ms"].median()),
        },
    }


def drift_report(change: pd.DataFrame, rng) -> dict:
    out = {}
    for mouse in MICE:
        out[mouse] = {}
        for kind in KINDS:
            q = change[(change["kind"] == kind) & (change["mouse"] == mouse)]
            mv = q[q["term"] == "movement"]
            drift = []
            for lo, hi in zip(DAY_BINS[:-1], DAY_BINS[1:], strict=False):
                d = q[(q["term"] == "drift") & (q["days"] >= lo) & (q["days"] < hi)]
                if len(d) >= 5:
                    drift.append(
                        {
                            "days_median": float(np.median(d["days"])),
                            "rms": rms(d["value"]),
                            "ci": track_boot(d["value"], d["track"], rng, stat=rms),
                            "pairs": int(len(d)),
                        }
                    )
            alld = q[q["term"] == "drift"]
            out[mouse][kind] = {
                "movement": {
                    "rms": rms(mv["value"]),
                    "ci": track_boot(mv["value"], mv["track"], rng, stat=rms),
                    "unit_sessions": int(len(mv)),
                    "median_signed_change": float(np.median(mv["signed"])),
                    "noise_rms_half_trials": float(np.sqrt(np.mean(mv["noise"]))),
                },
                "drift": drift,
                "drift_all_days": {
                    "rms": rms(alld["value"]) if len(alld) else None,
                    "ci": track_boot(alld["value"], alld["track"], rng, stat=rms)
                    if len(alld)
                    else None,
                    "pairs": int(len(alld)),
                },
            }
    return out


def main() -> None:
    rng = np.random.default_rng(0)
    Path("figures/unit_tracking").mkdir(parents=True, exist_ok=True)
    tracks, data = [], {}
    for mouse in MICE:
        t, units, pairs = build_tracks(mouse)
        tracks.extend(t)
        data[mouse] = (units, pairs)
    report: dict = {
        "tracks": {m: sum(t["mouse"] == m for t in tracks) for m in MICE},
        "tracks_by_class": {
            k: {
                m: sum(t["kind"] == k and t["mouse"] == m for t in tracks) for m in MICE
            }
            for k in KINDS
        },
    }
    change = movement_and_drift(tracks)
    report["movement_vs_drift"] = drift_report(change, rng)

    times = timing(tracks, rng)
    report["timing"] = {}
    keys = (
        "shift",
        "early",
        "late",
        "full",
        "shift, left turn",
        "shift, right turn",
        "fast in port, left turn",
        "fast in port, right turn",
        "fast after exit, left turn",
        "fast after exit, right turn",
        "fast left − right, in port",
        "fast left − right, after exit",
        "full change, left turn",
        "full change, right turn",
    )
    for kind in KINDS:
        items = [t for t in times if t["kind"] == kind]
        entry = {}
        for key in keys:
            groups = [
                np.array([t[key] for t in items if t["mouse"] == m and key in t])
                for m in MICE
            ]
            groups = [g[np.isfinite(g)] for g in groups if len(g)]
            entry[key] = hierarchical_interval(groups)
            entry[key]["units"] = [int(len(g)) for g in groups]
        entry["units_per_mouse"] = {
            m: sum(t["mouse"] == m for t in items) for m in MICE
        }
        entry["earlier_interval_below_0"] = {
            m: sum(t["ci"][1] < 0 for t in items if t["mouse"] == m) for m in MICE
        }
        entry["later_interval_above_0"] = {
            m: sum(t["ci"][0] > 0 for t in items if t["mouse"] == m) for m in MICE
        }
        entry["centre_ms_median"] = {
            m: {
                "in port": float(
                    np.median([t["port"] for t in items if t["mouse"] == m])
                ),
                "after exit": float(
                    np.median([t["move"] for t in items if t["mouse"] == m])
                ),
            }
            for m in MICE
            if any(t["mouse"] == m for t in items)
        }
        report["timing"][kind] = entry

    thirds = movement_thirds(tracks)
    report["movement_time"] = {
        "time_ms": {
            n: 1000 * float(np.median([t["time"][n] for t in thirds])) for n in THIRDS
        },
        "latency_ms": {
            n: 1000 * float(np.median([t["latency"][n] for t in thirds]))
            for n in THIRDS
        },
        "share_of_trials_paired": {m: pair_share(data[m][0], m) for m in MICE},
    }
    for kind in KINDS:
        items = [t for t in thirds if t["kind"] == kind]
        med, ci = {}, {}
        for n in THIRDS:
            groups = [
                np.array([t["changes"][n] for t in items if t["mouse"] == m])
                for m in MICE
            ]
            s = hierarchical_interval([g for g in groups if len(g)])
            med[n], ci[n] = s["median"], s["interval_95"]
        slope_groups = [
            np.array(
                [
                    t["changes"]["fast"] - t["changes"]["slow"]
                    for t in items
                    if t["mouse"] == m
                ]
            )
            for m in MICE
        ]
        report["movement_time"][kind] = {
            "units_per_mouse": {
                m: int(len(g)) for m, g in zip(MICE, slope_groups, strict=True)
            },
            "median": med,
            "ci": ci,
            "fast_minus_slow": hierarchical_interval(
                [g for g in slope_groups if len(g)]
            ),
        }

    threshold = narrow_threshold(np.array([t["spike_duration_ms"] for t in tracks]))
    report["spike_duration"] = {"threshold_ms": threshold}
    pooled = {}
    for t in tracks:
        port = np.concatenate([s["port"] for s in t["sessions"]])
        move = np.concatenate([s["move"] for s in t["sessions"]])
        pooled[t["track"]] = response(move, t["window"], t["norm"]) - response(
            port, t["window"], t["norm"]
        )
    allv = np.array(list(pooled.values()))
    lim = float(np.nanpercentile(np.abs(allv), 98))
    report["spike_duration"]["display_limit"] = lim
    report["spike_duration"]["past_axes"] = int(np.sum(np.abs(allv) > lim))
    for kind in KINDS:
        entry = {}
        for width, test in (
            ("narrow", lambda d: d < threshold),
            ("broad", lambda d: d >= threshold),
        ):
            sel = [
                t for t in tracks if t["kind"] == kind and test(t["spike_duration_ms"])
            ]
            by_mouse = {
                m: [pooled[t["track"]] for t in sel if t["mouse"] == m] for m in MICE
            }
            groups = [np.array(v) for v in by_mouse.values() if len(v)]
            s = hierarchical_interval(groups)
            entry[width] = {
                "units_per_mouse": {m: len(v) for m, v in by_mouse.items()},
                "median": s["median"],
                "ci": s["interval_95"],
                "median_of_each_mouse": s["median_of_each_mouse"],
                "by_mouse": by_mouse,
            }
        a = np.concatenate([np.array(v) for v in entry["narrow"]["by_mouse"].values()])
        b = np.concatenate([np.array(v) for v in entry["broad"]["by_mouse"].values()])
        entry["p_mann_whitney"] = float(mannwhitneyu(a, b).pvalue)
        report["spike_duration"][kind] = entry

    units64, pairs64 = data["GRB064"]
    report["behaviour_drift"] = behaviour_drift(units64, pairs64, rng)
    report["turn_side_within_rate"] = {
        m: turn_side_within_rate(
            m,
            sorted(s for s in data[m][0]["session"].unique() if s >= FIRST_SESSION[m]),
            rng,
        )
        for m in MICE
    }
    plot(report, change, thirds, tracks, times, threshold)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "movement.json").write_text(json.dumps(report, indent=1, default=float))
    print("written")


if __name__ == "__main__":
    main()
