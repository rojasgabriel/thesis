"""Compare the flash response in the port and after the center exit, paired in each trial.

Question: are V1 flash responses larger after the center exit than in the
port? First with the paired flashes of every trial, then split by the gap
from the flash before. No model.

Comparison: in each trial, the last flash in the port against the first flash
after the exit, so every unit has a value in both periods (paired).

- Sessions and trials: the GLM sessions (`flash_kernels.SESSIONS`) and the
  selected GLM trials. Units: the GLM units; the statistics use the
  flash-excited units (`peaks.load_excited_units`, 30 and 22).
- Flashes: the last flash whose window (`WINDOW_S`, -40 to 120 ms) lies in the
  port (center entry to center exit), and the first flash at or after the
  center exit whose window ends before the response-port entry. Only trials
  with both.
- By gap: in each trial, the last in-port flash and the first after-exit flash
  of the same gap class (gap from the flash before, rounded to 40 ms; 40, 80,
  120, 160, 200+ ms; the first flash of a trial has no gap).
- Measure (as `peaks.py`): per unit and condition, the mean rate in 10-ms
  bins; baseline = the in-port rate from -40 to 0 ms, the same baseline for
  both periods; peak = the mean over the bins within 20 ms of the largest bin
  from 30 to 120 ms (`peaks.window_peak`; until 2026-11-12 the largest bin
  alone), minus that baseline, found in each period apart;
  divided by the unit's mean rate (0.5 s before to 1.5 s after the first
  flash). The sampling unit is the unit; tests within each mouse (Wilcoxon
  signed-rank, after exit against in port).
- Example units: flash-excited, mean rate >= 3 spikes/s, the two with the
  largest (after - before) / sqrt(after + before) over the in-port flashes at
  least 80 ms after the flash before (as `dataset_overview.py`). The rasters
  show the same trials in the same order on both axes.
- Scatter: marker = mouse, color = the direction of the fixation modulation
  (`hold_effect.fixation_groups`, `plotting.FIXATION_COLORS`).
- Display limit of the scatter (`SCATTER_LIMIT`): points past it are drawn at
  the edge as open triangles and stay in the statistics; the report lists
  them.

- Fixation-suppressed units (`exit_flash_population_fixation_suppressed.svg`):
  the population figure for the flash-excited units whose fixation level is
  below 0 by more than 2 fold SDs (`hold_effect.fixation_suppressed`).

Writes `figures/glm/exit_flash_examples.svg`, `exit_flash_population.svg`,
`exit_flash_population_fixation_suppressed.svg`,
`exit_flash_by_gap.svg` and `results/glm/exit_flash_pairs.json`.

    THESIS_CACHE_DIR=figures/cache uv run python -m \\
        thesis.ephys.analyses.glm.exit_flash_pairs
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy.stats import wilcoxon

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thesis.ephys.analyses.glm import dmat
from thesis.ephys.analyses.glm.fit import analysis_unit_indices
from thesis.ephys.analyses.glm.flash_kernels import SESSIONS
from thesis.ephys.analyses.glm.hold_effect import (
    FIXATION_GROUPS,
    fixation_groups,
    fixation_suppressed,
)
from thesis.ephys.analyses.glm.peaks import load_excited_units, window_peak
from thesis.ephys.units import fetch_unit_table
from thesis.plotting import (
    FIGURE_STYLE,
    FIXATION_COLORS,
    PERIOD_COLORS,
    label_panels,
    mouse_marker,
    save_figure,
)

OUT = Path("results/glm")
WINDOW_S = (-0.04, 0.12)
EDGES = np.arange(WINDOW_S[0], WINDOW_S[1] + 1e-9, 0.01)
CENTERS = (EDGES[:-1] + EDGES[1:]) / 2
BASE = (CENTERS >= -0.04) & (CENTERS < 0)
PEAK = (CENTERS >= 0.03) & (CENTERS < 0.12)
GAPS = (40, 80, 120, 160, 200)
SCATTER_LIMIT = (-1.0, 4.0)
PORT, EXIT = PERIOD_COLORS["in port"], PERIOD_COLORS["after exit"]
NAMES = {"in port": "last in port", "after exit": "first after exit"}


def counts(spikes, events):
    i = np.searchsorted(spikes, events[:, None] + EDGES[None, :])
    return np.diff(i, axis=1)


def session(sess, excited_ids):
    subject, stamp = sess.split("_", 1)
    byg = {}
    prepared = dmat._load_windows(OUT / sess / "stimulus_windows_me.npz")
    trials = dmat._load_selected_trials(prepared)
    units = fetch_unit_table(subject, stamp, 1, 0, include_metrics=False)
    units = units.iloc[analysis_unit_indices(units, "all")].reset_index(drop=True)
    ev = {"in port": ([], []), "after exit": ([], [])}
    rank = []  # in-port flashes >= 80 ms after the flash before, to pick examples
    first = []
    for times, en, ex, rp in zip(
        trials["stim_pulse_times_s"],
        trials["center_entry_s"].to_numpy(float),
        trials["center_exit_s"].to_numpy(float),
        trials["response_port_entry_s"].to_numpy(float),
        strict=True,
    ):
        t = np.asarray(times, float)
        first.append(t[0])
        gap = np.round(np.diff(t, prepend=-np.inf) * 1000 / 40) * 40
        inside = (t + WINDOW_S[0] >= en) & (t + WINDOW_S[1] <= ex)
        outside = (t >= ex) & (t + WINDOW_S[1] <= rp)
        rank.extend(t[inside & (gap >= 80) & np.isfinite(gap)])
        if inside.any() and outside.any():
            i, o = np.flatnonzero(inside)[-1], np.flatnonzero(outside)[0]
            ev["in port"][0].append(t[i])
            ev["in port"][1].append(gap[i])
            ev["after exit"][0].append(t[o])
            ev["after exit"][1].append(gap[o])
        for g in GAPS:
            cls = (gap >= g) & np.isfinite(gap) if g == GAPS[-1] else gap == g
            if (inside & cls).any() and (outside & cls).any():
                byg.setdefault(g, {"in port": [], "after exit": []})
                byg[g]["in port"].append(t[np.flatnonzero(inside & cls)[-1]])
                byg[g]["after exit"].append(t[np.flatnonzero(outside & cls)[0]])
    ev = {k: (np.asarray(a), np.asarray(g)) for k, (a, g) in ev.items()}
    first = np.asarray(first)
    spikes = [np.sort(np.asarray(s, float)) for s in units["spike_times_s"]]
    ids = units["unit_id"].astype(int).to_numpy()
    mean_rate = np.array(
        [
            np.mean(
                np.diff(np.searchsorted(s, np.c_[first - 0.5, first + 1.5]), axis=1)
            )
            / 2.0
            for s in spikes
        ]
    )
    exc = np.array([i for i in range(len(ids)) if ids[i] in excited_ids])
    groups = fixation_groups(OUT / sess)
    return dict(
        group=np.asarray([groups.get(int(i), "") for i in ids]),
        subject=subject,
        ids=ids,
        spikes=spikes,
        ev=ev,
        mean_rate=mean_rate,
        exc=exc,
        rank=np.asarray(rank),
        byg={g: {k: np.asarray(v) for k, v in d.items()} for g, d in byg.items()},
    )


def rates(spk, events):
    return counts(spk, events).mean(0) / 0.01


def peaks(d, events=None):
    events = events or {k: d["ev"][k][0] for k in d["ev"]}
    out = {"in port": [], "after exit": []}
    for u in d["exc"]:
        r = {k: rates(d["spikes"][u], events[k]) for k in out}
        base = r["in port"][BASE].mean()
        for k in out:
            peak = window_peak(r[k], 1000 * CENTERS) - base
            out[k].append(peak / max(d["mean_rate"][u], 1e-9))
    return {k: np.asarray(v) for k, v in out.items()}


def plot_examples(data, report):
    """Four example units: paired raster axes (same trials, same order) and a PSTH."""
    with plt.rc_context(FIGURE_STYLE):
        fig = plt.figure(figsize=(11, 5.0))
        gs = fig.add_gridspec(
            3, 4, height_ratios=(1, 1, 1.05), hspace=0.12, wspace=0.35
        )
        col, tops = 0, []
        for d in data:
            cand = [u for u in d["exc"] if d["mean_rate"][u] >= 3]
            r = np.array([counts(d["spikes"][u], d["rank"]).mean(0) for u in cand])
            after, before = r[:, PEAK].sum(1), r[:, BASE].mean(1) * PEAK.sum()
            vis = (after - before) / np.sqrt(np.maximum(after + before, 1e-9))
            top = [cand[i] for i in np.argsort(vis)[::-1][:2]]
            report[d["subject"] + "_examples"] = [int(d["ids"][u]) for u in top]
            n = len(d["ev"]["in port"][0])
            for u in top:
                s_ = d["spikes"][u]
                axes = [fig.add_subplot(gs[row, col]) for row in range(3)]
                for ax_, k, c in (
                    (axes[0], "after exit", EXIT),
                    (axes[1], "in port", PORT),
                ):
                    for row, e in enumerate(d["ev"][k][0]):
                        x = s_[(s_ >= e + WINDOW_S[0]) & (s_ < e + WINDOW_S[1])] - e
                        ax_.plot(
                            1000 * x,
                            np.full(len(x), row),
                            "|",
                            color=c,
                            markersize=0.9,
                            mew=0.5,
                        )
                    ax_.set_ylim(n, 0)
                    ax_.set_xlim(1000 * WINDOW_S[0], 1000 * WINDOW_S[1])
                    ax_.axvline(0, color="0.6", ls="--", lw=0.7)
                    ax_.tick_params(labelbottom=False)
                    ax_.text(
                        1.02,
                        0.5,
                        NAMES[k],
                        transform=ax_.transAxes,
                        color=c,
                        rotation=90,
                        va="center",
                        fontsize=7,
                    )
                    ax_.set_yticks([1, n])
                    if col == 0:
                        ax_.set_ylabel("trial")
                axes[0].text(
                    0,
                    1.04,
                    f"{d['subject']}, unit {d['ids'][u]}",
                    transform=axes[0].transAxes,
                    va="bottom",
                )
                psth = axes[2]
                for k, c in (("in port", PORT), ("after exit", EXIT)):
                    cn = counts(s_, d["ev"][k][0]) / 0.01
                    m, se = cn.mean(0), cn.std(0, ddof=1) / np.sqrt(len(cn))
                    psth.plot(1000 * CENTERS, m, color=c)
                    psth.fill_between(
                        1000 * CENTERS, m - se, m + se, color=c, alpha=0.2, lw=0
                    )
                psth.axvline(0, color="0.6", ls="--", lw=0.7)
                psth.set_xlim(1000 * WINDOW_S[0], 1000 * WINDOW_S[1])
                psth.set_xlabel("time from flash (ms)")
                if col == 0:
                    psth.set_ylabel("rate (spikes/s)")
                tops.append(axes[0])
                col += 1
        label_panels(tops, x=-0.25, y=1.12)
        return save_figure(fig, OUT / "exit_flash_examples")


def plot_population(data, report, name="exit_flash_population", tag=""):
    """Mean PSTH of the flash-excited units of each mouse, and the paired peak scatter."""
    with plt.rc_context(FIGURE_STYLE):
        fig, ax = plt.subplots(
            1, 3, figsize=(12, 4.2), gridspec_kw={"width_ratios": (1, 1, 1.15)}
        )
        allp = {"in port": [], "after exit": []}
        for j, d in enumerate(data):
            a = ax[j]
            for k, c in (("in port", PORT), ("after exit", EXIT)):
                tr = np.array(
                    [
                        rates(d["spikes"][u], d["ev"][k][0])
                        / max(d["mean_rate"][u], 1e-9)
                        for u in d["exc"]
                    ]
                )
                m, se = tr.mean(0), tr.std(0, ddof=1) / np.sqrt(len(tr))
                a.plot(1000 * CENTERS, m, color=c)
                a.fill_between(1000 * CENTERS, m - se, m + se, color=c, alpha=0.2, lw=0)
                a.text(
                    0.98,
                    0.95 - 0.09 * (k == "after exit"),
                    f"{NAMES[k]} ({len(d['ev'][k][0])} trials)",
                    color=c,
                    transform=a.transAxes,
                    ha="right",
                    va="top",
                    fontsize=7,
                )
            a.axvline(0, color="0.6", ls="--", lw=0.7)
            a.set_xlabel("time from flash (ms)")
            a.text(
                0,
                1.02,
                f"{d['subject']}: {len(d['exc'])} "
                + d.get("exc_label", "flash-excited units"),
                transform=a.transAxes,
                va="bottom",
            )
            if j == 0:
                a.set_ylabel("rate ÷ mean rate")
            p = peaks(d)
            for k in allp:
                allp[k].append(p[k])
            w = wilcoxon(p["after exit"], p["in port"])
            report[d["subject"] + "_all" + tag] = dict(
                units=len(p["in port"]),
                higher=int(np.sum(p["after exit"] > p["in port"])),
                median_in_port=float(np.median(p["in port"])),
                median_after_exit=float(np.median(p["after exit"])),
                p=float(w.pvalue),
                flashes={k: int(len(d["ev"][k][0])) for k in d["ev"]},
            )
        ax[1].sharey(ax[0])
        ax[0].autoscale()
        a = ax[2]
        x, y = np.concatenate(allp["in port"]), np.concatenate(allp["after exit"])
        lo, hi = SCATTER_LIMIT
        out = (x > hi) | (y > hi) | (x < lo) | (y < lo)
        group = np.concatenate([d["group"][d["exc"]] for d in data])
        marker = np.concatenate(
            [
                [mouse_marker(d["subject"], j)] * len(d["exc"])
                for j, d in enumerate(data)
            ]
        )
        for fixation in FIXATION_GROUPS:
            color = FIXATION_COLORS[fixation]
            for mk in np.unique(marker):
                k = (group == fixation) & (marker == mk)
                a.scatter(
                    x[k & ~out],
                    y[k & ~out],
                    s=22,
                    marker=mk,
                    color=color,
                    alpha=0.75,
                    lw=0,
                )
                a.scatter(
                    np.clip(x[k & out], lo, hi),
                    np.clip(y[k & out], lo, hi),
                    s=34,
                    marker="^",
                    facecolor="white",
                    edgecolor=color,
                    lw=0.9,
                )
        shown = [f for f in FIXATION_GROUPS if (group == f).any()]
        for k, fixation in enumerate(shown):
            a.text(
                0.97,
                0.5 - 0.055 * k,
                fixation,
                color=FIXATION_COLORS[fixation],
                transform=a.transAxes,
                ha="right",
                va="top",
                fontsize=7,
            )
        report["display_limit" + tag] = dict(
            limit=list(SCATTER_LIMIT),
            units_past=int(out.sum()),
            values=[[float(u), float(v)] for u, v in zip(x[out], y[out])],
        )
        a.plot(SCATTER_LIMIT, SCATTER_LIMIT, color="0.6", ls="--", lw=0.8)
        a.set_xlim(lo - 0.2, hi + 0.3)
        a.set_ylim(lo - 0.2, hi + 0.3)
        a.set_aspect("equal")
        for j, d in enumerate(data):
            mx, my = np.median(allp["in port"][j]), np.median(allp["after exit"][j])
            a.plot(
                mx,
                my,
                mouse_marker(d["subject"], j),
                color="white",
                mec="black",
                ms=10,
                mew=1.8,
                zorder=5,
            )
        a.set_xlabel("peak in port (÷ mean rate)")
        a.set_ylabel("peak after exit (÷ mean rate)")
        a.text(
            0,
            1.02,
            f"{len(x)} units, 2 mice · {int(out.sum())} outside the axes: "
            "open triangles at the edge",
            transform=a.transAxes,
            va="bottom",
            fontsize=7.5,
        )
        label_panels(ax, x=-0.18, y=1.04)
        fig.tight_layout()
        return save_figure(fig, OUT / name)


def plot_by_gap(data, report):
    """Median peak (IQR over units) by gap class, last in port against first after exit."""
    with plt.rc_context(FIGURE_STYLE):
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
        for j, d in enumerate(data):
            a = ax[j]
            rows = []
            for g in GAPS:
                p = peaks(d, d["byg"][g])
                rows.append(p)
                w = wilcoxon(p["after exit"], p["in port"])
                report.setdefault(d["subject"] + "_by_gap", {})[
                    f"{g}{'+' if g == GAPS[-1] else ''}"
                ] = dict(
                    pairs=int(len(d["byg"][g]["in port"])),
                    higher=int(np.sum(p["after exit"] > p["in port"])),
                    units=len(p["in port"]),
                    median_in_port=float(np.median(p["in port"])),
                    median_after_exit=float(np.median(p["after exit"])),
                    p=float(w.pvalue),
                )
            xs = np.arange(len(GAPS))
            for k, c, dx in (("in port", PORT, -0.08), ("after exit", EXIT, 0.08)):
                med = np.array([np.median(r[k]) for r in rows])
                q = np.array([np.percentile(r[k], [25, 75]) for r in rows])
                a.errorbar(
                    xs + dx,
                    med,
                    yerr=[med - q[:, 0], q[:, 1] - med],
                    fmt=mouse_marker(d["subject"], j) + "-",
                    color=c,
                    ms=5,
                    capsize=2,
                )
                a.text(
                    xs[-1] + 0.15, med[-1], NAMES[k], color=c, va="center", fontsize=7
                )
            a.axhline(0, color="0.6", ls=":", lw=0.8)
            a.set_xticks(xs, [f"{g}" for g in GAPS[:-1]] + [f"{GAPS[-1]}+"])
            a.set_xlim(-0.4, len(GAPS) - 0.1)
            a.set_xlabel("gap from the flash before (ms)")
            a.text(
                0,
                1.02,
                f"{d['subject']}: {len(d['exc'])} flash-excited units (median, IQR)",
                transform=a.transAxes,
                va="bottom",
            )
            if j == 0:
                a.set_ylabel("flash peak ÷ mean rate")
        label_panels(ax, x=-0.15, y=1.04)
        fig.tight_layout()
        return save_figure(fig, OUT / "exit_flash_by_gap")


def main() -> None:
    excited = load_excited_units()
    data = [session(s, set(excited[s])) for s in SESSIONS]
    report = {}
    for plot in (plot_examples, plot_population, plot_by_gap):
        print(plot(data, report))
    # The fixation-suppressed flash-excited units only (Anne, 2026-11-12).
    for d, s in zip(data, SESSIONS, strict=True):
        keep = fixation_suppressed(OUT / s)
        d["exc"] = np.array([u for u in d["exc"] if d["ids"][u] in keep])
        d["exc_label"] = "fixation-suppressed flash-excited units"
    print(
        plot_population(
            data,
            report,
            "exit_flash_population_fixation_suppressed",
            "_fixation_suppressed",
        )
    )
    (OUT / "exit_flash_pairs.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
