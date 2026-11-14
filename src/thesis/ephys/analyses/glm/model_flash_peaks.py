"""Compare the GLM rate with the observed rate around the paired flashes.

Question: how much of the larger flash peak after the center exit does the
model predict, and which terms make that prediction? And which flashes does
the model follow, and which does it miss?

Comparison: the observed rate against the rate of the final refit of each unit
(all terms, observed spike history), on the same flashes:

- Peaks (`model_without_terms.svg`): the last in-port and the first
  after-exit flash of each trial (`peaks.paired_flashes`). Four versions: the
  observed rate; the full model; the model without the flash × after exit
  columns; the model also without the movement columns (turn direction and
  the video PCs). A version without terms subtracts their raw contribution
  (`results.raw_contribution`) from the log rate. Flash × after exit holds no
  movement column: it is the flash kernel switched on after the exit.
- Traces (`model_first_flash.svg`): the first flash of the trial (its window
  in the port), the last in-port flash and the first after-exit flash, the
  observed and the full-model rate.

- Sessions, trials and alignment: the GLM sessions (`flash_kernels.SESSIONS`)
  on the 1-ms grid of the fit. Units: the flash-excited units with a final fit
  (`peaks.load_excited_units`).
- Measure: per unit, the mean over flashes in 10-ms bins, divided by the
  unit's mean observed rate from 0.5 s before to 1.5 s after the first flash
  of each trial (`exit_flash_pairs.mean_rate`, the normalization of the
  observed peaks; until 2026-11-13: the mean over all valid bins); baseline = the in-port rate
  from -40 to 0 ms of the last in-port flash, the same for both periods; peak
  = the mean over the bins within 20 ms of the largest observed bin from 30
  to 120 ms (`peaks.window_peak`), minus that baseline. Every version is
  measured in the window of the observed peak, so the model is not measured
  at its own peak (until 2026-11-12: the largest bin of each version). Each
  light point is a unit; the open marker is the median of each mouse
  (`plotting.median_marker`; until 2026-11-13: mean ± SEM). Wilcoxon
  signed-rank (after exit against in port) within each mouse. Units past
  `exit_flash_pairs.SCATTER_LIMIT` are drawn at the edge as open triangles,
  stay in the statistics, and are listed in the report. The sampling unit is
  the unit.
- The final refit is fit to all trials, so this comparison is in-sample, not
  held out.

- By fixation group (`model_without_terms_by_fixation.svg`): the same peaks
  for the flash-excited units of each fixation group, one row each
  (`hold_effect.fixation_groups`: the fixation level below 0, above 0 or
  neither by more than 2 fold SDs).

Writes `figures/glm/model_without_terms.svg`, `figures/glm/model_first_flash.svg`,
`figures/glm/model_without_terms_by_fixation.svg`
and `results/glm/model_flash_peaks.json`.

    THESIS_CACHE_DIR=figures/cache uv run python -m \\
        thesis.ephys.analyses.glm.model_flash_peaks
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy.stats import wilcoxon

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thesis.ephys.analyses.glm import cli, dmat
from thesis.ephys.analyses.glm import peaks as P
from thesis.ephys.analyses.glm.exit_flash_pairs import SCATTER_LIMIT, mean_rate
from thesis.ephys.analyses.glm.fit import load_unit_records
from thesis.ephys.analyses.glm.flash_kernels import SESSIONS
from thesis.ephys.analyses.glm.hold_effect import FIXATION_GROUPS, fixation_groups
from thesis.ephys.analyses.glm.results import raw_contribution
from thesis.ephys.units import fetch_unit_table
from thesis.plotting import (
    FIGURE_STYLE,
    MODEL_COLOR,
    OBSERVED_COLOR,
    PERIOD_COLORS,
    POINT_ALPHA,
    label_panels,
    median_marker,
    mouse_marker,
    save_figure,
)

OUT = Path("results/glm")
LAGS = P.LAGS_MS  # -40..119 ms
EDGES_S = np.arange(LAGS[0], LAGS[-1] + 2) / 1000  # edges of the 1-ms lag bins
BIN = 10
VERSIONS = (
    "observed",
    "full model",
    "without flash × exit",
    "without flash × exit and movement",
)
FLASHES = ("first of trial", "last in port", "first after exit")


def first_flashes(trials):
    """Return the trials and times of the first flash with its window in the port."""
    rows, times = [], []
    for r, (f, en, ex) in enumerate(
        zip(
            trials["stim_pulse_times_s"],
            trials["center_entry_s"],
            trials["center_exit_s"],
            strict=True,
        )
    ):
        f = np.asarray(f, float)
        if f[0] - 0.04 >= en and f[0] + 0.12 <= ex:
            rows.append(r)
            times.append(f[0])
    return np.asarray(rows), np.asarray(times)


def session_traces(sess: str) -> dict:
    """Return, per flash-excited unit, the mean trace of each version and flash kind."""
    model = cli.PoissonGLM(root=OUT / sess)
    prepared = dmat._load_windows(model.windows)
    trials = dmat._load_selected_trials(prepared)
    al, rel = prepared["alignments"], prepared["relative_centers"]
    grid = (len(al), len(rel))
    valid = dmat._valid_bin_mask(prepared)
    rows, still, move = P.paired_flashes(trials)
    fr, ft = first_flashes(trials)
    # The first flash of every trial, for the normalization of `exit_flash_pairs`.
    first_of_each = np.array(
        [np.asarray(f, float)[0] for f in trials["stim_pulse_times_s"]]
    )
    conds = {
        "first of trial": (fr, ft),
        "last in port": (rows, still),
        "first after exit": (rows, move),
    }
    win = {}
    for k, (r_, t_) in conds.items():
        first = np.rint((t_ - al[r_] - rel[0]) / dmat.BINWIDTH_S).astype(int)
        win[k] = (r_, first[:, None] + LAGS[None, :])
    subject, session = (
        prepared["metadata"]["subject_name"],
        prepared["metadata"]["session_name"],
    )
    units = fetch_unit_table(
        subject,
        session,
        unit_criteria_id=1,
        stability_param_id=0,
        include_metrics=False,
    )
    excited = set(P.load_excited_units()[f"{subject}_{session}"])
    meta = json.loads(model.design.with_suffix(".json").read_text())
    names = meta["column_names"]

    def columns(prefix):
        return [i for i, n in enumerate(names) if n.startswith(prefix)]

    exit_columns = columns(f"{dmat.FLASH_AFTER_EXIT}_basis")
    movement = columns(f"{dmat.TURN_DIRECTION}_basis") + columns("video_pc")
    common = np.load(model.design, mmap_mode="r", allow_pickle=False)
    final = {
        int(it["unit_id"]): it
        for it in load_unit_records(
            model.fit_dir("all") / "checkpoints", "unit_*_final.json"
        )
    }
    out = []
    for uid, spk in zip(units["unit_id"], units["spike_times_s"], strict=True):
        if int(uid) not in final or int(uid) not in excited:
            continue
        rec = final[int(uid)]
        spk = np.asarray(spk, float)
        width = int(meta["base_columns"]) + int(rec["components"])
        w = np.asarray(rec["coefficients"], float)
        hist, _, _ = dmat.training_zscore(dmat.build_unit_history(al, spk), valid)
        log_rate = (
            float(rec["intercept"])
            + np.asarray(common[:, :width]) @ w[:width]
            + hist @ w[width:]
        )
        no_exit = log_rate - raw_contribution(common, w, meta, exit_columns)
        no_exit_movement = no_exit - raw_contribution(
            common, w, meta, [i for i in movement if i < width]
        )
        spk = np.sort(spk)
        rate = mean_rate(spk, first_of_each)
        # Observed: spikes in 1-ms bins that start at each flash, the counts of
        # `exit_flash_pairs.py`. The bins of the GLM grid are up to 0.5 ms off
        # the flash, which moves single spikes of sparse units between bins.
        observed = {
            k: np.diff(
                np.searchsorted(spk, t_[:, None] + EDGES_S[None, :]), axis=1
            ).mean(0)
            / dmat.BINWIDTH_S
            / rate
            for k, (_, t_) in conds.items()
        }
        sources = dict(
            zip(
                VERSIONS[1:],
                (
                    np.exp(log_rate).reshape(grid),
                    np.exp(no_exit).reshape(grid),
                    np.exp(no_exit_movement).reshape(grid),
                ),
                strict=True,
            )
        )
        out.append(
            {
                "id": int(uid),
                "observed": observed,
                **{
                    name: {
                        k: src[r_[:, None], c_].mean(0) / dmat.BINWIDTH_S / rate
                        for k, (r_, c_) in win.items()
                    }
                    for name, src in sources.items()
                },
            }
        )
    return dict(subject=subject, units=out, n={k: len(v[0]) for k, v in conds.items()})


def binned(trace):
    """1-ms bins to `BIN`-ms bins."""
    return trace.reshape(-1, BIN).mean(-1)


CENTERS = binned(LAGS.astype(float))
BASE = (CENTERS >= -40) & (CENTERS < 0)


def peak(unit, version, flash):
    """Window peak of a version, in the window of the observed peak of that flash."""
    base = binned(unit[version]["last in port"])[BASE].mean()
    trace = binned(unit[version][flash])
    observed = binned(unit["observed"][flash])
    return P.window_peak(trace, CENTERS, latency_from=observed) - base


def plot_peaks(rows, summary, name="model_without_terms"):
    """One row of the four versions per unit group; rows: [(label, data)].

    Each unit is a light point; the open marker is the median of each mouse.
    """
    lo, hi = SCATTER_LIMIT
    with plt.rc_context(FIGURE_STYLE):
        fig, axes = plt.subplots(
            len(rows),
            4,
            figsize=(10, 2.6 * len(rows) + 0.5),
            sharex=True,
            sharey=True,
            squeeze=False,
            layout="constrained",
        )
        for row, (label, data) in enumerate(rows):
            for a, version in zip(axes[row], VERSIONS, strict=True):
                color = OBSERVED_COLOR if version == "observed" else MODEL_COLOR
                for m, d in enumerate(data):
                    s = d["subject"]
                    x = np.array([peak(u, version, "last in port") for u in d["units"]])
                    y = np.array(
                        [peak(u, version, "first after exit") for u in d["units"]]
                    )
                    if not len(x):
                        continue
                    marker = mouse_marker(s, m)
                    out = (x < lo) | (x > hi) | (y < lo) | (y > hi)
                    a.scatter(
                        x[~out],
                        y[~out],
                        s=14,
                        marker=marker,
                        color=color,
                        alpha=POINT_ALPHA,
                        lw=0,
                    )
                    a.scatter(
                        np.clip(x[out], lo, hi),
                        np.clip(y[out], lo, hi),
                        s=22,
                        marker="^",
                        facecolor="white",
                        edgecolor=color,
                        lw=0.8,
                    )
                    median_marker(a, np.median(x), np.median(y), marker, color, 7)
                    summary.setdefault(label, {}).setdefault(version, {})[s] = dict(
                        units=len(x),
                        median_in_port=float(np.median(x)),
                        median_after_exit=float(np.median(y)),
                        median_increase=float(np.median(y - x)),
                        mean_increase=float(np.mean(y - x)),
                        higher=int(np.sum(y > x)),
                        p=float(wilcoxon(y, x).pvalue) if len(x) > 1 else None,
                        past_limit=[
                            [int(u["id"]), float(i), float(j)]
                            for u, i, j, o in zip(d["units"], x, y, out, strict=True)
                            if o
                        ],
                    )
                if row == 0:
                    a.text(
                        0,
                        1.12,
                        version,
                        transform=a.transAxes,
                        va="bottom",
                        fontsize=8,
                        color=color,
                    )
                if row == len(rows) - 1:
                    a.set_xlabel("peak, last in port")
            axes[row, 0].text(
                0,
                1.01,
                label,
                transform=axes[row, 0].transAxes,
                va="bottom",
                fontsize=8,
            ).set_in_layout(False)
            axes[row, 0].set_ylabel("peak, first after exit\n(÷ mean rate)")
        for a in axes.ravel():
            a.plot(SCATTER_LIMIT, SCATTER_LIMIT, color="0.6", ls="--", lw=0.8)
            a.set_xlim(lo - 0.2, hi + 0.3)
            a.set_ylim(lo - 0.2, hi + 0.3)
            a.set_aspect("equal")
        label_panels(axes[:, 0], x=-0.3, y=1.1)
        return save_figure(fig, OUT / name)


def plot_traces(data):
    with plt.rc_context(FIGURE_STYLE):
        fig, ax = plt.subplots(2, 3, figsize=(10, 5.2), sharex=True, sharey="row")
        for m, d in enumerate(data):
            for j, k in enumerate(FLASHES):
                a = ax[m, j]
                for version, col in (
                    ("observed", OBSERVED_COLOR),
                    ("full model", MODEL_COLOR),
                ):
                    tr = np.array([binned(u[version][k]) for u in d["units"]])
                    mu, se = tr.mean(0), tr.std(0, ddof=1) / np.sqrt(len(tr))
                    a.plot(CENTERS + BIN / 2, mu, color=col, lw=1.5)
                    a.fill_between(
                        CENTERS + BIN / 2, mu - se, mu + se, color=col, alpha=0.15, lw=0
                    )
                a.axvline(0, color="0.6", ls="--", lw=0.7)
                if m == 0:
                    period = "after exit" if k == "first after exit" else "in port"
                    a.set_title(k, fontsize=9, color=PERIOD_COLORS[period])
                if j == 0:
                    a.set_ylabel(f"{d['subject']}\nrate ÷ mean rate")
                if m == 1:
                    a.set_xlabel("time from flash (ms)")
                a.text(
                    0.98,
                    0.95,
                    f"{d['n'][k]} flashes",
                    transform=a.transAxes,
                    ha="right",
                    va="top",
                    fontsize=7,
                )
        ax[0, 0].text(
            0.03,
            0.95,
            "observed",
            color=OBSERVED_COLOR,
            transform=ax[0, 0].transAxes,
            va="top",
            fontsize=8,
        )
        ax[0, 0].text(
            0.03,
            0.85,
            "model",
            color=MODEL_COLOR,
            transform=ax[0, 0].transAxes,
            va="top",
            fontsize=8,
        )
        label_panels(ax.ravel(), x=-0.15, y=1.08)
        fig.tight_layout()
        return save_figure(fig, OUT / "model_first_flash")


def main() -> None:
    data = [session_traces(s) for s in SESSIONS]
    for d in data:
        print(d["subject"], len(d["units"]), d["n"], flush=True)
    summary = {}
    print(plot_peaks([("flash-excited units", data)], summary))
    print(plot_traces(data))
    # The same peaks in each fixation group (Anne, 2026-11-12; all groups 11-13).
    groups = [fixation_groups(OUT / s) for s in SESSIONS]
    rows = [
        (
            f"{name} units",
            [
                {**d, "units": [u for u in d["units"] if g.get(u["id"]) == name]}
                for d, g in zip(data, groups, strict=True)
            ],
        )
        for name in FIXATION_GROUPS
    ]
    by_group = {}
    print(plot_peaks(rows, by_group, "model_without_terms_by_fixation"))
    report = {
        "peaks": summary,
        "peaks_by_fixation_group": by_group,
        "flashes": {d["subject"]: d["n"] for d in data},
    }
    (OUT / "model_flash_peaks.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
