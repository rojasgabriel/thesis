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
  unit's mean observed rate over all valid bins; baseline = the in-port rate
  from -40 to 0 ms of the last in-port flash, the same for both periods; peak
  = the largest bin from 30 to 120 ms minus that baseline. Mean ± SEM over
  units, each mouse alone. The sampling unit is the unit.
- The final refit is fit to all trials, so this comparison is in-sample, not
  held out.

Writes `figures/glm/model_without_terms.svg`, `figures/glm/model_first_flash.svg`
and `results/glm/model_flash_peaks.json`.

    THESIS_CACHE_DIR=figures/cache uv run python -m \\
        thesis.ephys.analyses.glm.model_flash_peaks
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thesis.ephys.analyses.glm import cli, dmat
from thesis.ephys.analyses.glm import peaks as P
from thesis.ephys.analyses.glm.fit import build_unit_counts, load_unit_records
from thesis.ephys.analyses.glm.flash_kernels import SESSIONS
from thesis.ephys.analyses.glm.results import raw_contribution
from thesis.ephys.units import fetch_unit_table
from thesis.plotting import (
    FIGURE_STYLE,
    MODEL_COLOR,
    OBSERVED_COLOR,
    PERIOD_COLORS,
    label_panels,
    mouse_marker,
    save_figure,
)

OUT = Path("results/glm")
LAGS = P.LAGS_MS  # -40..119 ms
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
        obs = build_unit_counts(al, spk).reshape(grid)
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
        mean_rate = obs[valid.reshape(grid)].mean() / dmat.BINWIDTH_S
        sources = dict(
            zip(
                VERSIONS,
                (
                    obs,
                    np.exp(log_rate).reshape(grid),
                    np.exp(no_exit).reshape(grid),
                    np.exp(no_exit_movement).reshape(grid),
                ),
                strict=True,
            )
        )
        out.append(
            {
                name: {
                    k: src[r_[:, None], c_].mean(0) / dmat.BINWIDTH_S / mean_rate
                    for k, (r_, c_) in win.items()
                }
                for name, src in sources.items()
            }
        )
    return dict(subject=subject, units=out, n={k: len(v[0]) for k, v in conds.items()})


def binned(trace):
    """1-ms bins to `BIN`-ms bins."""
    return trace.reshape(-1, BIN).mean(-1)


CENTERS = binned(LAGS.astype(float))
BASE = (CENTERS >= -40) & (CENTERS < 0)
PEAK = (CENTERS >= 30) & (CENTERS < 120)


def peak(unit, version, flash):
    base = binned(unit[version]["last in port"])[BASE].mean()
    return (binned(unit[version][flash])[PEAK] - base).max()


def plot_peaks(data, summary):
    with plt.rc_context(FIGURE_STYLE):
        fig, ax = plt.subplots(1, 4, figsize=(12, 3.4), sharex=True, sharey=True)
        for a, version in zip(ax, VERSIONS, strict=True):
            color = OBSERVED_COLOR if version == "observed" else MODEL_COLOR
            for m, d in enumerate(data):
                s = d["subject"]
                x = np.array([peak(u, version, "last in port") for u in d["units"]])
                y = np.array([peak(u, version, "first after exit") for u in d["units"]])
                ex, ey = (v.std(ddof=1) / np.sqrt(len(v)) for v in (x, y))
                a.errorbar(
                    x.mean(),
                    y.mean(),
                    xerr=ex,
                    yerr=ey,
                    fmt=mouse_marker(s, m),
                    color=color,
                    ms=7,
                    capsize=3,
                )
                summary.setdefault(version, {})[s] = dict(
                    in_port=float(x.mean()),
                    after_exit=float(y.mean()),
                    increase=float(np.mean(y - x)),
                    units=len(x),
                )
                a.annotate(
                    s,
                    (x.mean(), y.mean()),
                    xytext=(6, -12 if m else 6),
                    textcoords="offset points",
                    fontsize=7,
                )
            lim = (0, 3.5)
            a.plot(lim, lim, color="0.6", ls="--", lw=0.8)
            a.set_xlim(lim)
            a.set_ylim(lim)
            a.set_aspect("equal")
            a.text(
                0,
                1.03,
                version,
                transform=a.transAxes,
                va="bottom",
                fontsize=8,
                color=color,
            )
            a.set_xlabel("peak, last in port")
        ax[0].set_ylabel("peak, first after exit\n(÷ mean rate)")
        label_panels(ax, x=-0.12, y=1.1)
        fig.tight_layout()
        return save_figure(fig, OUT / "model_without_terms")


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
    print(plot_peaks(data, summary))
    print(plot_traces(data))
    report = {"peaks": summary, "flashes": {d["subject"]: d["n"] for d in data}}
    (OUT / "model_flash_peaks.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
