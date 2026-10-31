"""Which flashes do the mice use, when do they leave, and does the last trial count?

Question: the psychometric curves (`psychometric.py`) show that the choice
follows the flash rate. This module looks at how: (a) the weight of a flash on
the choice at each time in the train (the psychophysical kernel); (b) when the
mouse leaves the center port; (c) whether the side and the outcome of the last
trial change the choice.

- Sessions and trials: as in `psychometric.py` (`SESSIONS`): completed trials
  with a left or right choice, boundary trials included. One session per
  mouse; each mouse is analyzed alone. The sampling unit is the trial.
- Alignment: the first flash of the trial (time 0). The train has 25 slots of
  40 ms; the first slot always has a flash and the number of flashes equals the
  rate, so only where the other flashes fall differs between trials of a rate.
- (a) Kernel: a logistic regression of the choice (right = 1) on the number of
  flashes in each of `KERNEL_BINS` bins of 4 slots (slots 1 to 24, 40 to 1000
  ms), with an intercept and no penalty. A weight is the change in the log-odds
  of a right choice for one more flash in that bin. A flat kernel means that
  the mouse counts every flash the same. The flashes go on after the exit, so
  the bins after the exit test whether the mouse uses flashes it sees while it
  moves. Flatness test: likelihood ratio against one weight for all bins
  (chi-square, `KERNEL_BINS` − 1 df). Error bars: 95% percentile interval over
  `N_BOOTSTRAPS` resamples of trials.
- (b) Timing: the center exit after the first flash, all completed trials;
  the line is the median go cue.
- (c) History: a logistic regression of the choice on the rate (Hz), the
  previous side after a rewarded trial, and the previous side after an
  unrewarded trial (±1, 0 when the condition is not true), with an intercept.
  Only trials after a completed trial with a choice. A positive weight is a
  tendency to repeat the previous side. Error bars: as in (a).

The script writes `figures/behavior/psychophysics.svg` and
`results/behavior/psychophysics.json`.

    THESIS_CACHE_DIR=figures/cache uv run python -m thesis.behavior.psychophysics
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thesis.behavior.psychometric import RIGHT, SESSIONS
from thesis.ephys.trials import build_trial_table
from thesis.plotting import (
    FIGURE_STYLE,
    OBSERVED_COLOR,
    REFERENCE_COLOR,
    label_panels,
    save_figure,
)

OUT = Path("results/behavior")
SLOT_S = 0.04
SLOTS = 25
KERNEL_BINS = 6  # bins of 4 slots over slots 1-24
N_BOOTSTRAPS = 1000
SEED = 0
HISTORY_NAMES = ("rate (per Hz)", "repeat after\nreward", "repeat after\nno reward")


def completed_trials(session: str):
    subject, stamp = session.split("_", 1)
    trials = build_trial_table(subject, stamp, include_frames=False)
    return trials[
        (trials["early_withdrawal"] == 0) & trials["response"].isin((-1, 1))
    ].copy()


def slot_counts(trials) -> np.ndarray:
    """Return the flashes in each 4-slot bin after the first slot, (trials, bins)."""
    counts = np.zeros((len(trials), KERNEL_BINS))
    for row, pulses in enumerate(trials["stim_pulse_times_s"]):
        pulses = np.asarray(pulses, dtype=float)
        slots = np.round((pulses - pulses[0]) / SLOT_S).astype(int)
        if (
            slots.max() >= SLOTS
            or np.abs((pulses - pulses[0]) / SLOT_S - slots).max() > 0.25
        ):
            raise ValueError("A flash is off the 40-ms slot grid.")
        slots = slots[slots > 0]
        np.add.at(counts[row], (slots - 1) // 4, 1)
    return counts


def logistic(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Return the weights (intercept first) and the log-likelihood."""
    model = LogisticRegression(penalty=None, max_iter=5000).fit(x, y)
    p = np.clip(model.predict_proba(x)[:, 1], 1e-12, 1 - 1e-12)
    loglik = float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return np.r_[model.intercept_, model.coef_[0]], loglik


def bootstrap(x: np.ndarray, y: np.ndarray, rng) -> np.ndarray:
    """Return the 2.5 and 97.5 percentiles of each weight (no intercept), (2, weights)."""
    draws = []
    for _ in range(N_BOOTSTRAPS):
        pick = rng.integers(0, len(y), len(y))
        if y[pick].min() == y[pick].max():
            continue
        draws.append(logistic(x[pick], y[pick])[0][1:])
    return np.percentile(draws, [2.5, 97.5], axis=0)


def session_psychophysics(session: str) -> dict:
    rng = np.random.default_rng(SEED)
    trials = completed_trials(session)
    right = (trials["response"] == RIGHT).to_numpy(int)
    counts = slot_counts(trials)
    weights, loglik = logistic(counts, right)
    _, flat_loglik = logistic(counts.sum(1, keepdims=True), right)
    lr = 2 * (loglik - flat_loglik)

    first = trials["stim_pulse_times_s"].map(lambda p: p[0]).to_numpy(float)
    exit_s = trials["center_exit_s"].to_numpy(float) - first
    go = np.asarray(
        [
            g[0] - f if len(g) else np.nan
            for g, f in zip(trials["go_cue_times_s"], first)
        ]
    )

    after = trials["prev_response"].isin((-1, 1)).to_numpy()
    previous = trials["prev_response"].to_numpy(float)[after]
    rewarded = trials["prev_rewarded"].to_numpy(float)[after] == 1
    history_x = np.column_stack(
        [
            trials["visual_stim_rate_hz"].to_numpy(float)[after],
            np.where(rewarded, previous, 0.0),
            np.where(~rewarded, previous, 0.0),
        ]
    )
    history_y = right[after]
    history, _ = logistic(history_x, history_y)
    return {
        "subject": session.split("_")[0],
        "session": session.split("_", 1)[1],
        "trials": int(len(trials)),
        "kernel_bin_centers_ms": [
            1000 * SLOT_S * (4 * j + 2.5) for j in range(KERNEL_BINS)
        ],
        "kernel_weights": weights[1:].tolist(),
        "kernel_weights_95": bootstrap(counts, right, rng).T.tolist(),
        "kernel_flat_lr_p": float(chi2.sf(lr, KERNEL_BINS - 1)),
        "exit_after_first_flash_s": exit_s.tolist(),
        "median_exit_s": float(np.median(exit_s)),
        "median_go_cue_s": float(np.nanmedian(go)),
        "history_trials": int(after.sum()),
        "history_weights": history[1:].tolist(),
        "history_weights_95": bootstrap(history_x, history_y, rng).T.tolist(),
    }


def plot(report: list[dict]) -> Path:
    with plt.rc_context(FIGURE_STYLE):
        figure, axes = plt.subplots(
            1, 3, figsize=(9.6, 2.9), layout="constrained", width_ratios=(1.3, 1, 1)
        )
        kernel, timing, history = axes
        kernel.axhline(0, color=REFERENCE_COLOR, lw=0.6)
        history.axhline(0, color=REFERENCE_COLOR, lw=0.6)
        bins = np.linspace(0.4, 1, 25)
        for row, data in enumerate(report):
            marker = "o"  # the mouse is the line style only
            line = ("-", "--")[row % 2]
            shift = 12 * (row - 0.5)  # drawing only, so the bars do not overlap
            x = np.asarray(data["kernel_bin_centers_ms"]) + shift
            w = np.asarray(data["kernel_weights"])
            ci = np.asarray(data["kernel_weights_95"]).T
            kernel.errorbar(
                x,
                w,
                yerr=[w - ci[0], ci[1] - w],
                fmt=marker,
                ls=line,
                color=OBSERVED_COLOR,
                mfc="white",
                ms=4,
                capsize=2,
                lw=1,
            )
            kernel.axvline(
                1000 * data["median_exit_s"], color=REFERENCE_COLOR, ls=line, lw=0.8
            )
            kernel.text(
                1000 * data["median_exit_s"] - 6,
                0.03,
                f"exit, {data['subject']}",
                color=REFERENCE_COLOR,
                fontsize=6.5,
                rotation=90,
                ha="right",
                va="bottom",
                transform=kernel.get_xaxis_transform(),
            )
            timing.hist(
                data["exit_after_first_flash_s"],
                bins=bins,
                density=True,
                histtype="step",
                color=OBSERVED_COLOR,
                ls=line,
                lw=1.1,
            )
            timing.axvline(
                data["median_go_cue_s"], color=REFERENCE_COLOR, ls=line, lw=0.8
            )
            h = np.asarray(data["history_weights"])[1:]
            hci = np.asarray(data["history_weights_95"]).T[:, 1:]
            hx = np.arange(2) + 0.12 * (row - 0.5)
            bars = history.errorbar(
                hx,
                h,
                yerr=[h - hci[0], hci[1] - h],
                fmt=marker,
                ls="none",
                color=OBSERVED_COLOR,
                mfc="white",
                ms=4,
                capsize=2,
                lw=1,
            )
            bars[2][0].set_linestyle(line)  # the mouse is the line style of the bar
            # The mouse's name next to a sample of its marker and line.
            y = 0.95 - 0.09 * row
            kernel.plot(
                [0.03, 0.11],
                [y, y],
                color=OBSERVED_COLOR,
                ls=line,
                lw=1.1,
                transform=kernel.transAxes,
            )
            kernel.plot(
                0.07,
                y,
                marker,
                color=OBSERVED_COLOR,
                mfc="white",
                ms=4,
                transform=kernel.transAxes,
            )
            kernel.text(
                0.13,
                y,
                data["subject"],
                va="center",
                fontsize=7,
                transform=kernel.transAxes,
            )
        kernel.set(
            xlabel="time of the flash from the first flash (ms)",
            ylabel="weight on a right choice\n(log-odds per flash)",
            xlim=(0, 1000),
        )
        timing.text(
            min(d["median_go_cue_s"] for d in report) - 0.005,
            0.97,
            "go cue",
            color=REFERENCE_COLOR,
            fontsize=6.5,
            rotation=90,
            ha="right",
            va="top",
            transform=timing.get_xaxis_transform(),
        )
        timing.set(
            xlabel="center exit from the first flash (s)",
            ylabel="trials (density)",
            xlim=(0.4, 1),
        )
        history.set_xticks((0, 1), HISTORY_NAMES[1:])
        history.set(
            xlim=(-0.5, 1.5), ylabel="weight on repeating\nthe previous side (log-odds)"
        )
        label_panels(axes)
        return save_figure(figure, OUT / "psychophysics")


def main() -> None:
    report = [session_psychophysics(session) for session in SESSIONS]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "psychophysics.json").write_text(json.dumps(report, indent=1))
    print(plot(report))
    for data in report:
        print(
            json.dumps(
                {k: v for k, v in data.items() if k != "exit_after_first_flash_s"}
            )
        )


if __name__ == "__main__":
    main()
