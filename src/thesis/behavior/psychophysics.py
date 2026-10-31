"""Which flashes do the mice use, when do they leave, and does the last trial count?

Question: the psychometric curves (`psychometric.py`) show that the choice
follows the flash rate. This module looks at how: (a) the weight of a flash on
the choice at each time in the train (the psychophysical kernel); (b) when the
mouse leaves the center port; (c) whether the side and the outcome of the last
trial change the choice.

- Sessions and trials (`behavior.trials`): by default the visual-only sessions with an
  ephys recording and at least 6 rates (no performance rule); with `--recorded` only the recorded session. Completed trials with a
  choice and a full 1-s train, boundary trials included. Each mouse is analyzed
  alone. The sampling unit is the trial, nested in sessions.
- Alignment: the first flash of the trial (time 0). The train has 25 slots of
  40 ms; the first slot always has a flash and the number of flashes equals the
  rate, so only where the other flashes fall differs between trials of a rate.
- (a) Kernel: a logistic regression of the choice (right = 1) on the number of
  flashes in each of `KERNEL_BINS` bins of 4 slots (slots 1 to 24, 40 to 1000
  ms), with one bias per session and no penalty. A weight is the change in the
  log-odds of a right choice for one more flash in that bin. A flat kernel means
  that the mouse counts every flash the same. The flashes go on after the exit,
  so the bins after the exit test whether the mouse uses flashes it sees while
  it moves. Flatness test: likelihood ratio against one weight for all bins
  (chi-square, `KERNEL_BINS` − 1 df; it treats trials as independent).
- (b) Timing: the center exit after the first flash (Bpod `t_react`), all used
  trials; the line is the median go cue. The axis ends at 1.2 s; the share of
  exits after it is in the json.
- (c) History: a logistic regression of the choice on the rate (Hz), the
  previous side after a rewarded trial, and the previous side after an
  unrewarded trial (±1, 0 when the condition is not true), with one bias per
  session; only trials right after a completed trial with a choice. A positive
  weight is a tendency to repeat the previous side.
- Error bars: 95% hierarchical bootstrap (sessions, then trials;
  `N_BOOTSTRAPS`); with one session, over trials.
- Each mouse is a line style (GRB006 solid, GRB064 dashed), one marker.

Writes `figures/behavior/psychophysics.svg` and `results/behavior/psychophysics.json`
(`psychophysics_recorded.*` with `--recorded`).

    THESIS_CACHE_DIR=figures/cache uv run python -m thesis.behavior.psychophysics
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thesis.behavior import psychometric
from thesis.behavior.trials import (
    SUBJECTS,
    hierarchical_samples,
    session_columns,
    task_trials,
)
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
HISTORY_NAMES = ("rate (per Hz)", "repeat after\nreward", "repeat after\nno reward")


def slot_counts(trials) -> np.ndarray:
    """Return the flashes in each 4-slot bin after the first slot, (trials, bins)."""
    counts = np.zeros((len(trials), KERNEL_BINS))
    for row, flashes in enumerate(trials["flashes_s"]):
        slots = np.round(flashes / SLOT_S).astype(int)
        if slots.max() >= SLOTS or np.abs(flashes / SLOT_S - slots).max() > 0.25:
            raise ValueError("A flash is off the 40-ms slot grid.")
        slots = slots[slots > 0]
        np.add.at(counts[row], (slots - 1) // 4, 1)
    return counts


def logistic(x: np.ndarray, y: np.ndarray, sessions) -> tuple[np.ndarray, float]:
    """Return the weights of `x` (one bias per session) and the log-likelihood."""
    design = np.column_stack([x, session_columns(sessions)])
    model = LogisticRegression(penalty=None, fit_intercept=False, max_iter=5000)
    model.fit(design, y)
    p = np.clip(model.predict_proba(design)[:, 1], 1e-12, 1 - 1e-12)
    loglik = float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return model.coef_[0][: x.shape[1]], loglik


def interval(x: np.ndarray, y: np.ndarray, sessions) -> np.ndarray:
    """Return the 2.5 and 97.5 percentiles of each weight, (weights, 2)."""
    draws = [
        logistic(x[rows], y[rows], labels)[0]
        for rows, labels in hierarchical_samples(sessions, N_BOOTSTRAPS)
        if 0 < y[rows].mean() < 1
    ]
    return np.percentile(draws, [2.5, 97.5], axis=0).T


def mouse_psychophysics(subject: str, recorded: bool) -> dict:
    trials = task_trials(subject, recorded)
    sessions = trials["session_name"].reset_index(drop=True)
    right = trials["right"].to_numpy()
    counts = slot_counts(trials)
    weights, loglik = logistic(counts, right, sessions)
    _, flat_loglik = logistic(counts.sum(1, keepdims=True), right, sessions)
    lr = 2 * (loglik - flat_loglik)
    exit_s = trials["exit_s"].to_numpy(float)

    after = trials["prev_response"].isin((-1, 1)).to_numpy()
    previous = trials["prev_response"].to_numpy(float)[after]
    rewarded = trials["prev_rewarded"].to_numpy(float)[after] == 1
    history_x = np.column_stack(
        [
            trials["rate_hz"].to_numpy(float)[after],
            np.where(rewarded, previous, 0.0),
            np.where(~rewarded, previous, 0.0),
        ]
    )
    history_sessions = sessions[after].reset_index(drop=True)
    history, _ = logistic(history_x, right[after], history_sessions)
    return {
        "subject": subject,
        "sessions": sorted(sessions.unique()),
        "trials": int(len(trials)),
        "kernel_bin_centers_ms": [
            1000 * SLOT_S * (4 * j + 2.5) for j in range(KERNEL_BINS)
        ],
        "kernel_weights": weights.tolist(),
        "kernel_weights_95": interval(counts, right, sessions).tolist(),
        "kernel_flat_lr_p": float(chi2.sf(lr, KERNEL_BINS - 1)),
        "exit_after_first_flash_s": exit_s.tolist(),
        "median_exit_s": float(np.median(exit_s)),
        "fraction_exit_after_1_2_s": float(np.mean(exit_s > 1.2)),
        "median_go_cue_s": float(np.nanmedian(trials["go_cue_s"])),
        "history_trials": int(after.sum()),
        "history_weights": history.tolist(),
        "history_weights_95": interval(
            history_x, right[after], history_sessions
        ).tolist(),
    }


def plot(report: list[dict], name: str, choices: list[dict] | None = None) -> Path:
    with plt.rc_context(FIGURE_STYLE):
        if choices is None:
            figure, axes = plt.subplots(
                1, 3, figsize=(9.6, 2.9), layout="constrained", width_ratios=(1.3, 1, 1)
            )
            kernel, timing, history = axes
        else:
            # With the psychometric curves: a 2 x 2 figure (for one slide).
            figure, axes = plt.subplots(2, 2, figsize=(7.4, 5.2), layout="constrained")
            psychometric.draw(axes[0, 0], choices)
            kernel, timing, history = axes[0, 1], axes[1, 0], axes[1, 1]
            axes = axes.ravel()
        kernel.axhline(0, color=REFERENCE_COLOR, lw=0.6)
        history.axhline(0, color=REFERENCE_COLOR, lw=0.6)
        bins = np.linspace(0.4, 1.2, 33)
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
            xlim=(0.4, 1.2),
        )
        history.set_xticks((0, 1), HISTORY_NAMES[1:])
        history.set(
            xlim=(-0.5, 1.5), ylabel="weight on repeating\nthe previous side (log-odds)"
        )
        label_panels(axes)
        return save_figure(figure, OUT / name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--recorded", action="store_true", help="the recorded session only"
    )
    args = parser.parse_args()
    name = "psychophysics_recorded" if args.recorded else "psychophysics"
    report = [mouse_psychophysics(subject, args.recorded) for subject in SUBJECTS]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(report, indent=1))
    # The recorded sessions get one figure with their psychometric curves.
    choices = (
        json.loads((OUT / "psychometric_recorded.json").read_text())
        if args.recorded
        else None
    )
    print(plot(report, name, choices))
    for data in report:
        print(
            json.dumps(
                {
                    k: v
                    for k, v in data.items()
                    if k not in ("exit_after_first_flash_s", "sessions")
                }
            )
        )


if __name__ == "__main__":
    main()
