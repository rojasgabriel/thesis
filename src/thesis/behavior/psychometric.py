"""Do the mice use the flash rate to choose a side?

Question: in the task, a train of flashes plays while the mouse holds its nose
in the center port. A rate above the category boundary is rewarded at the
right port, a rate below it at the left port. Does the choice follow the rate?

Comparison: the fraction of right choices at each flash rate, in each mouse.
No model of the neurons.

- Sessions and trials (`behavior.trials`): by default all task sessions of each
  mouse (visual-only discrimination, at least 6 rates); with `--recorded` only
  the recorded session (the one the neural analyses use). Completed trials with
  a left or right choice and a full 1-s train; boundary trials (rewarded at
  random) are shown.
- Points: the fraction of right choices at each rate, pooled over sessions,
  with a 95% hierarchical-bootstrap interval (sessions, then trials;
  `N_BOOTSTRAPS`). With one session this is a bootstrap over trials. The
  sampling unit is the trial, nested in sessions.
- Curve: a logistic fit of the choice on the rate, pooled (no session terms,
  no penalty, no lapse terms).
- Accuracy: the fraction rewarded of the non-boundary trials.
- x axis: stimulus strength (`trials.stimulus_strength`), (rate − boundary) / 8 Hz:
  0 at the boundary, −1 at 4 and +1 at 20 flashes/s. The fit is in Hz.
- Each mouse is a line style (GRB006 solid, GRB064 dashed), one marker.

Writes `figures/behavior/psychometric.svg` and `results/behavior/psychometric.json`
(`psychometric_recorded.*` with `--recorded`).

    THESIS_CACHE_DIR=figures/cache uv run python -m thesis.behavior.psychometric
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy.optimize import minimize

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thesis.behavior.trials import SUBJECTS, hierarchical_samples, task_trials
from thesis.ephys.trials import stimulus_strength
from thesis.plotting import (
    FIGURE_STYLE,
    OBSERVED_COLOR,
    REFERENCE_COLOR,
    save_figure,
)

OUT = Path("results/behavior")
N_BOOTSTRAPS = 1000
LINES = ("-", "--")  # the mice, in `SUBJECTS` order


def logistic_fit(rates: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    """Return (intercept, slope per Hz) of P(right) = 1 / (1 + exp(-(a + b rate)))."""

    def loss(params):
        z = params[0] + params[1] * rates
        return np.sum(np.logaddexp(0, z) - right * z)

    return tuple(minimize(loss, x0=(0.0, 0.0), method="BFGS").x)


def mouse_behavior(subject: str, recorded: bool) -> dict:
    trials = task_trials(subject, recorded)
    rates = trials["rate_hz"].to_numpy()
    right = trials["right"].to_numpy()
    levels = np.unique(rates)

    def fractions(rows):
        return np.array([right[rows][rates[rows] == r].mean() for r in levels])

    draws = np.array(
        [
            fractions(rows)
            for rows, _ in hierarchical_samples(trials["session_name"], N_BOOTSTRAPS)
        ]
    )
    a, b = logistic_fit(rates, right)
    clear = rates != 12
    return {
        "subject": subject,
        "sessions": sorted(trials["session_name"].unique()),
        "trials": int(len(trials)),
        "rates_hz": levels.tolist(),
        "trials_per_rate": [int((rates == r).sum()) for r in levels],
        "p_right": fractions(np.arange(len(trials))).tolist(),
        "p_right_95": np.nanpercentile(draws, [2.5, 97.5], axis=0).T.tolist(),
        "logistic_intercept": float(a),
        "logistic_slope_per_hz": float(b),
        "p_right_at_boundary": float(1 / (1 + np.exp(-(a + b * 12)))),
        "accuracy_non_boundary": float(trials.loc[clear, "rewarded"].mean()),
    }


def draw(axis, report: list[dict]) -> None:
    """Draw the psychometric curves of the mice on one axis."""
    axis.axvline(0, color=REFERENCE_COLOR, ls="--", lw=0.8)
    axis.axhline(0.5, color=REFERENCE_COLOR, lw=0.5)
    grid = np.linspace(3, 21, 200)
    for row, data in enumerate(report):
        line = LINES[row]
        rates = np.asarray(data["rates_hz"])
        p = np.asarray(data["p_right"])
        interval = np.asarray(data["p_right_95"]).T
        # A small shift per mouse so the points do not overlap (drawing only).
        bars = axis.errorbar(
            stimulus_strength(rates) + 0.05 * (row - 0.5),
            p,
            yerr=[p - interval[0], interval[1] - p],
            fmt="o",
            color=OBSERVED_COLOR,
            mfc="white",
            ms=4,
            capsize=2,
            lw=1,
        )
        bars[2][0].set_linestyle(line)
        a, b = data["logistic_intercept"], data["logistic_slope_per_hz"]
        axis.plot(
            stimulus_strength(grid),
            1 / (1 + np.exp(-(a + b * grid))),
            color=OBSERVED_COLOR,
            lw=1.2,
            ls=line,
        )
        y = 0.95 - 0.08 * row
        axis.plot(
            [0.02, 0.09],
            [y, y],
            color=OBSERVED_COLOR,
            lw=1.2,
            ls=line,
            transform=axis.transAxes,
        )
        sessions = len(data["sessions"])
        axis.text(
            0.11,
            y,
            f"{data['subject']}: {100 * data['accuracy_non_boundary']:.0f}% correct"
            + (f", {sessions} sessions" if sessions > 1 else ""),
            color=OBSERVED_COLOR,
            transform=axis.transAxes,
            va="center",
            fontsize=7,
        )
    axis.text(0.04, 0.03, "boundary", color=REFERENCE_COLOR, fontsize=7)
    axis.set(
        xlabel="stimulus strength (left − / right +)",
        ylabel="fraction of right choices",
        ylim=(0, 1),
        xlim=tuple(stimulus_strength((3, 21))),
        xticks=(-1, -0.5, 0, 0.5, 1),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--recorded", action="store_true", help="the recorded session only"
    )
    args = parser.parse_args()
    name = "psychometric_recorded" if args.recorded else "psychometric"
    report = [mouse_behavior(subject, args.recorded) for subject in SUBJECTS]
    with plt.rc_context(FIGURE_STYLE):
        figure, axis = plt.subplots(figsize=(3.6, 3.0), layout="constrained")
        draw(axis, report)
        print(save_figure(figure, OUT / name))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(report, indent=1))
    for data in report:
        print(
            data["subject"],
            len(data["sessions"]),
            "sessions",
            data["trials"],
            "trials",
            f"{data['accuracy_non_boundary']:.3f}",
        )


if __name__ == "__main__":
    main()
