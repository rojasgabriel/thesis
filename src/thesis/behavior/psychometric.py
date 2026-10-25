"""Do the mice use the flash rate to choose a side?

Question: in the task, a train of flashes plays while the mouse holds its nose
in the center port. A rate above the category boundary is rewarded at the
right port, a rate below it at the left port. Does the choice follow the rate?

Comparison: the fraction of right choices at each flash rate, in each mouse
(one session each, `SESSIONS`). No model of the neurons.

- Sessions: visual-only sessions with sorted units and the full range of
  rates. GRB006: the GLM session (11 rates). GRB064: 20261008_143359 (9 rates,
  180 completed trials); its GLM session (20260916_132939) has only the four
  easy rates 4, 6, 18 and 20 per s. GRB006 20240723_142451 behaves better but
  is a visual + auditory session.

- Trials: completed trials (no early withdrawal) with a left or right choice.
  Boundary trials (rate = boundary) are shown; they are rewarded at random.
- Points: the fraction of right choices at each rate, with a 95% Wilson
  interval. The sampling unit is the trial; each mouse is one session.
- Curve: a logistic fit of the choice on the rate (no penalty, no lapse
  terms). Bias = P(right) at the boundary.
- Accuracy: the fraction rewarded of the non-boundary trials.

The script writes `figures/behavior/psychometric.svg` and
`results/behavior/psychometric.json`.

    THESIS_CACHE_DIR=figures/cache uv run python -m thesis.behavior.psychometric
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thesis.ephys.analyses.stim_responses_by_locomotion import SUBJECT_COLORS
from thesis.ephys.trials import build_trial_table
from thesis.plotting import FIGURE_STYLE, REFERENCE_COLOR, save_figure

OUT = Path("results/behavior")
RIGHT = 1  # the response code of a right choice (left is -1)
SESSIONS = ("GRB006_20240821_121447", "GRB064_20261008_143359")


def wilson(k: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Return the 95% Wilson interval of k successes in n trials, (2, rates)."""
    z = norm.ppf(0.975)
    p = k / n
    center = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return np.stack([center - half, center + half])


def logistic_fit(rates: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    """Return (intercept, slope per Hz) of P(right) = 1 / (1 + exp(-(a + b rate)))."""

    def loss(params):
        z = params[0] + params[1] * rates
        return np.sum(np.logaddexp(0, z) - right * z)

    return tuple(minimize(loss, x0=(0.0, 0.0), method="BFGS").x)


def session_behavior(session: str) -> dict:
    subject, stamp = session.split("_", 1)
    trials = build_trial_table(subject, stamp, include_frames=False)
    done = trials[(trials["early_withdrawal"] == 0) & trials["response"].isin((-1, 1))]
    rates = done["visual_stim_rate_hz"].to_numpy(float)
    right = (done["response"] == RIGHT).to_numpy(int)
    boundary = float(done["category_boundary_hz"].iloc[0])
    levels = np.unique(rates)
    k = np.array([right[rates == r].sum() for r in levels])
    n = np.array([(rates == r).sum() for r in levels])
    a, b = logistic_fit(rates, right)
    clear = done["stim_category"] != "boundary"
    return {
        "subject": subject,
        "session": stamp,
        "trials": int(len(done)),
        "boundary_hz": boundary,
        "rates_hz": levels.tolist(),
        "trials_per_rate": n.tolist(),
        "p_right": (k / n).tolist(),
        "p_right_95": wilson(k, n).T.tolist(),
        "logistic_intercept": float(a),
        "logistic_slope_per_hz": float(b),
        "p_right_at_boundary": float(1 / (1 + np.exp(-(a + b * boundary)))),
        "accuracy_non_boundary": float(done.loc[clear, "rewarded"].mean()),
    }


def main() -> None:
    report = {session: session_behavior(session) for session in SESSIONS}
    colors = dict(zip(("GRB006", "GRB064"), SUBJECT_COLORS, strict=False))
    with plt.rc_context(FIGURE_STYLE):
        figure, axis = plt.subplots(figsize=(3.6, 3.0), layout="constrained")
        axis.axvline(12, color=REFERENCE_COLOR, ls="--", lw=0.8)
        axis.axhline(0.5, color=REFERENCE_COLOR, lw=0.5)
        grid = np.linspace(3, 21, 200)
        for row, data in enumerate(report.values()):
            color = colors[data["subject"]]
            rates = np.asarray(data["rates_hz"])
            p = np.asarray(data["p_right"])
            interval = np.asarray(data["p_right_95"]).T
            # A small shift per mouse so the points do not overlap (drawing only).
            axis.errorbar(
                rates + 0.4 * (row - 0.5),
                p,
                yerr=[p - interval[0], interval[1] - p],
                fmt="o",
                color=color,
                ms=4,
                capsize=2,
                lw=1,
            )
            a, b = data["logistic_intercept"], data["logistic_slope_per_hz"]
            axis.plot(grid, 1 / (1 + np.exp(-(a + b * grid))), color=color, lw=1.2)
            axis.text(
                0.03,
                0.97 - 0.09 * row,
                f"{data['subject']}: {100 * data['accuracy_non_boundary']:.0f}% correct",
                color=color,
                transform=axis.transAxes,
                va="top",
            )
        axis.text(12.3, 0.03, "boundary", color=REFERENCE_COLOR, fontsize=7)
        axis.set(
            xlabel="flash rate (flashes/s)",
            ylabel="fraction of right choices",
            ylim=(0, 1),
            xlim=(3, 21),
        )
        print(save_figure(figure, OUT / "psychometric"))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "psychometric.json").write_text(json.dumps(report, indent=1))
    print(
        json.dumps(
            {
                s: {k: v for k, v in r.items() if not isinstance(v, list)}
                for s, r in report.items()
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
