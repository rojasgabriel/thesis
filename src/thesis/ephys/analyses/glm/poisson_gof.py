"""Test whether held-out GLM spike counts follow the predicted conditional Poisson law.

Comparison: recorded 1-ms counts against the Poisson distribution with the
out-of-fold expected count of each bin, which includes strictly past observed
spike history. No new fits. Predictions, units, trials, first-flash alignment
and pre-response validity are those of conditional_dispersion.unit_predictions:
quality criterion 1 / stability set 0 units, fit.py exclusions, completed
left/right trials, bins up to response entry. Folds with zero test spikes have
no saved model and are not scored. No baseline subtraction.

Checks, per unit:
- Time rescaling with the discrete-time correction (Haslinger et al. 2010) on
  the event "bin has >= 1 spike". Rescaled time accumulates across all scored
  bins of a unit, so intervals cross trial gaps; only the one final censored
  interval is dropped (restarting each trial biases intervals short). Report
  KS distance / (1.36 / sqrt(n intervals)): > 1 rejects at 5 %.
- Serial dependence: lag-1 correlation of normal scores of successive rescaled
  intervals, as a z score (r * sqrt(n)).
- Count distribution beyond P(spike): observed bins with >= 2 spikes against
  the Poisson prediction sum(1 - e^-mu - mu e^-mu), as a z score. Refractory
  single units fall below it; merged multi-units can sit above it.
- Randomized PIT u = F(y-1) + v P(y); KS distance per predicted-rate group
  (< 10, 10-30, > 30 Hz). At 1 ms almost all mass is y = 0, so this has little
  power beyond the checks above; it stays in the JSON only.
Per session, pooled over units: sum of (y - mu) / sqrt(sum mu) in 10-ms blocks
of time from first flash; ~N(0, 1) per block if rates are calibrated in time.

Rejection means the fitted conditional Poisson model is wrong; it does not
separate a wrong rate from a wrong noise family. Sampling unit: unit, sessions
reported separately. Missing sessions are explicit.

Run: uv run python -m thesis.ephys.analyses.glm.poisson_gof \\
    --data-root <checkout containing saved GLM fits and spike cache>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

from thesis.behavior.trials import SESSIONS
from thesis.ephys.analyses.glm.conditional_dispersion import unit_predictions

RATE_EDGES_HZ = (0, 10, 30, np.inf)
BLOCK_BINS = 10


def rescaled_intervals(y, mu, rng):
    """Discrete-time-corrected rescaled ISIs (uniform under the model)."""
    spike = y > 0
    r = rng.uniform(size=len(y))
    step = np.where(spike, -np.log1p(r * np.expm1(-mu)), mu)
    total = np.cumsum(step)[spike]
    return -np.expm1(-np.diff(total, prepend=0.0))


def ks_ratio(z):
    return stats.kstest(z, "uniform").statistic * np.sqrt(len(z)) / 1.36


def self_check():
    rng = np.random.default_rng(5)
    trials, bins = 3000, 300
    y = np.zeros((trials, bins))
    mu = np.empty_like(y)
    for t in range(bins):  # 3 Hz with refractoriness and a rebound
        recent = y[:, max(t - 3, 0) : t].sum(axis=1)
        rebound = y[:, max(t - 20, 0) : max(t - 3, 0)].sum(axis=1)
        mu[:, t] = 0.003 * np.exp(-3 * recent + 0.8 * rebound)
        y[:, t] = rng.poisson(mu[:, t])
    z = rescaled_intervals(y.ravel(), mu.ravel(), rng)
    assert ks_ratio(z) < 1, ks_ratio(z)
    restarted = np.concatenate([rescaled_intervals(a, b, rng) for a, b in zip(y, mu)])
    assert ks_ratio(restarted) > 3, "per-trial restart should show censoring bias"
    assert ks_ratio(rescaled_intervals(y.ravel(), 1.5 * mu.ravel(), rng)) > 3


def check_unit(unit):
    y, mu, scored = (
        unit["y"][unit["scored"]],
        unit["mu"][unit["scored"]],
        unit["scored"],
    )
    rng = np.random.default_rng([20261118, unit["unit_id"]])
    z = rescaled_intervals(y, mu, rng)
    g = stats.norm.ppf(np.clip(z, 1e-12, 1 - 1e-12))
    lag1 = float(np.corrcoef(g[:-1], g[1:])[0, 1]) if len(g) > 3 else np.nan
    p2 = -np.expm1(-mu) - mu * np.exp(-mu)
    u = stats.poisson.cdf(y - 1, mu) + rng.uniform(size=len(y)) * stats.poisson.pmf(
        y, mu
    )
    group = np.digitize(mu * 1000, RATE_EDGES_HZ[1:-1])
    pit = [
        [
            float(stats.kstest(u[group == g_], "uniform").statistic),
            int((group == g_).sum()),
        ]
        if (group == g_).any()
        else [None, 0]
        for g_ in range(len(RATE_EDGES_HZ) - 1)
    ]
    residual = np.where(scored, unit["y"] - unit["mu"], 0).sum(axis=0)
    expected = np.where(scored, unit["mu"], 0).sum(axis=0)
    return (
        dict(
            unit_id=unit["unit_id"],
            intervals=len(z),
            ks_ratio=float(ks_ratio(z)),
            ks_p=float(stats.kstest(z, "uniform").pvalue),
            lag1_r=lag1,
            lag1_z=lag1 * np.sqrt(len(g)),
            multi_observed=int((y >= 2).sum()),
            multi_predicted=float(p2.sum()),
            multi_z=float(((y >= 2).sum() - p2.sum()) / np.sqrt((p2 * (1 - p2)).sum())),
            pit_ks_by_rate=pit,
            max_predicted_hz=float(mu.max() * 1000),
            omitted_folds=unit["omitted_folds"],
        ),
        residual,
        expected,
    )


def session(data_root, name):
    units, residual, expected, centers = [], 0, 0, None
    for unit in unit_predictions(data_root, name):
        result, r, e = check_unit(unit)
        units.append(result)
        residual, expected, centers = residual + r, expected + e, unit["centers_s"]
    stop = len(centers) // BLOCK_BINS * BLOCK_BINS

    def blocks(a):
        return a[:stop].reshape(-1, BLOCK_BINS).sum(axis=1)

    time = dict(
        time_s=centers[:stop].reshape(-1, BLOCK_BINS).mean(axis=1).tolist(),
        z=(blocks(residual) / np.sqrt(blocks(expected))).tolist(),
    )
    ratio = np.array([u["ks_ratio"] for u in units])
    summary = dict(
        session=name,
        units=len(units),
        ks_reject=int((ratio > 1).sum()),
        ks_ratio_median=float(np.median(ratio)),
        lag1_reject=int((np.abs([u["lag1_z"] for u in units]) > 1.96).sum()),
        multi_z_median=float(np.median([u["multi_z"] for u in units])),
    )
    print(json.dumps(summary), flush=True)
    return dict(**summary, time=time, unit_results=units)


def plot_report(results, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    markers = dict(GRB006="o", GRB064="s", GRB061="^", GRB068="D")
    fig, axes = plt.subplots(2, 2, figsize=(8, 6.4), layout="constrained")
    rng = np.random.default_rng(7)
    names = [r["session"].split("_")[0] for r in results]
    for index, (r, mouse) in enumerate(zip(results, names)):
        m = markers.get(mouse, "o")
        units = r["unit_results"]
        x = index + rng.uniform(-0.18, 0.18, len(units))
        for axis, key in ((axes[0, 0], "ks_ratio"), (axes[0, 1], "lag1_z")):
            values = np.array([u[key] for u in units])
            axis.scatter(
                x,
                values,
                s=9,
                marker=m,
                facecolor="none",
                edgecolor="black",
                lw=0.5,
                alpha=0.6,
            )
            axis.plot(
                [index - 0.22, index + 0.22],
                [np.median(values)] * 2,
                color="black",
                lw=2,
            )
        observed = np.array([u["multi_observed"] for u in units])
        predicted = np.array([u["multi_predicted"] for u in units])
        axes[1, 0].scatter(
            predicted,
            observed + 0.5,
            s=10,
            marker=m,
            facecolor="none",
            edgecolor="black",
            lw=0.5,
            alpha=0.7,
        )
        axes[1, 1].plot(
            r["time"]["time_s"],
            r["time"]["z"],
            color=("black", "gray", "silver", "tan")[index],
            lw=0.8,
            marker=m,
            markevery=20,
            ms=4,
            mfc="white",
        )
    for axis in axes[0]:
        axis.set_xticks(range(len(names)), names)
    axes[0, 0].set(yscale="log", ylabel="KS distance / 5% critical value")
    axes[0, 0].axhline(1, color="gray", ls="--", lw=0.8)
    axes[0, 0].annotate(
        "reject\nabove",
        (1.01, 1),
        xycoords=("axes fraction", "data"),
        color="gray",
        fontsize=8,
        va="center",
    )
    axes[0, 1].set(ylabel="lag-1 correlation of rescaled intervals (z)")
    for level in (-1.96, 1.96):
        axes[0, 1].axhline(level, color="gray", ls="--", lw=0.8)
    lim = [0.3, max(axes[1, 0].get_xlim()[1], axes[1, 0].get_ylim()[1])]
    axes[1, 0].plot(lim, lim, color="gray", ls="--", lw=0.8)
    axes[1, 0].set(
        xscale="log",
        yscale="log",
        xlim=lim,
        ylim=lim,
        xlabel="predicted bins with ≥2 spikes",
        ylabel="observed bins with ≥2 spikes (+0.5)",
    )
    axes[1, 1].axvline(0, color="gray", ls="--", lw=0.8)
    for level in (-1.96, 1.96):
        axes[1, 1].axhline(level, color="gray", ls=":", lw=0.8)
    axes[1, 1].set(
        xlabel="time from first flash (s)",
        ylabel="summed residual / √expected (10-ms bins)",
    )
    for letter, axis in zip("abcd", axes.flat):
        axis.text(-0.16, 1.04, letter, transform=axis.transAxes, weight="bold")
    fig.savefig(out / "poisson_gof.svg")
    fig.savefig(out / "poisson_gof.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/fano"))
    args = parser.parse_args()
    self_check()
    args.out.mkdir(parents=True, exist_ok=True)
    results, missing = [], []
    for name in SESSIONS:
        try:
            results.append(session(args.data_root, name))
        except FileNotFoundError as error:
            missing.append(dict(session=name, reason=str(error)))
            print(f"Skipped {name}: {error}", flush=True)
        (args.out / "poisson_gof.json").write_text(
            json.dumps(
                dict(
                    rate_edges_hz=[0, 10, 30, None], sessions=results, missing=missing
                ),
                indent=2,
            )
        )
    plot_report(results, args.out)


if __name__ == "__main__":
    main()
