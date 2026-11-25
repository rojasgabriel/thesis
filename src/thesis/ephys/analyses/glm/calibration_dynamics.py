"""Do conditional calibration errors accompany generated-history instability?

Compare all 172/115 fitted units from the main GRB006/GRB064 sessions; retain
the original quality/stability exclusions, completed trials, first-flash
alignment and response-entry censoring. No sensory-response filter or refits.
Calibration uses the original held-out 1-ms bins and observed history. Dynamics
uses the saved 20-repeat uncapped check, observed pre-window history, and its
final-grid-bin exclusion. The two diagnostics therefore differ by that bin.

Calibration compression is median observed/predicted count at predicted-rate
percentiles 75--95 minus that at 5--70, within each unit; zero is the calibrated
baseline. Ratios, rather than logs, retain groups with zero observed spikes.
The primary comparison is its within-mouse Spearman association with the
fraction of generated trajectories crossing the saved failure threshold.
Secondary checks are absolute calibration error and adjustment for mean firing
rate (mean saved fold fit_mean_count), using partial rank correlation.

The sampling unit is the neuron. Plot individual scores by any/no crossing,
with median and unit-bootstrap 95% intervals. These intervals and associations
condition on the saved fits and simulations; they do not propagate calibration
or simulation estimation uncertainty, establish causation, or pool mice.

The legacy calibration cache has no unit IDs. Recover its row order using its
producer's sorted load_unit_records order, require equal unit sets, and verify
every fold hash against dynamics. Cached-row spot checks are recorded separately.

    uv run python -m thesis.ephys.analyses.glm.calibration_dynamics \\
        --root <main-session-parent>
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy.stats import rankdata, spearmanr, t

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thesis.ephys.analyses.glm import fit
from thesis.ephys.analyses.glm.cap_sensitivity import SESSIONS
from thesis.plotting import FIGURE_STYLE, MODEL_COLOR, save_figure


def scores(ratio):
    low = np.median(ratio[:, 1:14], axis=1)
    high = np.median(ratio[:, 15:19], axis=1)
    return low, high, high - low, np.mean(np.abs(ratio - 1), axis=1)


def partial_rank(x, y, rate):
    z = np.column_stack([np.ones(len(rate)), rankdata(rate)])
    a, b = [
        rankdata(v) - z @ np.linalg.lstsq(z, rankdata(v), rcond=None)[0] for v in (x, y)
    ]
    r = float(np.corrcoef(a, b)[0, 1])
    statistic = r * np.sqrt((len(x) - 3) / (1 - r * r))
    return {"rho": r, "approximate_p": float(2 * t.sf(abs(statistic), len(x) - 3))}


def analyze(root, dynamics, out):
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261117)
    report = {}
    with plt.rc_context(FIGURE_STYLE):
        fig, axes = plt.subplots(
            1, 2, figsize=(8, 3.5), layout="constrained", sharey=True
        )
        for letter, ax, session in zip("ab", axes, SESSIONS, strict=True):
            directory = root / session / "all_fit_me"
            cache = directory / "calibration.npz"
            records = fit.load_unit_records(
                directory / "checkpoints", "unit_*_folds.json"
            )
            path = dynamics / session / "dynamics.json"
            diagnostic = json.loads(path.read_text())
            units = {u["unit_id"]: u for u in diagnostic["units"]}
            with np.load(cache) as data:
                ratio = data["ratio"].copy()
            ids = [r["unit_id"] for r in records]
            if set(ids) != set(units) or ratio.shape != (len(ids), 20):
                raise ValueError("Calibration rows and dynamics unit set disagree.")
            for unit in ids:
                fold = directory / "checkpoints" / f"unit_{unit}_folds.json"
                if (
                    hashlib.sha256(fold.read_bytes()).hexdigest()
                    != diagnostic["fold_files_sha256"][str(unit)]
                ):
                    raise ValueError(
                        f"Dynamics and calibration fits differ for {unit}."
                    )
            if not np.isfinite(ratio).all():
                raise ValueError("Nonfinite calibration ratios need explicit handling.")
            failures = np.array([units[u]["failed_trajectories"] for u in ids])
            fraction = failures / np.array([units[u]["trajectories"] for u in ids])
            rate = 1000 * np.array(
                [np.mean([f["fit_mean_count"] for f in r["folds"]]) for r in records]
            )
            low, high, compression, error = scores(ratio)
            metrics = {
                "compression": compression,
                "absolute_error": error,
                "low_ratio": low,
                "high_ratio": high,
            }
            stats = {}
            for name, values in metrics.items():
                r = spearmanr(values, fraction)
                stats[name] = {
                    "rho": float(r.statistic),
                    "p": float(r.pvalue),
                    "median_no_crossing": float(np.median(values[failures == 0])),
                    "median_any_crossing": float(np.median(values[failures > 0])),
                }
            stats["compression_adjusted_for_rate"] = partial_rank(
                compression, fraction, rate
            )
            stats["rate_vs_failure"] = {
                "rho": float(spearmanr(rate, fraction).statistic)
            }
            groups = []
            for k, flagged in enumerate((False, True)):
                values = compression[(failures > 0) == flagged]
                boot = np.median(rng.choice(values, (2000, len(values))), axis=1)
                lo, hi = np.quantile(boot, [0.025, 0.975])
                median = float(np.median(values))
                groups.append(
                    {"n": len(values), "median": median, "bootstrap95": [lo, hi]}
                )
                ax.scatter(
                    k + rng.uniform(-0.15, 0.15, len(values)),
                    values,
                    color=MODEL_COLOR,
                    alpha=0.45,
                    s=12,
                )
                ax.errorbar(
                    k,
                    median,
                    yerr=[[median - lo], [hi - median]],
                    fmt="o",
                    color=MODEL_COLOR,
                    capsize=4,
                    markersize=7,
                )
            subject = session.split("_")[0]
            ax.axhline(0, color="black", lw=0.7)
            ax.set(
                xticks=[0, 1],
                xticklabels=[
                    f"No crossings\nn={groups[0]['n']}",
                    f"≥1 crossing\nn={groups[1]['n']}",
                ],
                ylabel="High-rate − low-rate observed / predicted",
            )
            ax.text(0.02, 1.03, subject, transform=ax.transAxes, va="bottom")
            ax.text(-0.12, 1.05, letter, transform=ax.transAxes, weight="bold")
            report[subject] = {
                "statistics": stats,
                "groups": groups,
                "calibration_sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
                "dynamics_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "units": [
                    {
                        "unit_id": u,
                        "failure_fraction": float(fraction[j]),
                        "mean_fit_rate_hz": float(rate[j]),
                        **{name: float(v[j]) for name, v in metrics.items()},
                    }
                    for j, u in enumerate(ids)
                ],
            }
        fig.savefig(out / "calibration_dynamics.png", dpi=160, bbox_inches="tight")
        save_figure(fig, out / "calibration_dynamics.svg")
    (out / "summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({s: e["statistics"] for s, e in report.items()}, indent=2))


def self_check():
    ratio = np.ones((2, 20))
    ratio[1, 1:14], ratio[1, 15:19] = 0.8, 1.2
    low, high, compression, error = scores(ratio)
    np.testing.assert_allclose(compression, [0, 0.4])
    assert low[1] < 1 < high[1] and error[0] == 0
    rate = np.arange(20)
    x = np.array([1, 4, 2, 5, 3] * 4)
    assert partial_rank(x, x + np.sin(rate) * 0.01, rate)["rho"] > 0.95
    print("Calibration contrast and partial-rank self-check passed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--dynamics", type=Path, default=Path("results/glm/dynamics"))
    parser.add_argument(
        "--out", type=Path, default=Path("results/glm/calibration_dynamics")
    )
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
    elif args.root is None:
        parser.error("--root is required")
    else:
        analyze(args.root, args.dynamics, args.out)


if __name__ == "__main__":
    main()
