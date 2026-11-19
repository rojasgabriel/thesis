"""Check held-out conditional Poisson dispersion of the saved GLM.

Comparison: recorded counts against out-of-fold expected counts, including
strictly past observed spike history; no new fits or simulations of history.
Units pass quality criterion 1 and stability set 0, without a sensory-response
filter; fit.py exclusions apply. Trials are completed left/right trials with
measured first flashes and finite port events, using saved GLM windows.
First-flash alignment and pre-response validity are those of fit.py. No baseline subtraction; the reference is dispersion 1.

Report mean squared Pearson residual at native 1-ms resolution, and sum of
squared accumulated count innovations divided by sum of expected counts over
nonoverlapping 1/10/50/100/200-ms blocks. Blocks start at the saved trial grid;
incomplete blocks or those containing invalid bins are omitted. Trials never
share a block. Correct conditional Poisson innovations have variance equal to
expected counts and zero serial covariance, so the latter statistic targets 1
at each scale. It is not an ordinary Poisson test of aggregated counts with
observed future history treated as a fixed covariate. Predictable history makes
the compensator random. Mean error and residual dependence also affect this
statistic: departures do not isolate a wrong noise family.

Sampling unit: neuron; summarize each session separately. Per-neuron intervals
bootstrap whole trials 500 times, holding predictions fixed (no refit
uncertainty). Folds with no saved model because they have zero spikes are
reported and omitted, following calibration.py. Missing sessions are explicit.

Run: uv run python -m thesis.ephys.analyses.glm.conditional_dispersion \\
    --data-root <checkout containing saved GLM fits and spike cache>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from thesis.behavior.trials import SESSIONS
from thesis.ephys.analyses.glm import dmat, fit

WIDTHS = (1, 10, 50, 100, 200)


def trial_statistics(y, mu, valid, width):
    """Return squared block innovations and expected counts, summed per trial."""
    n_trials, bins = y.shape
    stop = bins // width * width
    shape = (n_trials, -1, width)
    keep = valid[:, :stop].reshape(shape).all(axis=2)
    residual = (y[:, :stop] - mu[:, :stop]).reshape(shape).sum(axis=2)
    expected = mu[:, :stop].reshape(shape).sum(axis=2)
    return (residual**2 * keep).sum(axis=1), (expected * keep).sum(axis=1)


def self_check():
    y = np.array([[0.0, 2.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0]])
    mu = np.ones_like(y)
    valid = np.ones_like(y, dtype=bool)
    a, b = trial_statistics(y, mu, valid, 2)
    assert np.array_equal(a, [0, 4]) and np.array_equal(b, [4, 4])
    valid[1, -1] = False
    a, b = trial_statistics(y, mu, valid, 2)
    assert np.array_equal(a, [0, 0]) and np.array_equal(b, [4, 2])
    rng = np.random.default_rng(3)
    mu = rng.uniform(0.01, 0.2, (2000, 200))
    y = rng.poisson(mu)
    for width in WIDTHS:
        a, b = trial_statistics(y, mu, np.ones_like(y, dtype=bool), width)
        assert abs(a.sum() / b.sum() - 1) < 0.1

    # The reference must also hold when intensity depends on earlier spikes.
    rng = np.random.default_rng(24)
    y = np.zeros((2000, 200))
    mu = np.empty_like(y)
    for t in range(y.shape[1]):
        mu[:, t] = 0.05 + (0.3 * y[:, t - 1] if t else 0)
        y[:, t] = rng.poisson(mu[:, t])
    for width in WIDTHS:
        a, b = trial_statistics(y, mu, np.ones_like(y, dtype=bool), width)
        assert abs(a.sum() / b.sum() - 1) < 0.1


def unit_predictions(data_root, name):
    """Yield each unit's observed counts and out-of-fold expected counts.

    Arrays are trials x bins on the saved GLM grid; ``scored`` marks valid
    held-out bins. Each fold must reproduce its saved held-out likelihood.
    """
    subject, date = name.split("_", 1)
    path = data_root / "results/glm" / name
    units = pd.read_pickle(
        data_root
        / "figures/cache"
        / f"fetch_unit_table_subject-{subject}_session-{date}_unit_criteria_id-1_stability_param_id-0_include_metrics-False.pkl"
    ).set_index("unit_id")
    prepared = dmat._load_windows(path / "stimulus_windows_me.npz")
    metadata = json.loads((path / "common_design_me.json").read_text())
    common = np.load(path / "common_design_me.npy", mmap_mode="r", allow_pickle=False)
    valid = dmat._valid_bin_mask(prepared)
    n_trials = len(prepared["alignments"])
    bins = len(valid) // n_trials
    parts = fit.cross_validation_partitions(n_trials, bins, valid)
    records = fit.load_unit_records(
        path / "all_fit_me/checkpoints", "unit_*_folds.json"
    )
    if not records:
        raise ValueError(f"{name}: no fold fits")
    for index, record in enumerate(records):
        unit_id = int(record["unit_id"])
        spikes = np.asarray(units.loc[unit_id, "spike_times_s"], dtype=float)
        y = fit.build_unit_counts(prepared["alignments"], spikes)
        history = dmat.build_unit_history(prepared["alignments"], spikes)
        mu = np.zeros(len(y))
        scored = np.zeros(len(y), dtype=bool)
        eligible = [part for part in parts if y[part["test"]].sum() > 0]
        columns = int(metadata["base_columns"]) + int(record["components"])
        for part, fold in zip(eligible, record["folds"], strict=True):
            scaled, _, _ = dmat.training_zscore(history, part["fit"])
            design = dmat._design_with_history(common, scaled, columns, part["test"])
            pred = np.exp(fold["intercept"] + design @ np.asarray(fold["coefficients"]))
            if not np.isfinite(pred).all() or (pred <= 0).any():
                raise ValueError(f"{name}/{unit_id}: invalid expected counts")
            # Saved held-out likelihood is an independent prediction reconstruction check.
            metrics = fit.poisson_metrics(y[part["test"]], pred, fold["fit_mean_count"])
            if not np.isclose(
                metrics["mean_nll"], fold["test"]["mean_nll"], rtol=2e-5, atol=1e-8
            ):
                raise ValueError(f"{name}/{unit_id}: reconstructed likelihood differs")
            mu[part["test"]] = pred
            scored[part["test"]] = True
        shape = (n_trials, bins)
        yield dict(
            unit_id=unit_id,
            y=y.reshape(shape),
            mu=mu.reshape(shape),
            scored=scored.reshape(shape),
            omitted_folds=len(parts) - len(eligible),
            centers_s=prepared["relative_centers"],
            n_trials=n_trials,
        )
        if (index + 1) % 20 == 0:
            print(f"{name}: {index + 1}/{len(records)} neurons", flush=True)


def session(data_root, name):
    results = []
    for unit in unit_predictions(data_root, name):
        unit_id, n_trials = unit["unit_id"], unit["n_trials"]
        y, mu, scored = (unit[k].ravel() for k in ("y", "mu", "scored"))
        residual = y[scored] - mu[scored]
        pearson = float(np.mean(residual**2 / mu[scored]))
        stats = [
            trial_statistics(unit["y"], unit["mu"], unit["scored"], width)
            for width in WIDTHS
        ]
        numerator, denominator = (
            np.stack([s[i] for s in stats], axis=1) for i in (0, 1)
        )
        keep_trials = denominator[:, 0] > 0
        numerator, denominator = numerator[keep_trials], denominator[keep_trials]
        rng = np.random.default_rng([20261118, unit_id])
        draws = rng.integers(len(numerator), size=(500, len(numerator)))
        boot = numerator[draws].sum(axis=1) / denominator[draws].sum(axis=1)
        ratio = numerator.sum(axis=0) / denominator.sum(axis=0)
        results.append(
            dict(
                unit_id=unit_id,
                pearson_1ms=pearson,
                innovation_dispersion=ratio.tolist(),
                interval_95=np.percentile(boot, [2.5, 97.5], axis=0).T.tolist(),
                observed_predicted=float(y[scored].sum() / mu[scored].sum()),
                scored_bins=int(scored.sum()),
                omitted_folds=unit["omitted_folds"],
                max_predicted_hz=float(mu[scored].max() * 1000),
            )
        )
    values = np.array([r["innovation_dispersion"] for r in results])
    pearson = np.array([r["pearson_1ms"] for r in results])
    summary = dict(
        session=name,
        neurons=len(results),
        glm_trials=n_trials,
        innovation_median=np.median(values, axis=0).tolist(),
        innovation_iqr=np.percentile(values, [25, 75], axis=0).T.tolist(),
        pearson_median=float(np.median(pearson)),
        pearson_iqr=np.percentile(pearson, [25, 75]).tolist(),
        observed_predicted_median=float(
            np.median([r["observed_predicted"] for r in results])
        ),
        omitted_folds_total=sum(r["omitted_folds"] for r in results),
    )
    print(json.dumps(summary), flush=True)
    return dict(**summary, units=results)


def plot_report(report, out):
    """Show timescale dependence and individual 100-ms estimates."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter

    results = report["sessions"]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.7), layout="constrained")
    rng = np.random.default_rng(7)
    for index, (r, color) in enumerate(zip(results, ("#3679ad", "#bd6836", "#66904c"))):
        subject = r["session"].split("_")[0]
        values = r["innovation_median"]
        axes[0].plot(WIDTHS, values, "o-", color=color)
        axes[0].annotate(
            subject,
            (200, values[-1]),
            xytext=(5, 8 if index else -8),
            textcoords="offset points",
            color=color,
            fontsize=9,
        )
        y = np.array([u["innovation_dispersion"][3] for u in r["units"]])
        ci = np.array([u["interval_95"][3] for u in r["units"]])
        x = index + rng.uniform(-0.18, 0.18, len(y))
        axes[1].vlines(x, ci[:, 0], ci[:, 1], color=color, alpha=0.16, lw=0.6)
        axes[1].scatter(x, y, s=7, color=color, alpha=0.5)
        axes[1].plot(
            [index - 0.2, index + 0.2], [np.median(y)] * 2, color="black", lw=2
        )
    axes[0].set(
        xscale="log",
        xlabel="accumulation window (ms)",
        ylabel="conditional dispersion (median neuron)",
        xlim=(0.8, 600),
    )
    axes[1].set(
        xticks=range(len(results)),
        xticklabels=[r["session"].split("_")[0] for r in results],
        ylabel="100-ms conditional dispersion",
        yscale="log",
    )
    axes[1].yaxis.set_major_locator(FixedLocator([0.5, 1, 2, 5, 10, 50, 100]))
    axes[1].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axes[1].yaxis.set_minor_locator(FixedLocator([]))
    for letter, axis in zip("ab", axes):
        axis.axhline(1, color="gray", linestyle="--", lw=0.8)
        axis.text(-0.13, 1.04, letter, transform=axis.transAxes, weight="bold")
    fig.savefig(out / "conditional.svg")
    fig.savefig(out / "conditional.png", dpi=180)
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
        (args.out / "conditional.json").write_text(
            json.dumps(
                dict(widths_ms=WIDTHS, sessions=results, missing=missing),
                indent=2,
                allow_nan=False,
            )
        )

    plot_report(dict(sessions=results), args.out)


if __name__ == "__main__":
    main()
