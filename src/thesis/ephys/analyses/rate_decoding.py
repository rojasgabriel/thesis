"""Decode nominal visual stimulus rate from V1 population activity.

Comparison: the tuning of single units to the stimulus rate (a, b), then ridge
regression of the stimulus rate on held-out trials against the same decoder
trained on shuffled rates (c, d).

- Panel a: raw tuning, with no decoder. Each row is one unit and each column
  is one stimulus rate. The color is the unit's mean firing rate at that rate,
  as a z-score over the ten rates. Units are sorted by the value in panel b.
- Panel b: each unit's Spearman correlation between its firing rate and the
  stimulus rate, over trials. Gray is the same after the rates are permuted
  over trials (100 times, all units pooled).
- Panel c: the mean decoded rate for each true rate.
- Panel d: the residual rate, the decoded minus the true rate.
- Shuffle (gray in panels c and d): the rates are permuted over trials before
  the decoder is trained. That decoder has no information, so it answers near
  the mean rate for every trial: a flat line in panel c and a steep line in
  panel d. The real decoder is not perfect, so it is pulled toward the mean
  rate too and sits between the shuffle and the dotted line of a perfect
  decoder.

Trials: non-boundary trials with a left or right response. Features: each
unit's mean firing rate from 0 to 1 s after the first measured flash. Choice is
not removed, so this analysis tests whether rate is readable, not whether the
signal is purely sensory.

Folds: 10 random folds for the figure and the shuffles. The results file also
has 10 blocked folds (consecutive trials), as a session-order control.
Sampling unit: the trial. The black points are the mean and SEM over trials.
The gray line is the median over 100 shuffles, and its band is the 2.5th to
97.5th percentile of the shuffles.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from thesis.ephys.analyses._decoding import load_decoding_data
from thesis.plotting import (
    FIGURE_STYLE,
    OBSERVED_COLOR,
    REFERENCE_COLOR,
    label_panels,
    save_figure,
)

WINDOW_START_S = 0.0
WINDOW_STOP_S = 1.0
CV_FOLDS = 10
RIDGE_ALPHAS = np.logspace(-4, 6, 21)
RANDOM_SEED = 0
# Each shuffle is a full nested cross-validation (about 2 s).
N_SHUFFLES = 100


def _ridge_predictions(
    firing_rates: np.ndarray, rate_hz: np.ndarray, cv: KFold, alphas: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    predictions = np.full(len(rate_hz), np.nan)
    fold_r2 = []
    for train, test in cv.split(firing_rates):
        model = GridSearchCV(
            make_pipeline(StandardScaler(), Ridge()),
            {"ridge__alpha": alphas},
            cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED),
            scoring="r2",
        ).fit(firing_rates[train], rate_hz[train])
        predictions[test] = model.predict(firing_rates[test])
        fold_r2.append(model.score(firing_rates[test], rate_hz[test]))
    return predictions, np.asarray(fold_r2)


def _unit_rate_correlations(
    firing_rates: np.ndarray, rate_hz: np.ndarray
) -> np.ndarray:
    """Return each unit's Spearman correlation of trial firing rate with stimulus rate."""
    unit_ranks = rankdata(firing_rates, axis=0)
    rate_ranks = rankdata(rate_hz)
    unit_ranks = unit_ranks - unit_ranks.mean(axis=0)
    rate_ranks = rate_ranks - rate_ranks.mean()
    return (unit_ranks.T @ rate_ranks) / (
        np.linalg.norm(unit_ranks, axis=0) * np.linalg.norm(rate_ranks)
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    required = parser.add_argument_group("required arguments")
    required.add_argument("-a", "--subject", required=True)
    required.add_argument("-s", "--session", required=True)
    optional = parser.add_argument_group("optional arguments")
    optional.add_argument(
        "--unit-criteria-id", type=int, default=1, help="Unit quality criteria"
    )
    optional.add_argument(
        "--stability-param-id", type=int, help="Require passing unit stability"
    )
    optional.add_argument(
        "--quick", action="store_true", help="Use a smaller ridge grid"
    )
    optional.add_argument("--show", action="store_true", help="Show the figure")
    optional.add_argument(
        "--no-save", action="store_true", help="Do not save the figure or results"
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.show:
        matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    units, trials, firing_rates = load_decoding_data(
        args.subject,
        args.session,
        unit_criteria_id=args.unit_criteria_id,
        stability_param_id=args.stability_param_id,
        window_start_s=WINDOW_START_S,
        window_stop_s=WINDOW_STOP_S,
    )
    rate_hz = trials["visual_stim_rate_hz"].to_numpy(dtype=float)
    alphas = np.logspace(-4, 6, 7) if args.quick else RIDGE_ALPHAS
    random_cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    blocked_cv = KFold(n_splits=CV_FOLDS, shuffle=False)
    random_pred, random_r2 = _ridge_predictions(
        firing_rates, rate_hz, random_cv, alphas
    )
    blocked_pred, blocked_r2 = _ridge_predictions(
        firing_rates, rate_hz, blocked_cv, alphas
    )

    shuffle_rng = np.random.default_rng(RANDOM_SEED + 1)
    rates = np.unique(rate_hz)
    null_r2, null_means = [], []
    for _ in range(3 if args.quick else N_SHUFFLES):
        shuffled = shuffle_rng.permutation(rate_hz)
        prediction, fold_r2 = _ridge_predictions(
            firing_rates, shuffled, random_cv, alphas
        )
        null_r2.append(fold_r2.mean())
        # Group by the rate the decoder was asked to find in that shuffle.
        null_means.append([prediction[shuffled == rate].mean() for rate in rates])
    null_r2, null_means = np.asarray(null_r2), np.asarray(null_means)
    null_low, null_mid, null_high = np.percentile(null_means, [2.5, 50, 97.5], axis=0)

    # Raw tuning, with no decoder: each unit's mean rate at each stimulus rate,
    # and its rank correlation with the stimulus rate over trials.
    tuning = np.stack([firing_rates[rate_hz == rate].mean(axis=0) for rate in rates])
    spread = tuning.std(axis=0)
    tuning_z = (tuning - tuning.mean(axis=0)) / np.where(spread > 0, spread, 1)
    unit_rho = _unit_rate_correlations(firing_rates, rate_hz)
    null_rho = np.concatenate(
        [
            _unit_rate_correlations(firing_rates, shuffle_rng.permutation(rate_hz))
            for _ in range(N_SHUFFLES)
        ]
    )
    rho_low, rho_high = np.percentile(null_rho, [2.5, 97.5])
    rate_summary = (
        pd.DataFrame({"rate_hz": rate_hz, "prediction_hz": random_pred})
        .groupby("rate_hz", sort=True)["prediction_hz"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    rate_summary["sem"] = rate_summary["std"] / np.sqrt(rate_summary["count"])
    rate_summary["residual_hz"] = rate_summary["mean"] - rate_summary["rate_hz"]
    calibration_slope, calibration_intercept = np.polyfit(rate_hz, random_pred, 1)
    results = {
        "subject": args.subject,
        "session": args.session,
        "unit_criteria_id": args.unit_criteria_id,
        "stability_param_id": args.stability_param_id,
        "n_units": len(units),
        "n_trials": len(trials),
        "window_s": [WINDOW_START_S, WINDOW_STOP_S],
        "continuous_rate": {
            "random_cv_r2_mean": float(random_r2.mean()),
            "random_cv_r2_folds": random_r2.tolist(),
            "blocked_cv_r2_mean": float(blocked_r2.mean()),
            "blocked_cv_r2_folds": blocked_r2.tolist(),
            "calibration_slope": float(calibration_slope),
            "calibration_intercept_hz": float(calibration_intercept),
            "mae_hz": float(np.mean(np.abs(random_pred - rate_hz))),
            "rmse_hz": float(np.sqrt(np.mean((random_pred - rate_hz) ** 2))),
        },
        "shuffle": {
            "n_shuffles": len(null_r2),
            "random_cv_r2_mean": float(null_r2.mean()),
            "p_one_sided": float(
                (np.sum(null_r2 >= random_r2.mean()) + 1) / (len(null_r2) + 1)
            ),
            "decoded_rate_by_rate_hz": null_means.mean(axis=0).tolist(),
        },
        "unit_tuning": {
            "spearman_rho_median": float(np.median(unit_rho)),
            "shuffle_95_percent_range": [float(rho_low), float(rho_high)],
            "fraction_units_above_range": float(np.mean(unit_rho > rho_high)),
            "fraction_units_below_range": float(np.mean(unit_rho < rho_low)),
        },
        "by_rate": rate_summary.round(6).to_dict(orient="records"),
    }

    with plt.rc_context(FIGURE_STYLE):
        fig, axes = plt.subplots(
            1, 4, figsize=(10.5, 2.7), gridspec_kw={"width_ratios": (1.25, 1, 1, 1)}
        )
        order = np.argsort(unit_rho)
        image = axes[0].imshow(
            tuning_z[:, order].T,
            aspect="auto",
            cmap="RdBu_r",
            vmin=-2.5,
            vmax=2.5,
            interpolation="nearest",
        )
        axes[0].set_xticks(np.arange(len(rates)), rates.astype(int))
        axes[0].set_xlabel("stimulus rate (Hz)")
        axes[0].set_ylabel("units, sorted by rate correlation")
        fig.colorbar(image, ax=axes[0], label="firing rate (z-score over rates)")

        bins = np.linspace(-0.6, 0.6, 31)
        axes[1].hist(null_rho, bins=bins, density=True, color="0.8")
        axes[1].hist(
            unit_rho, bins=bins, density=True, histtype="step", color=OBSERVED_COLOR
        )
        axes[1].axvline(0, color=REFERENCE_COLOR, linestyle=":", linewidth=0.8)
        axes[1].set_xlabel(
            "correlation of firing rate\nwith stimulus rate (Spearman ρ)"
        )
        axes[1].set_ylabel("density of units")
        for row, (label, color) in enumerate(
            (("units", OBSERVED_COLOR), ("shuffle", "0.5"))
        ):
            axes[1].text(
                0.04,
                0.95 - 0.09 * row,
                label,
                color=color,
                transform=axes[1].transAxes,
                va="top",
            )

        for axis, real, offset in (
            (axes[2], rate_summary["mean"], 0),
            (axes[3], rate_summary["residual_hz"], rates),
        ):
            axis.fill_between(
                rates, null_low - offset, null_high - offset, color="0.85", linewidth=0
            )
            axis.plot(rates, null_mid - offset, color="0.6", linewidth=1)
            axis.errorbar(
                rate_summary["rate_hz"],
                real,
                yerr=rate_summary["sem"],
                color=OBSERVED_COLOR,
                marker="o",
                ms=4,
                capsize=2,
            )
            axis.set_xlabel("stimulus rate (Hz)")
        limits = [float(rate_hz.min()), float(rate_hz.max())]
        axes[2].plot(
            limits, limits, color=REFERENCE_COLOR, linestyle=":", linewidth=0.8
        )
        axes[2].set_ylabel("decoded rate (Hz)")
        axes[3].axhline(0, color=REFERENCE_COLOR, linestyle=":", linewidth=0.8)
        axes[3].set_ylabel("residual rate (Hz)")
        for row, (label, color) in enumerate(
            (("decoder", OBSERVED_COLOR), ("shuffle", "0.5"))
        ):
            axes[2].text(
                0.04,
                0.95 - 0.09 * row,
                label,
                color=color,
                transform=axes[2].transAxes,
                va="top",
            )
        label_panels(axes, -0.3)
        fig.tight_layout(w_pad=1.2)

    print(json.dumps(results, indent=2))
    if args.show:
        plt.show()
    if not args.no_save:
        output_dir = (
            Path(os.environ.get("THESIS_FIGURE_ROOT", "figures")) / "rate_decoding"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        save_figure(fig, output_dir / "rate_decoding")
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
