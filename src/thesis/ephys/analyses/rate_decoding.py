"""Decode nominal visual stimulus rate from V1 population activity.

The analysis uses non-boundary trials with a left or right response. Neural
features are each unit's mean firing rate from 0 to 1 s after the first
stimulus. Outer cross-validation estimates continuous rate prediction and
exact-rate classification on held-out trials. Random folds measure general
held-out performance and contiguous folds test sensitivity to session order.
The sampling unit is the trial. Because choice is not removed, this analysis
tests whether rate is readable, not whether the signal is purely sensory.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from thesis.ephys.analyses._decoding import load_decoding_data

WINDOW_START_S = 0.0
WINDOW_STOP_S = 1.0
CV_FOLDS = 10
RIDGE_ALPHAS = np.logspace(-4, 6, 21)
RANDOM_SEED = 0


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


def _balanced_rate_indices(rate_hz: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    rates, counts = np.unique(rate_hz, return_counts=True)
    n_per_rate = int(counts.min())
    return np.sort(
        np.concatenate(
            [
                rng.choice(np.flatnonzero(rate_hz == rate), n_per_rate, replace=False)
                for rate in rates
            ]
        )
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

    balanced = _balanced_rate_indices(rate_hz, np.random.default_rng(RANDOM_SEED))
    balanced_rates = rate_hz[balanced]
    classifier = make_pipeline(
        StandardScaler(), LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    )
    class_pred = cross_val_predict(
        classifier,
        firing_rates[balanced],
        balanced_rates,
        cv=KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED),
    )
    exact_rates = np.unique(rate_hz)
    confusion = confusion_matrix(
        balanced_rates, class_pred, labels=exact_rates, normalize="true"
    )
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
        "exact_rate": {
            "n_balanced_trials": len(balanced),
            "balanced_accuracy": float(
                balanced_accuracy_score(balanced_rates, class_pred)
            ),
            "chance": float(1 / len(exact_rates)),
            "rates_hz": exact_rates.tolist(),
            "confusion": confusion.tolist(),
        },
        "by_rate": rate_summary.round(6).to_dict(orient="records"),
    }

    fig, axes = plt.subplots(1, 3, figsize=(8.0, 2.8), constrained_layout=True)
    axes[0].errorbar(
        rate_summary["rate_hz"],
        rate_summary["mean"],
        yerr=rate_summary["sem"],
        color="#377EB8",
        marker="o",
        capsize=2,
    )
    limits = [float(rate_hz.min()), float(rate_hz.max())]
    axes[0].plot(limits, limits, color="0.5", linestyle="--", linewidth=1)
    axes[0].set_xlabel("stimulus rate (Hz)")
    axes[0].set_ylabel("decoded rate (Hz)")
    axes[1].errorbar(
        rate_summary["rate_hz"],
        rate_summary["residual_hz"],
        yerr=rate_summary["sem"],
        color="#377EB8",
        marker="o",
        capsize=2,
    )
    axes[1].axhline(0, color="0.5", linestyle="--", linewidth=1)
    axes[1].set_xlabel("stimulus rate (Hz)")
    axes[1].set_ylabel("decoded - true (Hz)")
    image = axes[2].imshow(confusion, vmin=0, vmax=1, cmap="viridis")
    axes[2].set_xticks(
        np.arange(len(exact_rates)), exact_rates.astype(int), rotation=45
    )
    axes[2].set_yticks(np.arange(len(exact_rates)), exact_rates.astype(int))
    axes[2].set_xlabel("decoded rate (Hz)")
    axes[2].set_ylabel("stimulus rate (Hz)")
    fig.colorbar(image, ax=axes[2], label="fraction")
    for letter, ax in zip("abc", axes, strict=True):
        ax.text(
            -0.18,
            1.08,
            letter,
            fontweight="bold",
            transform=ax.transAxes,
        )

    output_dir = Path(os.environ.get("THESIS_FIGURE_ROOT", "figures")) / "rate_decoding"
    if not args.no_save:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / "rate_decoding.pdf", bbox_inches="tight")
        (output_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    if args.show:
        plt.show()
    else:
        plt.close(fig)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
