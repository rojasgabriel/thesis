"""Decode low- versus high-rate stimulus category from V1 population activity.

The analysis uses non-boundary trials with a left or right response. Neural
features are each unit's mean firing rate from 0 to 1 s after the first
stimulus. Each resample contains equal numbers of category x choice trials, so
category accuracy is not driven by an unequal choice distribution. The
sampling unit is the trial. Random stratified folds estimate held-out accuracy;
contiguous folds provide a session-order control. This analysis does not claim
that all choice or movement effects have been removed.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from thesis.ephys.analyses._decoding import load_decoding_data

WINDOW_START_S = 0.0
WINDOW_STOP_S = 1.0
CV_FOLDS = 10
N_RESAMPLES = 20
N_SHUFFLES = 100
RANDOM_SEED = 0


def _decoder():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000),
    )


def _balanced_indices(
    category: np.ndarray,
    choice: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    category_choice_pairs = [(cat, response) for cat in (0, 1) for response in (-1, 1)]
    counts = [
        np.sum((category == cat) & (choice == response))
        for cat, response in category_choice_pairs
    ]
    n_trials_per_pair = int(min(counts))
    if n_trials_per_pair < 2:
        raise ValueError(
            "Too few category x choice trials; least represented pair has "
            f"{n_trials_per_pair}"
        )
    return np.sort(
        np.concatenate(
            [
                rng.choice(
                    np.flatnonzero((category == cat) & (choice == response)),
                    n_trials_per_pair,
                    replace=False,
                )
                for cat, response in category_choice_pairs
            ]
        )
    )


def _scores(
    firing_rates: np.ndarray,
    category: np.ndarray,
    choice: np.ndarray,
    *,
    n_resamples: int,
    shuffle: bool,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, int]:
    random_scores = np.empty(n_resamples)
    blocked_scores = np.empty(n_resamples)
    n_trials = 0
    for resample in range(n_resamples):
        labels = category.copy()
        if shuffle:
            for response in (-1, 1):
                mask = choice == response
                labels[mask] = rng.permutation(labels[mask])
        indices = _balanced_indices(labels, choice, rng)
        features = firing_rates[indices]
        targets = labels[indices]
        n_trials = len(indices)
        random_cv = StratifiedKFold(
            n_splits=min(CV_FOLDS, int(np.bincount(targets).min())),
            shuffle=True,
            random_state=RANDOM_SEED + resample,
        )
        blocked_cv = [
            (train, test)
            for train, test in KFold(
                n_splits=min(CV_FOLDS, len(targets)), shuffle=False
            ).split(features)
            if np.unique(targets[train]).size == 2
            and np.unique(targets[test]).size == 2
        ]
        if len(blocked_cv) < 2:
            raise ValueError("Fewer than two contiguous folds contain both categories")
        random_scores[resample] = cross_val_score(
            _decoder(), features, targets, cv=random_cv, scoring="balanced_accuracy"
        ).mean()
        blocked_scores[resample] = cross_val_score(
            _decoder(), features, targets, cv=blocked_cv, scoring="balanced_accuracy"
        ).mean()
    return random_scores, blocked_scores, n_trials


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
        "--quick", action="store_true", help="Use 3 resamples and shuffles"
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
    category = trials["stim_category"].eq("high_rate").to_numpy(dtype=int)
    choice = trials["response"].to_numpy(dtype=int)
    repeats = 3 if args.quick else N_RESAMPLES
    shuffles = 3 if args.quick else N_SHUFFLES
    real_random, real_blocked, n_balanced = _scores(
        firing_rates,
        category,
        choice,
        n_resamples=repeats,
        shuffle=False,
        rng=np.random.default_rng(RANDOM_SEED),
    )
    null_random, null_blocked, _ = _scores(
        firing_rates,
        category,
        choice,
        n_resamples=shuffles,
        shuffle=True,
        rng=np.random.default_rng(RANDOM_SEED + 1),
    )
    results = {
        "subject": args.subject,
        "session": args.session,
        "unit_criteria_id": args.unit_criteria_id,
        "stability_param_id": args.stability_param_id,
        "n_units": len(units),
        "n_eligible_trials": len(trials),
        "n_trials_per_balanced_resample": n_balanced,
        "window_s": [WINDOW_START_S, WINDOW_STOP_S],
        "random_cv": {
            "accuracy_mean": float(real_random.mean()),
            "accuracy_sd": float(real_random.std(ddof=1)),
            "shuffle_mean": float(null_random.mean()),
            "p_one_sided": float(
                (np.sum(null_random >= real_random.mean()) + 1) / (len(null_random) + 1)
            ),
        },
        "blocked_cv": {
            "accuracy_mean": float(real_blocked.mean()),
            "accuracy_sd": float(real_blocked.std(ddof=1)),
            "shuffle_mean": float(null_blocked.mean()),
            "p_one_sided": float(
                (np.sum(null_blocked >= real_blocked.mean()) + 1)
                / (len(null_blocked) + 1)
            ),
        },
    }

    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    values = [real_random, null_random, real_blocked, null_blocked]
    colors = ["#377EB8", "0.75", "#E41A1C", "0.75"]
    positions = np.arange(4)
    for position, scores, color in zip(positions, values, colors, strict=True):
        ax.scatter(
            np.full(len(scores), position), scores, color=color, alpha=0.35, s=14
        )
        ax.plot(position, np.mean(scores), "o", color=color, ms=7)
    ax.axhline(0.5, color="0.5", linestyle="--", linewidth=1)
    ax.set_xticks(positions, ["random", "shuffle", "blocked", "shuffle"], rotation=30)
    ax.set_ylabel("balanced category accuracy")
    fig.tight_layout()

    output_dir = (
        Path(os.environ.get("THESIS_FIGURE_ROOT", "figures")) / "category_decoding"
    )
    if not args.no_save:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / "category_decoding.pdf", bbox_inches="tight")
        (output_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    if args.show:
        plt.show()
    else:
        plt.close(fig)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
