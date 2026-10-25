"""Decode stimulus category and choice from V1 population activity.

Comparison: two decoders on the same trials. Panel a decodes the low- or
high-rate stimulus category. Panel b decodes the left or right choice. Each is
compared with a label shuffle.

Trials: non-boundary trials with a left or right response. Each resample has
equal numbers of trials in the four category x choice groups, so the category
decoder cannot use choice and the choice decoder cannot use category. The
shuffle permutes the decoded label inside each level of the other variable.

Units: by default the single units and multi-units of the unit criteria (not
noise), with no stability filter; `--units sua`: single units only (outputs
with `_sua`), with the optional stability filter.

Features: each unit's mean firing rate from 0 to 1 s after the first measured
flash. This window includes the time after withdrawal for many trials, so the
choice decoder can read the turn to the chosen side. Choice and movement are not
removed from the category decoder.

Folds: "random folds" assign trials to the 10 test folds at random, with both
labels in each fold. "Blocked folds" cut the session into 10 blocks of
consecutive trials and test on one block at a time. If the two agree, the
decoder does not depend on slow changes over the session, such as drift.

Sampling unit: the trial. Each point is the mean held-out balanced accuracy of
one resample (black) or one shuffle (gray); the large marker is their mean.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

from thesis.ephys.analyses._decoding import balanced_indices as _balanced_indices
from thesis.ephys.analyses._decoding import decoder as _decoder
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
N_RESAMPLES = 20
N_SHUFFLES = 100
RANDOM_SEED = 0


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


# The y axis of both panels in every session (display only).
ACCURACY_LIMITS = (0.2, 1.02)


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
    optional.add_argument(
        "--units",
        choices=("all", "sua"),
        default="all",
        help="all: single units and multi-units (no stability filter); sua: single units only",
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
        include_mua=args.units == "all",
        window_stop_s=WINDOW_STOP_S,
    )
    category = trials["stim_category"].eq("high_rate").to_numpy(dtype=int)
    choice = trials["response"].to_numpy(dtype=int)
    repeats = 3 if args.quick else N_RESAMPLES
    shuffles = 3 if args.quick else N_SHUFFLES
    # Each target is decoded with the other variable balanced, so one decoder
    # cannot use the other variable. `_scores` wants a 0/1 target and a -1/1
    # control.
    targets = {
        "category": (category, choice),
        "choice": ((choice == 1).astype(int), np.where(category == 1, 1, -1)),
    }
    results = {
        "subject": args.subject,
        "session": args.session,
        "unit_criteria_id": args.unit_criteria_id,
        "stability_param_id": args.stability_param_id,
        "units": args.units,
        "n_units": len(units),
        "n_eligible_trials": len(trials),
        "window_s": [WINDOW_START_S, WINDOW_STOP_S],
    }
    scores = {}
    for name, (target, control) in targets.items():
        real_random, real_blocked, n_balanced = _scores(
            firing_rates,
            target,
            control,
            n_resamples=repeats,
            shuffle=False,
            rng=np.random.default_rng(RANDOM_SEED),
        )
        null_random, null_blocked, _ = _scores(
            firing_rates,
            target,
            control,
            n_resamples=shuffles,
            shuffle=True,
            rng=np.random.default_rng(RANDOM_SEED + 1),
        )
        scores[name] = [real_random, null_random, real_blocked, null_blocked]
        results[name] = {"n_trials_per_balanced_resample": n_balanced}
        for folds, real, null in (
            ("random_cv", real_random, null_random),
            ("blocked_cv", real_blocked, null_blocked),
        ):
            results[name][folds] = {
                "accuracy_mean": float(real.mean()),
                "accuracy_sd": float(real.std(ddof=1)),
                "shuffle_mean": float(null.mean()),
                "p_one_sided": float(
                    (np.sum(null >= real.mean()) + 1) / (len(null) + 1)
                ),
            }

    with plt.rc_context(FIGURE_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0), sharey=True)
        positions = np.arange(4)
        for ax, (name, values) in zip(axes, scores.items(), strict=True):
            # One comparison: decoder against shuffle. The fold type is on the axis.
            for position, accuracy, color in zip(
                positions, values, (OBSERVED_COLOR, "0.75") * 2, strict=True
            ):
                ax.scatter(
                    np.full(len(accuracy), position),
                    accuracy,
                    color=color,
                    alpha=0.35,
                    s=10,
                    linewidths=0,
                )
                ax.plot(position, np.mean(accuracy), "o", color=color, ms=6)
            ax.axhline(0.5, color=REFERENCE_COLOR, linestyle=":", linewidth=0.8)
            ax.set_xticks(
                positions,
                ["random folds", "shuffle", "blocked folds", "shuffle"],
                rotation=30,
                ha="right",
            )
            ax.set_ylabel(f"balanced {name} accuracy")
            # One scale for every session, so that sessions and mice compare.
            ax.set_ylim(*ACCURACY_LIMITS)
            outside = sum(
                int(
                    np.sum(
                        (np.asarray(v) < ACCURACY_LIMITS[0])
                        | (np.asarray(v) > ACCURACY_LIMITS[1])
                    )
                )
                for v in values
            )
            if outside:
                print(
                    f"{name}: {outside} values outside the axis limits (kept in the statistics)"
                )
            ax.tick_params(labelleft=True)
        label_panels(axes, -0.22)
        fig.tight_layout(w_pad=2.0)

    print(json.dumps(results, indent=2))
    if args.show:
        plt.show()
    if not args.no_save:
        output_dir = (
            Path(os.environ.get("THESIS_FIGURE_ROOT", "figures"))
            / "category_decoding"
            / (
                f"{args.subject}_{args.session}"
                + ("_sua" if args.units == "sua" else "")
            )
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        save_figure(fig, output_dir / "category_decoding")
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
