"""Test whether the V1 stimulus-category readout survives leaving the center port.

Question: when the mouse leaves the center port, does V1 keep the same
population axis for the stimulus category (additive or multiplicative gain), or
does the category move to other dimensions (reformatting)? A decoder trained in
one state should then fail in the other.

Comparison: a logistic decoder of the low- or high-rate category, trained in
one window and tested on held-out trials in the same window (within) or in the
other window (transfer). Each trial gives three windows of equal length W,
aligned to its matched center exit:

- early stationary: exit - 2W to exit - W
- stationary: exit - W to exit
- movement: exit to exit + W

"Movement" is a task epoch, as in `stim_responses_by_locomotion`; this script
does not measure locomotion. The stationary and movement windows differ in
state and are W apart in time. The early-stationary and stationary windows are
also W apart but share the state, so their transfer cost is the cost of elapsed
time, flash adaptation, and pulse order alone (panel c).

Scaling: each window's features are z-scored with the mean and SD of that
window's training trials, without labels. A per-unit gain or offset change
between states therefore does not by itself break transfer; only a change in
which units carry the category does. The results file also reports transfer
with the training window's scaling ("same scaling"), where a gain change can
break transfer.

Trials: non-boundary trials with a left or right response, a matched center
entry, exit, and response-port entry, a first flash at least 2W before exit, and
a stimulus (first flash + stimulus duration) and response-port entry at least W
after exit. Each resample has equal numbers of trials in the four category x
choice groups, so the decoder cannot read the turn direction as category. The
shuffle permutes category inside each choice before balancing.

Units: units that pass quality criterion 1 (and stability, if requested).
Features: each unit's mean firing rate in the window. Folds: 10 stratified
random folds of the trials; train and test windows always come from different
trials.

Panels: (a) held-out balanced accuracy for each train -> test pair; each point
is one resample (black, filled), open circles are transfer with the training
window's scaling, and the gray bar is the 2.5th-97.5th percentile of the shuffle. (b) each unit's
category selectivity (d', averaged over the two choices) in the stationary and
movement windows, with bootstrap SE in both directions. (c) transfer cost,
within minus transfer accuracy, for the time-only shift and the time + state
shift; lines pair the two costs of one resample.

Sampling unit: the trial. Resamples reuse trials, so their spread shows the
variability of the balanced subsample, not independent replicates. One session
is descriptive; the test is the same comparison repeated over sessions and
animals.

Use `--synthetic gain` or `--synthetic reformat` to run the same analysis on
simulated units where the answer is known.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from thesis.ephys.analyses._decoding import eligible_choice_trials, window_firing_rates
from thesis.ephys.trials import build_trial_table
from thesis.ephys.units import fetch_unit_table
from thesis.plotting import (
    FIGURE_STYLE,
    OBSERVED_COLOR,
    REFERENCE_COLOR,
    label_panels,
    save_figure,
)

WINDOW_S = 0.2
CV_FOLDS = 10
N_RESAMPLES = 20
N_SHUFFLES = 100
N_BOOTSTRAPS = 200
RANDOM_SEED = 0
WINDOWS = ("early stationary", "stationary", "movement")
# (train window, test window) pairs, in panel a order.
PAIRS = (
    ("stationary", "stationary"),
    ("movement", "movement"),
    ("stationary", "movement"),
    ("movement", "stationary"),
)
TIME_PAIRS = (
    ("early stationary", "early stationary"),
    ("early stationary", "stationary"),
    ("stationary", "early stationary"),
)


def select_trials(trials: pd.DataFrame, window_s: float) -> pd.DataFrame:
    """Return the trials whose three windows fit inside the stimulus and task."""
    first_stim_s = trials["stim_pulse_times_s"].str[0].astype(float)
    exit_s = trials["center_exit_s"]
    keep = (
        np.isfinite(trials["center_entry_s"])
        & np.isfinite(exit_s)
        & np.isfinite(trials["response_port_entry_s"])
        & (first_stim_s <= exit_s - 2 * window_s)
        & (first_stim_s + trials["stim_duration_s"] >= exit_s + window_s)
        & (trials["response_port_entry_s"] >= exit_s + window_s)
    )
    return trials[keep].reset_index(drop=True)


def load_session(
    subject: str,
    session: str,
    *,
    unit_criteria_id: int,
    stability_param_id: int | None,
    window_s: float,
) -> tuple[int, pd.DataFrame, dict[str, np.ndarray], int]:
    """Return unit count, trials, firing rates by window, and eligible count."""
    eligible = eligible_choice_trials(build_trial_table(subject, session))
    trials = select_trials(eligible, window_s)
    if trials.empty:
        raise ValueError(
            f"No trials in {subject} {session} fit three {window_s * 1000:.0f} ms "
            "windows around center exit"
        )
    units = fetch_unit_table(
        subject,
        session,
        unit_criteria_id=unit_criteria_id,
        stability_param_id=stability_param_id,
        include_metrics=False,
    )
    exit_s = trials["center_exit_s"].to_numpy(dtype=float)
    rates = {
        window: window_firing_rates(
            units["spike_times_s"],
            exit_s + offset * window_s,
            exit_s + (offset + 1) * window_s,
        )
        for window, offset in zip(WINDOWS, (-2, -1, 0), strict=True)
    }
    return len(units), trials, rates, len(eligible)


def simulate_session(
    kind: str, rng: np.random.Generator, window_s: float
) -> tuple[int, pd.DataFrame, dict[str, np.ndarray], int]:
    """Simulate Poisson units with a known gain or reformatting change."""
    n_units, n_trials = 120, 300
    category = rng.integers(0, 2, n_trials)
    # 75% correct, so all four category x choice groups have trials.
    choice = np.where(rng.random(n_trials) < 0.75, 2 * category - 1, 1 - 2 * category)
    base_hz = rng.lognormal(np.log(8), 0.6, n_units)
    axis = rng.normal(0, 1, n_units) * (rng.random(n_units) < 0.3)
    adaptation = rng.uniform(0.8, 1.0, n_units)
    gain = rng.uniform(1.2, 2.0, n_units)
    turn = rng.normal(0, 1, n_units)
    if kind == "gain":
        movement_axis = axis
    elif kind == "reformat":
        movement_axis = rng.normal(0, 1, n_units) * (rng.random(n_units) < 0.3)
    else:
        raise ValueError(f"Unknown synthetic kind {kind!r}")
    signal = 2 * category[:, None] - 1
    signal_hz = 1.0
    mean_hz = {
        "early stationary": base_hz + signal_hz * signal * axis,
        "stationary": adaptation * (base_hz + signal_hz * signal * axis),
        "movement": gain * adaptation * (base_hz + signal_hz * signal * movement_axis)
        + 4 * choice[:, None] * turn,
    }
    rates = {
        window: rng.poisson(np.clip(hz, 0.1, None) * window_s) / window_s
        for window, hz in mean_hz.items()
    }
    trials = pd.DataFrame(
        {
            "stim_category": np.where(category == 1, "high_rate", "low_rate"),
            "response": choice,
        }
    )
    return n_units, trials, rates, n_trials


def _balanced_indices(
    category: np.ndarray, choice: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    groups = [
        np.flatnonzero((category == cat) & (choice == response))
        for cat in (0, 1)
        for response in (-1, 1)
    ]
    n_per_group = min(len(group) for group in groups)
    if n_per_group < CV_FOLDS // 2:
        raise ValueError(
            "Too few category x choice trials; the least represented group has "
            f"{n_per_group}"
        )
    return np.sort(
        np.concatenate(
            [rng.choice(group, n_per_group, replace=False) for group in groups]
        )
    )


def _resample_scores(
    rates: dict[str, np.ndarray],
    labels: np.ndarray,
    indices: np.ndarray,
    seed: int,
) -> dict[tuple[str, str, str], float]:
    """Return mean held-out balanced accuracy for each train, test, scaling."""
    targets = labels[indices]
    pairs = PAIRS + TIME_PAIRS
    fold_scores: dict[tuple[str, str, str], list[float]] = {}
    for train, test in StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=seed
    ).split(indices, targets):
        scalers = {
            window: StandardScaler().fit(rates[window][indices[train]])
            for window in WINDOWS
        }
        decoders = {
            window: LogisticRegression(C=1.0, max_iter=1000).fit(
                scalers[window].transform(rates[window][indices[train]]),
                targets[train],
            )
            for window in WINDOWS
        }
        for train_window, test_window in pairs:
            test_rates = rates[test_window][indices[test]]
            for scaling, scaler in (
                ("window", scalers[test_window]),
                ("same", scalers[train_window]),
            ):
                predicted = decoders[train_window].predict(scaler.transform(test_rates))
                fold_scores.setdefault((train_window, test_window, scaling), []).append(
                    balanced_accuracy_score(targets[test], predicted)
                )
    return {key: float(np.mean(values)) for key, values in fold_scores.items()}


def decode(
    rates: dict[str, np.ndarray],
    category: np.ndarray,
    choice: np.ndarray,
    *,
    n_resamples: int,
    shuffle: bool,
    rng: np.random.Generator,
) -> tuple[dict[tuple[str, str, str], np.ndarray], int]:
    """Return accuracy over resamples for each train window, test window, scaling."""
    scores: dict[tuple[str, str, str], list[float]] = {}
    n_trials = 0
    for resample in range(n_resamples):
        labels = category.copy()
        if shuffle:
            for response in (-1, 1):
                mask = choice == response
                labels[mask] = rng.permutation(labels[mask])
        indices = _balanced_indices(labels, choice, rng)
        n_trials = len(indices)
        for key, value in _resample_scores(
            rates, labels, indices, RANDOM_SEED + resample
        ).items():
            scores.setdefault(key, []).append(value)
    return {key: np.asarray(values) for key, values in scores.items()}, n_trials


def _choice_averaged_dprime(
    rates: np.ndarray, category: np.ndarray, choice: np.ndarray
) -> np.ndarray:
    """Return each unit's high- minus low-rate d', averaged over the choices."""
    dprimes = []
    for response in (-1, 1):
        high = rates[(category == 1) & (choice == response)]
        low = rates[(category == 0) & (choice == response)]
        pooled_sd = np.sqrt((high.var(axis=0, ddof=1) + low.var(axis=0, ddof=1)) / 2)
        with np.errstate(invalid="ignore", divide="ignore"):
            dprimes.append((high.mean(axis=0) - low.mean(axis=0)) / pooled_sd)
    return np.mean(dprimes, axis=0)


def unit_selectivity(
    rates: dict[str, np.ndarray],
    category: np.ndarray,
    choice: np.ndarray,
    rng: np.random.Generator,
    n_bootstraps: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return each unit's d' and its bootstrap SE in the two state windows."""
    groups = [
        np.flatnonzero((category == cat) & (choice == response))
        for cat in (0, 1)
        for response in (-1, 1)
    ]
    boot_indices = [
        np.concatenate([rng.choice(group, len(group)) for group in groups])
        for _ in range(n_bootstraps)
    ]
    selectivity = {}
    for window in ("stationary", "movement"):
        estimate = _choice_averaged_dprime(rates[window], category, choice)
        boots = np.stack(
            [
                _choice_averaged_dprime(rates[window][idx], category[idx], choice[idx])
                for idx in boot_indices
            ]
        )
        selectivity[window] = (estimate, np.nanstd(boots, axis=0, ddof=1))
    return selectivity


def transfer_costs(
    scores: dict[tuple[str, str, str], np.ndarray],
) -> dict[str, np.ndarray]:
    """Return within minus transfer accuracy for each shift, per resample."""

    def cost(first: str, second: str) -> np.ndarray:
        within = (
            scores[(first, first, "window")] + scores[(second, second, "window")]
        ) / 2
        transfer = (
            scores[(first, second, "window")] + scores[(second, first, "window")]
        ) / 2
        return within - transfer

    return {
        "time": cost("early stationary", "stationary"),
        "time + state": cost("stationary", "movement"),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("-a", "--subject", help="Subject (needs --session)")
    source.add_argument(
        "--synthetic", choices=("gain", "reformat"), help="Simulated units"
    )
    parser.add_argument("-s", "--session")
    parser.add_argument(
        "--window-s", type=float, default=WINDOW_S, help="Window length W (s)"
    )
    parser.add_argument(
        "--unit-criteria-id", type=int, default=1, help="Unit quality criteria"
    )
    parser.add_argument(
        "--stability-param-id", type=int, help="Require passing unit stability"
    )
    parser.add_argument(
        "--quick", action="store_true", help="Use 3 resamples and 10 shuffles"
    )
    parser.add_argument("--show", action="store_true", help="Show the figure")
    parser.add_argument(
        "--no-save", action="store_true", help="Do not save the figure or results"
    )
    args = parser.parse_args()
    if args.subject and not args.session:
        parser.error("--subject needs --session")
    return args


def _summary(values: np.ndarray, null: np.ndarray | None = None) -> dict:
    summary = {"mean": float(values.mean()), "sd": float(values.std(ddof=1))}
    if null is not None:
        summary["shuffle_mean"] = float(null.mean())
        summary["p_one_sided"] = float(
            (np.sum(null >= values.mean()) + 1) / (len(null) + 1)
        )
    return summary


def main() -> None:
    args = _parse_args()
    if not args.show:
        matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    rng = np.random.default_rng(RANDOM_SEED)
    if args.synthetic:
        name = f"synthetic_{args.synthetic}"
        n_units, trials, rates, n_eligible = simulate_session(
            args.synthetic, rng, args.window_s
        )
    else:
        name = f"{args.subject}_{args.session}"
        n_units, trials, rates, n_eligible = load_session(
            args.subject,
            args.session,
            unit_criteria_id=args.unit_criteria_id,
            stability_param_id=args.stability_param_id,
            window_s=args.window_s,
        )
    category = trials["stim_category"].eq("high_rate").to_numpy(dtype=int)
    choice = trials["response"].to_numpy(dtype=int)

    n_resamples = 3 if args.quick else N_RESAMPLES
    scores, n_balanced = decode(
        rates, category, choice, n_resamples=n_resamples, shuffle=False, rng=rng
    )
    null, _ = decode(
        rates,
        category,
        choice,
        n_resamples=10 if args.quick else N_SHUFFLES,
        shuffle=True,
        rng=np.random.default_rng(RANDOM_SEED + 1),
    )
    costs = transfer_costs(scores)
    selectivity = unit_selectivity(
        rates, category, choice, np.random.default_rng(RANDOM_SEED + 2), N_BOOTSTRAPS
    )
    (stat_d, stat_se), (move_d, move_se) = selectivity.values()
    finite = np.isfinite(stat_d) & np.isfinite(move_d)
    selectivity_r = float(np.corrcoef(stat_d[finite], move_d[finite])[0, 1])

    results = {
        "source": name,
        "unit_criteria_id": None if args.synthetic else args.unit_criteria_id,
        "stability_param_id": None if args.synthetic else args.stability_param_id,
        "window_s": args.window_s,
        "n_units": n_units,
        "n_eligible_choice_trials": n_eligible,
        "n_trials_with_three_windows": len(trials),
        "n_trials_per_balanced_resample": n_balanced,
        "n_resamples": n_resamples,
        "accuracy": {
            f"{train} -> {test}, {scaling} scaling": _summary(
                values, null[(train, test, scaling)]
            )
            for (train, test, scaling), values in scores.items()
            if scaling == "window" or train != test
        },
        "transfer_cost": {shift: _summary(cost) for shift, cost in costs.items()},
        "state_minus_time_cost": {
            **_summary(costs["time + state"] - costs["time"]),
            "fraction_of_resamples_above_zero": float(
                np.mean(costs["time + state"] > costs["time"])
            ),
        },
        "unit_selectivity": {
            "n_units_with_dprime": int(finite.sum()),
            "pearson_r_stationary_vs_movement": selectivity_r,
        },
    }

    with plt.rc_context(FIGURE_STYLE):
        fig, axes = plt.subplots(
            1, 3, figsize=(10.0, 3.1), gridspec_kw={"width_ratios": (1.5, 1.1, 0.9)}
        )

        ax = axes[0]
        jitter = np.random.default_rng(RANDOM_SEED).uniform(-0.08, 0.08, n_resamples)
        for position, (train, test) in enumerate(PAIRS):
            low, high = np.percentile(null[(train, test, "window")], [2.5, 97.5])
            ax.fill_between(
                [position - 0.25, position + 0.25],
                low,
                high,
                color="0.88",
                linewidth=0,
            )
            values = scores[(train, test, "window")]
            ax.scatter(
                position - 0.1 + jitter,
                values,
                color=OBSERVED_COLOR,
                alpha=0.35,
                s=10,
                linewidths=0,
            )
            ax.plot(position - 0.1, values.mean(), "o", color=OBSERVED_COLOR, ms=6)
            if train != test:
                same = scores[(train, test, "same")]
                ax.scatter(
                    position + 0.1 + jitter,
                    same,
                    facecolors="none",
                    edgecolors=OBSERVED_COLOR,
                    alpha=0.5,
                    s=10,
                    linewidths=0.6,
                )
                ax.plot(
                    position + 0.1,
                    same.mean(),
                    "o",
                    mfc="white",
                    mec=OBSERVED_COLOR,
                    ms=6,
                )
        ax.axhline(0.5, color=REFERENCE_COLOR, linestyle=":", linewidth=0.8)
        ax.set_xticks(
            range(len(PAIRS)),
            [
                "stationary",
                "movement",
                "stationary\n→ movement",
                "movement\n→ stationary",
            ],
        )
        ax.set_xlim(-0.5, len(PAIRS) - 0.5)
        ax.set_ylabel("balanced category accuracy\n(held-out trials)")
        ax.set_xlabel("training window → test window")
        # Headroom above the data keeps the direct labels off the points.
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom, top + 0.18 * (top - bottom))
        for x, label, color in (
            (0.5, "gray: 95% of shuffles", "0.5"),
            (2.5, "open: training-window scaling", OBSERVED_COLOR),
        ):
            ax.text(
                x,
                0.98,
                label,
                ha="center",
                va="top",
                transform=ax.get_xaxis_transform(),
                color=color,
            )

        ax = axes[1]
        ax.errorbar(
            stat_d[finite],
            move_d[finite],
            xerr=stat_se[finite],
            yerr=move_se[finite],
            fmt="none",
            ecolor="0.8",
            elinewidth=0.5,
        )
        ax.scatter(
            stat_d[finite], move_d[finite], s=12, color=OBSERVED_COLOR, linewidths=0
        )
        limit = 1.05 * np.nanmax(np.abs(np.r_[stat_d[finite], move_d[finite]]))
        ax.plot(
            [-limit, limit],
            [-limit, limit],
            color=REFERENCE_COLOR,
            linestyle=":",
            linewidth=0.8,
        )
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_aspect("equal")
        ax.set_xlabel("stationary category d'")
        ax.set_ylabel("movement category d'")
        ax.text(
            0.04,
            0.96,
            f"r = {selectivity_r:.2f}\n{finite.sum()} units",
            transform=ax.transAxes,
            va="top",
        )

        ax = axes[2]
        time_cost, state_cost = costs["time"], costs["time + state"]
        for first, second in zip(time_cost, state_cost, strict=True):
            ax.plot([0, 1], [first, second], color="0.75", linewidth=0.6)
        ax.scatter(
            np.zeros(n_resamples),
            time_cost,
            color=OBSERVED_COLOR,
            s=10,
            alpha=0.35,
            linewidths=0,
        )
        ax.scatter(
            np.ones(n_resamples),
            state_cost,
            color=OBSERVED_COLOR,
            s=10,
            alpha=0.35,
            linewidths=0,
        )
        ax.plot(
            [0, 1],
            [time_cost.mean(), state_cost.mean()],
            "o-",
            color=OBSERVED_COLOR,
            ms=6,
        )
        ax.axhline(0, color=REFERENCE_COLOR, linestyle=":", linewidth=0.8)
        ax.set_xticks(
            [0, 1],
            [
                "time only\n(early →\nstationary)",
                "time + state\n(stationary ↔\nmovement)",
            ],
        )
        ax.set_xlim(-0.5, 1.5)
        ax.set_xlabel("shift between windows")
        ax.set_ylabel("transfer cost\n(within − transfer accuracy)")

        label_panels(axes, -0.25)
        fig.tight_layout(w_pad=1.5)

    print(json.dumps(results, indent=2))
    if args.show:
        plt.show()
    if not args.no_save:
        output_dir = (
            Path(os.environ.get("THESIS_FIGURE_ROOT", "figures"))
            / "cross_state_decoding"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"{name}.json").write_text(json.dumps(results, indent=2) + "\n")
        save_figure(fig, output_dir / name)
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
