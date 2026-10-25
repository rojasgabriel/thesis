"""Does a category readout learned while the mouse is still work while it moves?

Question: flash responses grow after the mouse leaves the center port, and the
GLM says that this is mostly one gain on all activity. If moving only scales
activity, the population axis that tells fast from slow trains while the mouse
is still should tell them apart just as well while it moves. If moving changes
the visual code, a readout trained while still should transfer worse.

Comparison: a category decoder (fast or slow, with equal numbers of left and
right choices in each category) trained on one period and tested on held-out
trials of the same period and of the other period.

- Periods: `WINDOW_S` before the center exit (still, in the port) and
  `WINDOW_S` after it (moving); the flashes go on in both.
- Features: each unit's rate in the window; single units and multi-units of
  unit criteria 1, no stability filter.
- Score: the area under the ROC curve of the decoder output on the test
  trials. A gain on all activity changes the output by a positive factor and
  a shift, which moves a fixed threshold but leaves the AUC unchanged; so a
  drop of the AUC in transfer means a change of the readout axis, not a gain.
- Folds: `CV_FOLDS` stratified folds of the trials; a transfer score trains
  on the training trials of one period and tests on the test trials of the
  other, so no trial is in both. Mean over `N_RESAMPLES` balanced resamples;
  chance from `N_SHUFFLES` label shuffles within each choice.
- PCA (`--components`, outputs with `_pcaK`): the readout is z-score, PCA
  with K components, z-score and logistic regression, all fit on the training
  trials of the training period; the test period is projected with them. Rule
  fixed before the run (2026-10-24): K = 10 is the test; K = 5 and 20 are only
  a sensitivity check. It asks whether the asymmetry without PCA (636 and 485
  features, about 100 trials) comes from a noisy in-port readout.
- Statistic: transfer minus within (each test period against the decoder
  trained on that period), over resamples: mean and 2.5–97.5 percentiles.
  The sampling unit is the trial; each mouse is one session.

The script writes `figures/decoding_transfer/decoding_transfer.svg` and
`results/decoding_transfer/decoding_transfer.json`.

    THESIS_CACHE_DIR=figures/cache uv run python -m \\
        thesis.ephys.analyses.decoding_transfer
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thesis.ephys.analyses._decoding import (
    balanced_indices,
    decoder,
    load_decoding_data,
    window_rates,
)
from thesis.plotting import (
    CONCEPT_COLORS,
    FIGURE_STYLE,
    OBSERVED_COLOR,
    REFERENCE_COLOR,
    label_panels,
    mouse_marker,
    save_figure,
)

OUT = Path("results/decoding_transfer")
SESSIONS = ("GRB006_20240821_121447", "GRB064_20261008_143359")
WINDOW_S = 0.3
CV_FOLDS = 10
N_RESAMPLES = 50
N_SHUFFLES = 50
SEED = 0
PERIODS = ("still", "moving")
# (train, test) pairs in the order of the figure.
PAIRS = (
    ("still", "still"),
    ("still", "moving"),
    ("moving", "moving"),
    ("moving", "still"),
)
AUC_LIMITS = (0.2, 1.02)  # the lowest shuffle band edge is 0.25
UNITS = "all"  # set by --units
COMPONENTS = None  # set by --components


def scores(
    rates: dict, category, choice, rng, shuffle: bool, n_resamples: int = N_RESAMPLES
) -> np.ndarray:
    """Return the AUC of each (train, test) pair per resample, (resample, pair)."""
    out = np.empty((n_resamples, len(PAIRS)))
    for resample in range(n_resamples):
        labels = category.copy()
        if shuffle:
            for side in (-1, 1):
                mask = choice == side
                labels[mask] = rng.permutation(labels[mask])
        index = balanced_indices(labels, choice, rng)
        y = labels[index]
        folds = StratifiedKFold(
            n_splits=CV_FOLDS, shuffle=True, random_state=SEED + resample
        )
        outputs = {pair: np.empty(len(index)) for pair in PAIRS}
        for train, test in folds.split(index, y):
            for period in PERIODS:
                model = make_decoder().fit(rates[period][index][train], y[train])
                for target in PERIODS:
                    outputs[(period, target)][test] = model.decision_function(
                        rates[target][index][test]
                    )
        out[resample] = [roc_auc_score(y, outputs[pair]) for pair in PAIRS]
    return out


def session_transfer(session: str) -> dict:
    subject, stamp = session.split("_", 1)
    units, trials, _ = load_decoding_data(
        subject,
        stamp,
        unit_criteria_id=1,
        stability_param_id=0 if UNITS == "sua" else None,
        window_start_s=0.0,
        window_stop_s=1.0,
        include_mua=UNITS == "all",
    )
    exit_s = trials["center_exit_s"].to_numpy(float)
    rates = {
        "still": window_rates(units, exit_s - WINDOW_S, exit_s),
        "moving": window_rates(units, exit_s, exit_s + WINDOW_S),
    }
    category = trials["stim_category"].eq("high_rate").to_numpy(int)
    choice = trials["response"].to_numpy(int)
    rng = np.random.default_rng(SEED)
    real = scores(rates, category, choice, rng, shuffle=False)
    null = scores(rates, category, choice, rng, shuffle=True, n_resamples=N_SHUFFLES)
    within = {"still": real[:, 0], "moving": real[:, 2]}
    # Each test period: transfer (trained on the other period) minus within.
    drop = {
        "test moving": real[:, 1] - within["moving"],
        "test still": real[:, 3] - within["still"],
    }
    first = trials["stim_pulse_times_s"].str[0].to_numpy(float)
    return {
        "subject": subject,
        "units": len(units),
        "trials_per_resample": int(
            len(balanced_indices(category, choice, np.random.default_rng(0)))
        ),
        "median_exit_after_first_flash_s": float(np.median(exit_s - first)),
        "pairs": [f"train {a}, test {b}" for a, b in PAIRS],
        "auc_mean": real.mean(0).tolist(),
        "auc_95": np.percentile(real, [2.5, 97.5], axis=0).T.tolist(),
        "shuffle_auc_95": np.percentile(null, [2.5, 97.5], axis=0).T.tolist(),
        "transfer_minus_within": {
            k: {
                "mean": float(v.mean()),
                "interval_95": np.percentile(v, [2.5, 97.5]).tolist(),
            }
            for k, v in drop.items()
        },
    }


def plot(results: list[dict]) -> Path:
    labels = ["still → still", "still → moving", "moving → moving", "moving → still"]
    with plt.rc_context(FIGURE_STYLE):
        figure, axes = plt.subplots(
            1, len(results), figsize=(3.4 * len(results), 3.2), sharey=True
        )
        # Category decoding: the flash color; each panel is one mouse.
        color = CONCEPT_COLORS["visual"]
        for index, (axis, data) in enumerate(zip(axes, results, strict=True)):
            x = np.arange(len(labels))
            mean = np.asarray(data["auc_mean"])
            low, high = np.asarray(data["auc_95"]).T
            null_low, null_high = np.asarray(data["shuffle_auc_95"]).T
            for xi, lo, hi in zip(x, null_low, null_high, strict=True):
                axis.fill_between(
                    [xi - 0.3, xi + 0.3], lo, hi, color="0.85", linewidth=0
                )
            axis.errorbar(
                x,
                mean,
                yerr=[mean - low, high - mean],
                fmt=mouse_marker(data["subject"], index),
                color=color,
                capsize=3,
                ms=6,
            )
            axis.axhline(0.5, color=REFERENCE_COLOR, linestyle=":", linewidth=0.8)
            axis.axvline(1.5, color="0.8", linewidth=0.6)
            axis.set_xticks(x, labels, rotation=30, ha="right")
            axis.set_ylim(*AUC_LIMITS)
            axis.text(
                0.02,
                0.97,
                data["subject"],
                color=OBSERVED_COLOR,
                transform=axis.transAxes,
                va="top",
            )
            axis.text(
                0.98,
                0.01,
                "shuffle",
                color="0.5",
                transform=axis.transAxes,
                ha="right",
                va="bottom",
            )
        axes[0].set_ylabel("category AUC (train → test)")
        label_panels(axes, x=-0.18, y=1.03)
        figure.tight_layout()
        return save_figure(figure, OUT / f"decoding_transfer{suffix()}")


def make_decoder():
    """Return the decoder; with `COMPONENTS`, z-score, PCA, z-score, logistic."""
    if COMPONENTS is None:
        return decoder()
    return make_pipeline(
        StandardScaler(),
        PCA(n_components=COMPONENTS, random_state=SEED),
        StandardScaler(),
        LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000),
    )


def suffix() -> str:
    return ("_sua" if UNITS == "sua" else "") + (
        f"_pca{COMPONENTS}" if COMPONENTS else ""
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--units",
        choices=("all", "sua"),
        default="all",
        help="all: single units and multi-units (no stability filter); sua: the stable single units of the GLM",
    )
    parser.add_argument(
        "--components",
        type=int,
        default=None,
        help="PCA components of the decoder (fit on the training trials of the training period)",
    )
    return parser.parse_args()


def main() -> None:
    global UNITS, COMPONENTS
    args = parse_args()
    UNITS, COMPONENTS = args.units, args.components
    results = [session_transfer(s) for s in SESSIONS]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"decoding_transfer{suffix()}.json").write_text(
        json.dumps(results, indent=1)
    )
    print(
        json.dumps(
            [
                {
                    k: v
                    for k, v in r.items()
                    if k in ("subject", "units", "auc_mean", "transfer_minus_within")
                }
                for r in results
            ],
            indent=1,
        )
    )
    print(plot(results))


if __name__ == "__main__":
    main()
