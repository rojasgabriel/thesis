"""Does the V1 choice signal before the exit follow the choice or the stimulus?

Question: the choice can be decoded from V1 around the center exit
(`decoding_time_course.py`). In correct trials the choice and the stimulus
category agree (fast train: right), so a choice signal and a category signal
look the same there. In error trials they disagree. Before the exit, which one
does V1 follow?

Comparison: a decoder trained on correct trials only, then tested on error
trials. In correct trials its label is both the side and the category. On
error trials, its output is scored against the choice: an AUC above 0.5 means
it follows the choice, below 0.5 the category (the stimulus); 0.5 neither.
The same decoders are also scored on held-out correct trials (the reference).

- Windows: `WIDTH_S` wide, every `STEP_S`, aligned to the center exit
  (`CENTERS_S`). Features: each unit's rate in the window; single units and
  multi-units of unit criteria 1, no stability filter (the decoder default).
- Trials: the decoder trials (completed, not at the category boundary, with
  flashes); correct = the side of the category (fast: right).
- Training: in each of `N_RESAMPLES` resamples, equal numbers of right and
  left correct trials; `CV_FOLDS` stratified folds of them. Each fold's
  decoder (`_decoding.decoder`) is scored on its held-out correct trials and
  on all error trials.
- Score: AUC; mean over folds and resamples. Error bars: the SEM over the
  folds of each resample, averaged over resamples (fold scores share
  training trials, so this SEM is a little too small).
- Chance: the same with the correct-trial labels shuffled (`N_SHUFFLES`
  times, one resample each); the gray band is the 2.5 to 97.5 percentile of
  the error-trial AUC. The sampling unit is the trial.
- One window: neighboring windows share most of their spikes, so they are
  not independent tests. The test is on one window, `PRE_EXIT_S` (500 to
  100 ms before the exit): the error-trial AUC of the mean output of all
  decoders, against permuted error-trial choice labels (two-sided).
- A window is plotted at its center, so the window at -50 ms covers -150 to
  +50 ms and holds time after the exit.
- Sessions: `decoding_time_course.SESSIONS` (one per mouse).

The script writes `figures/choice_by_outcome/choice_by_outcome.svg` and
`results/choice_by_outcome/choice_by_outcome.json`.

    THESIS_CACHE_DIR=figures/cache uv run python -m \\
        thesis.ephys.analyses.choice_by_outcome
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thesis.ephys.analyses._decoding import decoder, load_decoding_data, window_rates
from thesis.ephys.analyses.decoding_time_course import SESSIONS
from thesis.plotting import (
    CONCEPT_COLORS,
    FIGURE_STYLE,
    MOUSE_MARKERS,
    OBSERVED_COLOR,
    REFERENCE_COLOR,
    label_panels,
    save_figure,
)

OUT = Path("results/choice_by_outcome")
WIDTH_S = 0.2
STEP_S = 0.05
CENTERS_S = np.arange(-0.6, 0.3001, STEP_S)
CV_FOLDS = 10
N_RESAMPLES = 20
N_SHUFFLES = 20
SEED = 0
ERROR_COLOR = CONCEPT_COLORS["task"]
AUC_LIMITS = (0.0, 1.02)
PRE_EXIT_S = (-0.5, -0.1)
N_PERMUTATIONS = 10000


def scores(rates, side, correct, rng, shuffle: bool, n_resamples: int) -> np.ndarray:
    """Return the AUC of each fold, (resample, fold, test set: correct, error)."""
    train = np.flatnonzero(correct)
    errors = np.flatnonzero(~correct)
    out = np.full((n_resamples, CV_FOLDS, 2), np.nan)
    for resample in range(n_resamples):
        labels = side.copy()
        if shuffle:
            labels[train] = rng.permutation(labels[train])
        right, left = (train[labels[train] == v] for v in (1, 0))
        n = min(len(right), len(left))
        picked = np.sort(
            np.concatenate(
                [
                    rng.choice(right, n, replace=False),
                    rng.choice(left, n, replace=False),
                ]
            )
        )
        folds = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEED + resample)
        for fold, (fit, test) in enumerate(folds.split(picked, labels[picked])):
            model = decoder().fit(rates[picked[fit]], labels[picked[fit]])
            held = picked[test]
            out[resample, fold, 0] = roc_auc_score(
                labels[held], model.decision_function(rates[held])
            )
            out[resample, fold, 1] = roc_auc_score(
                side[errors], model.decision_function(rates[errors])
            )
    return out


def summary(values: np.ndarray) -> tuple[float, float]:
    """Return the mean AUC and the SEM over folds (averaged over resamples)."""
    sem = values.std(1, ddof=1) / np.sqrt(values.shape[1])
    return float(values.mean()), float(sem.mean())


def pre_exit_test(rates, side, correct, rng) -> dict:
    """Score one window before the exit; test the error-trial AUC by permutation.

    The decoders of all resamples and folds (trained on correct trials) give
    each error trial a mean output. Its AUC for the choice is compared with
    the AUC after the choice labels of the error trials are permuted
    (`N_PERMUTATIONS`, two-sided around 0.5): the outputs are fixed, so the
    test does not depend on how the windows overlap.
    """
    train = np.flatnonzero(correct)
    errors = np.flatnonzero(~correct)
    outputs, held = [], []
    for resample in range(N_RESAMPLES):
        right, left = (train[side[train] == v] for v in (1, 0))
        n = min(len(right), len(left))
        picked = np.sort(
            np.concatenate(
                [
                    rng.choice(right, n, replace=False),
                    rng.choice(left, n, replace=False),
                ]
            )
        )
        folds = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEED + resample)
        for fit, test in folds.split(picked, side[picked]):
            model = decoder().fit(rates[picked[fit]], side[picked[fit]])
            outputs.append(model.decision_function(rates[errors]))
            held.append(
                roc_auc_score(
                    side[picked[test]], model.decision_function(rates[picked[test]])
                )
            )
    output = np.mean(outputs, 0)
    labels = side[errors]
    auc = roc_auc_score(labels, output)
    null = np.asarray(
        [roc_auc_score(rng.permutation(labels), output) for _ in range(N_PERMUTATIONS)]
    )
    p = (np.sum(np.abs(null - 0.5) >= abs(auc - 0.5)) + 1) / (N_PERMUTATIONS + 1)
    held = np.asarray(held).reshape(N_RESAMPLES, CV_FOLDS)
    return {
        "window_s": list(PRE_EXIT_S),
        "error_auc": float(auc),
        "error_auc_permutation_p": float(p),
        "correct_auc": float(held.mean()),
        "correct_auc_sem": float((held.std(1, ddof=1) / np.sqrt(CV_FOLDS)).mean()),
    }


def session_course(session: str) -> dict:
    subject, stamp = session.split("_", 1)
    units, trials, _ = load_decoding_data(
        subject,
        stamp,
        unit_criteria_id=1,
        stability_param_id=None,
        window_start_s=0.0,
        window_stop_s=1.0,
        include_mua=True,
    )
    exit_s = trials["center_exit_s"].to_numpy(float)
    category = trials["stim_category"].eq("high_rate").to_numpy(int)
    side = (trials["response"].to_numpy(int) == 1).astype(int)
    correct = category == side
    rng = np.random.default_rng(SEED)
    curves = {
        "correct": [],
        "correct_sem": [],
        "error": [],
        "error_sem": [],
        "chance": [],
    }
    for center in CENTERS_S:
        rates = window_rates(
            units, exit_s + center - WIDTH_S / 2, exit_s + center + WIDTH_S / 2
        )
        real = scores(rates, side, correct, rng, False, N_RESAMPLES)
        for index, name in enumerate(("correct", "error")):
            mean, sem = summary(real[..., index])
            curves[name].append(mean)
            curves[f"{name}_sem"].append(sem)
        null = [
            scores(rates, side, correct, rng, True, 1)[..., 1].mean()
            for _ in range(N_SHUFFLES)
        ]
        curves["chance"].append(np.percentile(null, [2.5, 97.5]).tolist())
    pre_exit = pre_exit_test(
        window_rates(units, exit_s + PRE_EXIT_S[0], exit_s + PRE_EXIT_S[1]),
        side,
        correct,
        rng,
    )
    print(subject, "done", pre_exit, flush=True)
    return {
        "subject": subject,
        "pre_exit": pre_exit,
        "units": len(units),
        "correct_trials": {
            "right": int((correct & (side == 1)).sum()),
            "left": int((correct & (side == 0)).sum()),
        },
        "error_trials": {
            "right": int((~correct & (side == 1)).sum()),
            "left": int((~correct & (side == 0)).sum()),
        },
        "centers_s": CENTERS_S.tolist(),
        **curves,
    }


def plot(results: list[dict]) -> Path:
    with plt.rc_context(FIGURE_STYLE):
        figure, axes = plt.subplots(
            1,
            len(results),
            figsize=(4.4 * len(results), 3.4),
            sharey=True,
            squeeze=False,
        )
        for axis, result in zip(axes[0], results, strict=True):
            x = 1000 * np.asarray(result["centers_s"])
            chance = np.asarray(result["chance"])
            axis.fill_between(x, chance[:, 0], chance[:, 1], color="0.85", lw=0)
            for name, color in (("correct", OBSERVED_COLOR), ("error", ERROR_COLOR)):
                mean = np.asarray(result[name])
                sem = np.asarray(result[f"{name}_sem"])
                axis.fill_between(
                    x, mean - sem, mean + sem, color=color, alpha=0.25, lw=0
                )
                axis.plot(
                    x,
                    mean,
                    color=color,
                    lw=1.4,
                    marker=MOUSE_MARKERS.get(result["subject"]),
                    ms=3,
                )
            axis.axhline(0.5, color=REFERENCE_COLOR, ls=":", lw=0.8)
            axis.axvline(0, color=REFERENCE_COLOR, lw=0.8)
            axis.axvspan(*(1000 * np.asarray(PRE_EXIT_S)), color="0.93", lw=0, zorder=0)
            axis.text(
                1000 * np.mean(PRE_EXIT_S),
                0.9,
                "tested window",
                ha="center",
                fontsize=6,
                color="0.5",
            )
            axis.set_ylim(AUC_LIMITS)
            axis.set_xlabel("time from center exit (ms)")
            errors = result["error_trials"]
            axis.set_title(
                f"{result['subject']}: {result['units']} units, "
                f"{errors['right'] + errors['left']} error trials\n"
                f"500-100 ms before exit: error AUC {result['pre_exit']['error_auc']:.2f}"
                f" (p = {result['pre_exit']['error_auc_permutation_p']:.2g})",
                fontsize=8,
            )
        first = axes[0, 0]
        first.set_ylabel("AUC for the choice\n(decoder trained on correct trials)")
        for y, text, color in (
            (0.97, "held-out correct trials", OBSERVED_COLOR),
            (0.89, "error trials", ERROR_COLOR),
            (0.12, "below 0.5 on error trials:\nfollows the stimulus", ERROR_COLOR),
        ):
            first.text(
                0.03,
                y,
                text,
                color=color,
                transform=first.transAxes,
                va="top",
                fontsize=7,
            )
        axes[0, -1].text(
            0.97,
            0.03,
            "shuffle",
            color="0.55",
            transform=axes[0, -1].transAxes,
            ha="right",
            fontsize=7,
        )
        figure.tight_layout()
        label_panels(axes.ravel())
        return save_figure(figure, OUT / "choice_by_outcome")


def main() -> None:
    results = [session_course(s) for s in SESSIONS]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "choice_by_outcome.json").write_text(json.dumps(results, indent=1))
    print(plot(results))


if __name__ == "__main__":
    main()
