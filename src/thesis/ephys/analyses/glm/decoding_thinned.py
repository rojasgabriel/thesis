"""Is the clearer category signal after the exit a result of the higher rates?

Question: the stimulus category (fast or slow train) is decoded better just
after the center exit than in the port (`decoding_time_course.py`), and at
the exit a gain raises all activity. With Poisson spikes a gain g makes the
difference between the categories g times larger and the noise only sqrt(g)
times larger. Is the higher rate enough to explain the better decoding?

Comparison: the category decoder on three versions of the same trials, paired
in each resample:

- in port: the spike count of each unit in `WINDOW_S` before the exit;
- after exit: the count in `WINDOW_S` after it;
- after exit, thinned: each spike after the exit is kept with the probability
  p = (mean in-port count of the unit) / (its mean after-exit count), at most
  1. Each unit that fires more after the exit then has its in-port mean
  count. Thinning a Poisson count gives a Poisson count with p times the
  rate, so it takes away the gain and the signal over noise that the gain
  adds, and keeps all else that differs after the exit (the flashes in the
  window, the first flash after the exit, the response shape). A unit that
  fires less after the exit is not changed. p uses the means over all trials
  and no category label. Each resample has a new thinning draw.

Reading: thinned equal to in port, the higher rate explains the better
decoding. Thinned above in port, the spikes after the exit carry more about
the category per spike.

- Units: the GLM units (single units, stable) and, apart, the decoder default
  (single units and multi-units).
- Trials, windows, balance of category × choice, decoder, AUC, error bars
  (SEM across folds), and chance band: those of `decoding_without_terms.py`.
  The sampling unit is the trial; each mouse is one session.
- Sessions: `SESSIONS`, or only the mice named as arguments.

The script writes `figures/glm/category_thinned.svg` and
`results/glm/category_thinned.json`.

    THESIS_CACHE_DIR=figures/cache uv run python -m \\
        thesis.ephys.analyses.glm.decoding_thinned
"""

from __future__ import annotations

import json
import sys

import matplotlib
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thesis.ephys.analyses.glm import decoding_without_terms as base
from thesis.ephys.analyses.glm.flash_kernels import SESSIONS
from thesis.plotting import (
    FIGURE_STYLE,
    PERIOD_COLORS,
    label_panels,
    mouse_marker,
    save_figure,
)

OUT = base.OUT
VERSIONS = ("in port", "after exit", "after exit, thinned")
COLORS = (PERIOD_COLORS["in port"], *[PERIOD_COLORS["after exit"]] * 2)
# Unit set: its rates in `decoding_without_terms.session_rates`.
UNIT_SETS = {"single units": "observed", "single and multi-units": base.ALL_UNITS}
N_SHUFFLES = 20


def aucs(port, after, labels, other, rng, n_resamples: int):
    """Return the AUC, (resample, version), and the AUC of each fold, (resample, version, fold)."""
    keep = np.minimum(port.mean(0) / np.maximum(after.mean(0), 1e-12), 1.0)
    pooled = np.empty((n_resamples, len(VERSIONS)))
    by_fold = np.empty((*pooled.shape, base.CV_FOLDS))
    for resample in range(n_resamples):
        index = base.balanced_indices(labels, other, rng)
        y = labels[index]
        folds = list(
            StratifiedKFold(
                base.CV_FOLDS, shuffle=True, random_state=base.SEED + resample
            ).split(index, y)
        )
        for v, counts in enumerate((port, after, rng.binomial(after, keep))):
            x, output = counts[index], np.empty(len(index))
            for f, (train, test) in enumerate(folds):
                model = base.decoder().fit(x[train], y[train])
                output[test] = model.decision_function(x[test])
                by_fold[resample, v, f] = roc_auc_score(y[test], output[test])
            pooled[resample, v] = roc_auc_score(y, output)
    return pooled, by_fold


def session_report(session: str) -> dict:
    data = base.session_rates(session)
    labels, other = base.target_labels(data, "category")
    report = {"subject": data["subject"]}
    for name, version in UNIT_SETS.items():
        port, after = (
            np.rint(data["rates"][(p, version)] * base.WINDOW_S).astype(int)
            for p in base.PERIODS
        )
        rng = np.random.default_rng(base.SEED)
        real, by_fold = aucs(port, after, labels, other, rng, base.N_RESAMPLES)
        null = np.asarray(
            [
                aucs(
                    port,
                    after,
                    base.shuffled(labels, other, rng),
                    other,
                    rng,
                    base.NULL_RESAMPLES,
                )[0].mean(0)
                for _ in range(N_SHUFFLES)
            ]
        )
        fires = (port.mean(0) > 0) & (after.mean(0) > 0)
        report[name] = {
            "units": int(port.shape[1]),
            "units_thinned": int(np.sum(after.mean(0) > port.mean(0))),
            "median_rate_after_exit_over_in_port": float(
                np.median(after.mean(0)[fires] / port.mean(0)[fires])
            ),
            **{
                label: {
                    "auc_mean": float(real[:, v].mean()),
                    "auc_sem_across_folds": float(base.fold_sem(by_fold[:, v])),
                    "shuffle_auc_95": np.percentile(null[:, v], [2.5, 97.5]).tolist(),
                }
                for v, label in enumerate(VERSIONS)
            },
            **{
                f"thinned minus {VERSIONS[v]}": {
                    "mean": float((real[:, 2] - real[:, v]).mean()),
                    "sem_across_folds": float(
                        base.fold_sem(by_fold[:, 2] - by_fold[:, v])
                    ),
                }
                for v in (0, 1)
            },
        }
        print(data["subject"], name, "done", flush=True)
    return report


def plot(reports: list[dict]):
    with plt.rc_context(FIGURE_STYLE):
        figure, axes = plt.subplots(
            1, len(reports), figsize=(3.6 * len(reports), 3.0), sharey=True
        )
        for index, (axis, report) in enumerate(zip(axes, reports, strict=True)):
            for s, name in enumerate(UNIT_SETS):
                for v, label in enumerate(VERSIONS):
                    values = report[name][label]
                    x = v + (s - 0.5) * 0.24
                    axis.fill_between(
                        [x - 0.09, x + 0.09],
                        *values["shuffle_auc_95"],
                        color="0.85",
                        linewidth=0,
                    )
                    axis.errorbar(
                        x,
                        values["auc_mean"],
                        yerr=values["auc_sem_across_folds"],
                        fmt=mouse_marker(report["subject"], index),
                        color=COLORS[v],
                        mfc=COLORS[v] if s == 0 else "white",
                        capsize=2.5,
                        ms=5.5,
                    )
            axis.axhline(0.5, color="0.6", linestyle=":", linewidth=0.8)
            axis.set_xticks(
                range(len(VERSIONS)),
                ("in port", "after exit", "after exit,\nthinned to the\nin-port rate"),
            )
            for tick, color in zip(axis.get_xticklabels(), COLORS, strict=True):
                tick.set_color(color)
            axis.set_xlim(-0.5, len(VERSIONS) - 0.5)
            axis.set_ylim(0.3, 1.0)
            axis.text(
                0.02,
                1.02,
                f"{report['subject']}: {report['single units']['units']} single units"
                f" ({report['single and multi-units']['units']} with multi-units)",
                transform=axis.transAxes,
                va="bottom",
            )
        axes[0].text(
            0.02,
            0.02,
            "filled: single units\nopen: with multi-units\ngray: shuffle",
            color="0.4",
            transform=axes[0].transAxes,
            va="bottom",
            fontsize=7,
        )
        axes[0].set_ylabel("category AUC (held out)")
        label_panels(axes, x=-0.18, y=1.03)
        figure.tight_layout()
        return save_figure(figure, OUT / "category_thinned")


def main() -> None:
    subjects = sys.argv[1:]  # only the named mice; none: all
    reports = [
        session_report(s)
        for s in SESSIONS
        if not subjects or s.split("_")[0] in subjects
    ]
    (OUT / "category_thinned.json").write_text(json.dumps(reports, indent=1))
    print(json.dumps(reports, indent=1))
    print(plot(reports))


if __name__ == "__main__":
    main()
